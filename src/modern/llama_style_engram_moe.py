# src/modern/llama_style_engram_moe.py

# Author: Yiwei Wang
# Date: 2026-03-02

# Information about the script:
# This script implements a Llama-style Transformer language model with Engram (injected before attention) and Mixture of Experts (MoE) FFN (injected after attention).

# Importing the necessary libraries
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# relative imports
from src.modern.llama_style_transformer import (
    RMSNorm,
    LlamaStyleAttention,
    LlamaStyleTransformerBlock
)

from src.modern.llama_style_moe import LlamaStyleMoEFFN, all_load_balancing_loss, all_router_logits
from src.modern.engram import (
    EngramConfig,
    EngramLookup,
    EngramKVProjector,
    EngramContextGate,
    EngramFusion,
)


# --------- Engram + MoE FFN Layer ---------
class LlamaStyleEngramMoETransformerBlock(nn.Module):
    """
    x -> (optional Engram injection) -> attention -> MoE FFN
    Engram injection:
      mem_flat = lookup(input_ids)
      key, value = KV(mem_flat)
      gate = Gate(hidden_states, key)
      delta = Fusion(gate, value)
      x = x + delta
    """
    def __init__(
        self,
        layer_id: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        n_kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        engram_config: Optional[EngramConfig] = None,
        use_engram: bool = True,
    ) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = LlamaStyleAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            attn_dropout=attn_dropout,
            proj_dropout=attn_dropout,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        self.moe = LlamaStyleMoEFFN(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            dropout=ffn_dropout,
        )
        self.use_engram = bool(
            use_engram and (engram_config is not None) and (layer_id in engram_config.layer_ids)
        )
        self.engram_config = engram_config
        if use_engram:
            self.lookup = EngramLookup(
                cfg=engram_config,
                layer_id=layer_id,
            )
            self.kv_projector = EngramKVProjector(
                mem_dim=self.lookup.mem_dim,
                d_model=d_model,
                bias=False,
            )
            self.context_gate = EngramContextGate(
                d_model=d_model,
                use_signed_sqrt=engram_config.gate_signed_sqrt
            )
            self.engram_fusion = EngramFusion(
                d_model=d_model,
                use_conv=engram_config.use_conv,
                conv_kernel_size=engram_config.conv_kernel_size,
            )
        else:
            self.lookup = None
            self.kv_projector = None
            self.context_gate = None
            self.engram_fusion = None

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_router_logits: bool = False,
        return_load_balancing_loss: bool = False,
        return_engram_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for the LlamaStyleEngramMoETransformerBlock module.

        x: (batch_size, seq_len, d_model)
        input_ids: (batch_size, seq_len)
        attn_mask: (batch_size, seq_len)
        return_router_logits: bool = False
        return_load_balancing_loss: bool = False
        return_engram_aux: bool = False

        Returns:
            x: (batch_size, seq_len, d_model)
            router_logits: (batch_size, seq_len, num_experts) if return_router_logits is True
            load_balancing_loss: (batch_size, seq_len) if return_load_balancing_loss is True
            engram_aux: Dict[str, torch.Tensor] if return_engram_aux is True
        """
        engram_aux = None

        # Engram injection
        if self.use_engram:
            mem_flat, hash_ids = self.lookup(input_ids) # (batch_size, seq_len, mem_dim), (batch_size, seq_len, head_total_size)
            key, value = self.kv_projector(mem_flat) # (batch_size, seq_len, d_model), (batch_size, seq_len, d_model)
            gate = self.context_gate(x, key) # (batch_size, seq_len, 1)
            delta = self.engram_fusion(gate, value) # (batch_size, seq_len, d_model)
            x = x + delta # residual connection
            if return_engram_aux:
                engram_aux = {
                    "gate": gate,
                    "hash_ids": hash_ids,
                }
        
        # self attention
        h = x
        x_norm = self.ln1(x)
        a = self.attn(
            x_norm,
            attn_mask,
        )
        x = h + a # residual connection

        # MoE FFN
        h = x
        x_norm = self.ln2(x)
        f, router_logits = self.moe(
            x_norm,
            return_router_logits=return_router_logits,
        )
        x = h + f # residual connection

        load_balancing_loss = None
        if return_load_balancing_loss and router_logits is not None:
            load_balancing_loss = self.moe.moe_load_balancing_loss(router_logits)
        
        if not return_router_logits:
            router_logits = None
        
        return x, router_logits, load_balancing_loss, engram_aux



# --------- Engram + MoE FFN Transformer Language Model ---------
class LlamaStyleEngramMoETransformerLM(nn.Module):
    """
    Llama-style Transformer language model with Engram (injected before attention) and Mixture of Experts (MoE) FFN (injected after attention) blocks.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_layers: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        max_seq_len: int = 2048,
        n_kv_heads: Optional[int] = None,
        rope_base: float = 10000.0,
        # MoE FFN
        num_experts: int = 16,
        top_k: int = 2,
        moe_layer_indices: Optional[List[int]] = None,
        # Engram
        engram_config: Optional[EngramConfig] = None,
        engram_layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        if moe_layer_indices is None:
            moe_layer_indices = list(range(n_layers))
        self.moe_layer_indices = set(moe_layer_indices)

        if engram_layer_indices is None:
            engram_layer_indices = []
        self.engram_layer_indices = set(engram_layer_indices)

        # token + positional embeddings
        self.W_E = nn.Embedding(vocab_size, d_model)

        # engram + moe ffn blocks
        blocks = []
        for layer_idx in range(n_layers):
            if layer_idx in self.moe_layer_indices:
                use_engram = (engram_config is not None) and (layer_idx in engram_layer_indices)
                blocks.append(
                    LlamaStyleEngramMoETransformerBlock(
                        layer_id=layer_idx,
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        num_experts=num_experts,
                        top_k=top_k,
                        n_kv_heads=n_kv_heads,
                        attn_dropout=attn_dropout,
                        ffn_dropout=ffn_dropout,
                        max_seq_len=max_seq_len,
                        rope_base=rope_base,
                        engram_config=engram_config,
                        use_engram=use_engram,
                    )
                )
            else:
                blocks.append(
                    LlamaStyleTransformerBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        n_kv_heads=n_kv_heads,
                        attn_dropout=attn_dropout,
                        ffn_dropout=ffn_dropout,
                        max_seq_len=max_seq_len,
                        rope_base=rope_base,
                    )
                )
        
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(d_model)

        # tied embedding / unembedding
        self.W_U = nn.Linear(d_model, vocab_size, bias=False)
        self.W_U.weight = self.W_E.weight # tie the weights of the embedding and unembedding layers
    
    def _causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a causal mask for the attention mechanism. Torch.tril creates a lower triangular matrix with ones on the diagonal and zeros above the diagonal. This is used to mask the attention scores to ensure that the model only attends to previous tokens.

        Args:
            seq_len: The length of the sequence
            device: The device to create the mask on

        Returns:
            A causal mask of shape (1, 1, seq_len, seq_len) covering the first two dimensions.
        """
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool,
            )
        ) # (seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len) covering the first two dimensions.

    def forward(
        self,
        input_ids: torch.Tensor, # (batch_size, seq_len)
        return_router_logits: bool = False,
        return_load_balancing_loss: bool = False,
        return_engram_aux: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """
        Forward pass for the LlamaStyleEngramMoETransformerLM module.

        input_ids: (batch_size, seq_len)
        return_router_logits: bool = False
        return_load_balancing_loss: bool = False
        return_engram_aux: bool = False

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        assert seq_len <= self.max_seq_len, "Sequence length must be less than or equal to max_seq_len"

        # embedding & positional encoding
        h = self.W_E(input_ids) # (batch_size, seq_len, d_model)

        # attention mask
        attn_mask = self._causal_mask(seq_len, device) # (1, 1, seq_len, seq_len)

        # engram + moe ffn blocks
        all_router_logits = [] if return_router_logits else None
        all_load_balancing_loss = [] if return_load_balancing_loss else None
        all_engram_aux = [] if return_engram_aux else None

        for block in self.blocks:
            if isinstance(block, LlamaStyleEngramMoETransformerBlock):
                h, router_logits, load_balancing_loss, engram_aux = block(
                    h,
                    input_ids=input_ids,
                    attn_mask=attn_mask,
                    return_router_logits=return_router_logits,
                    return_load_balancing_loss=return_load_balancing_loss,
                    return_engram_aux=return_engram_aux,
                )
                if return_router_logits and router_logits is not None:
                    all_router_logits.append(router_logits)
                if return_load_balancing_loss and load_balancing_loss is not None:
                    all_load_balancing_loss.append(load_balancing_loss)
                if return_engram_aux and engram_aux is not None:
                    all_engram_aux.append(engram_aux)
            else:
                # regular Transformer blocks return (h)
                h = block(
                    h,
                    attn_mask=attn_mask,
                )
        h = self.final_norm(h)
        logits = self.W_U(h)

        return logits, all_router_logits, all_load_balancing_loss, all_engram_aux