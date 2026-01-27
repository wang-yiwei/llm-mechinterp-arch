#src/modern/llama_style_moe.py

# Author: Yiwei Wang
# Date: 2026-01-20



# Information about the script:
# This script implements a MoE FFN layer with the following features:


# Importing the necessary libraries
import math
from optparse import Option
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# relative imports
from src.modern.llama_style_transformer import LlamaStyleTransformerBlock, RMSNorm, SwiGLU, LlamaStyleAttention


# --------- MoE FFN Layer ---------
class LlamaStyleMoEFFN(nn.Module):
    """
    LlamaStyle Mixture of Experts Feed-Forward Network:
    - Experts: each is a SwiGLU MLP
    - Router: linear gate from hidden state to experts logits
    - top_k: how many experts to select for each token (k=2 in Mixtral/Qwen2MoE)
    - load-balancing loss: a simple loss to balance the load on the experts

    Input:
        x: (batch_size, seq_len, d_model)

    Returns:
        y: (batch_size, seq_len, d_model)
        router_logits: (batch_size, seq_len, num_experts) if return_router_logits is True
        load_balancing_loss: (batch_size, seq_len) if return_load_balancing_loss is True
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.0,
        load_balancing_loss: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_experts == 0, "d_model must be divisible by num_experts"
        assert top_k <= num_experts, "top_k must be <= num_experts"

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Experts = full SwiGLU MLPs
        self.experts = nn.ModuleList(
            [
                SwiGLU(
                    d_model=d_model,
                    d_ff=d_ff,
                    drop_out=dropout,
                )
                for _ in range(num_experts)
            ]
        )

        # Router: hidden state -> experts logits
        self.router = nn.Linear(d_model, num_experts, bias=False)

    
    def forward(
        self,
        x: torch.Tensor,
        return_router_logits: bool = False,
    ) -> Tuple[torch.Tensor. Optional[torch.Tensor]]:
        """
        x: (batch_size, seq_len, d_model)
        """

        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, "Input dimension mismatch"

        # flatten the input tensor
        x_flat = x.view(batch_size * seq_len, d_model) # to process the tokens in parallel, N = batch_size * seq_len

        # router: logits & top-k selection over experts
        router_logits = self.router(x_flat) # (N, num_experts)
        router_probs = F.softmax(router_logits, dim=-1) # (N, num_experts)

        # top-k sparse routing
        if self.top_k < self.num_experts:
            top_k_router_values, top_k_router_idx = torch.topk(
                router_probs,
                k=self.top_k,
                dim=-1,
            ) # (N, top_k)
            mask = torch.zeros_like(router_probs) # to
            mask.scatter_(
                dim=1,
                index=top_k_router_idx,
                src=1.0, # set the top_k values to 1.0
            )
            router_probs = router_probs * mask # (N, num_experts)
            router_probs = router_probs / (router_probs.sum(dim=-1, keepdim=True) + 1e-9) # normalize the probabilities to sum to 1

        # exper outputs & weighted sum: we will compute all experts outputs in parallel and then mix them according to the router probabilities
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_flat)) # each: (N, d_model)
        expert_outputs = torch.stack(expert_outputs, dim=1) # (N, num_experts, d_model)

        # mix experts per token: (N, experts, 1) @ (N, experts, d_model) -> sum over experts dimension -> (N, d_model)
        mixed_expert_outputs = (router_probs.unsqueeze(-1) * expert_outputs).sum(dim=1) # (N, d_model)
        mixed_expert_outputs = mixed_expert_outputs.view(batch_size, seq_len, d_model) # back to the original shape

        if return_router_logits:
            return mixed_expert_outputs, router_logits.view(batch_size, seq_len, self.num_experts)
        else:
            return mixed_expert_outputs, None


# --------- Load Balancing Loss ---------

    @staticmethod
    def moe_load_balancing_loss(router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the load balancing loss for the MoE router to encourage the router to select the experts uniformly.

        router_logits: (N, num_experts)

        Returns:
            Load balancing loss: scalar
        """
        router_probs = F.softmax(router_logits, dim=-1) # (N, num_experts)
        expert_probs = router_probs.mean(dim=0) # (num_experts,): average over the batch dimension
        target_probs = torch.full_like(
            expert_probs,
            fill_value=1.0 / expert_probs.numel()
        )
        # KL divergence between the expert probabilities and the target probabilities
        loss = F.kl_div(
            expert_probs.log(),
            target_probs,
            reduction="batchmean", # average over the batch dimension
        )
        return loss


# --------- LlamaStyle MoE Transformer Block ---------

class LlamaStyleMoETransformerBlock(nn.Module):
    """
    Llama-style Transformer block where the FFN is replaced by a Mixture of Experts (MoE) FFN.

    Structure:
        x -> RMSNorm
        -> RoPE + GQA Attention
        -> residual connection
        -> RMSNorm
        -> MoE (SwiGLU experts + top-k sparse router + load-balancing loss)
        -> residual connection
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_experts: int = 16,
        top_k: int = 2,
        n_kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
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
            top_k=top_k,dropout=ffn_dropout,
            load_balancing_loss=True,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_router_logits: bool = False,
        return_load_balancing_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for the LlamaStyleMoETransformerBlock module.

        x: (batch_size, seq_len, d_model)
        """
        
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
        f, router_logits = self.moe(
            x,
            return_router_logits=return_router_logits,
        )
        x = h + f # residual connection

        if return_load_balancing_loss:
            load_balancing_loss = LlamaStyleMoEFFN.moe_load_balancing_loss(router_logits)
            return x, router_logits, load_balancing_loss
        else:
            return x, router_logits, None


# --------- LlamaStyle MoE Transformer Language Model ---------

class LlamaStyleMoETransformerLM(nn.Module):
    """
    Llama-style Transformer language model with a Mixture of Experts (MoE) FFN:
    - RoPE applied to Q and K
    - SwiGLU feed-forward network
    - Optional grouped-query attention (GQA) via n_kv_heads < n_heads
    - MoE FFN with top-k sparse router and load-balancing loss
    - Optional MoE FFN layers at specified indices, which can be chosen to be used as the FFN layer for the Transformer blocks, by passing `moe_layer_indices`
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
        num_experts: int = 16,
        top_k: int = 2,
        moe_layer_indices: Optional[List[int]] = None,
        moe_load_balancing_loss: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.rope_base = rope_base
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_layer_indices = moe_layer_indices
        self.moe_load_balancing_loss = moe_load_balancing_loss

        assert moe_layer_indices is None or all(0 <= idx < n_layers for idx in moe_layer_indices), "moe_layer_indices must be a list of integers between 0 and n_layers - 1"
        assert moe_load_balancing_loss is True, "moe_load_balancing_loss must be True"

        # Token + positional embeddings
        self.W_E = nn.Embedding(vocab_size, d_model)

        # MoE Transformer blocks
        if moe_layer_indices is None:
            moe_layer_indices = list(range(n_layers)) # use all layers as MoE layers
        blocks = []
        for layer_idx in range(n_layers):
            if layer_idx in moe_layer_indices:
                blocks.append(
                    LlamaStyleMoETransformerBlock(
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

        # Tied embedding / unembedding
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
        )
        return mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

    def forward(
        self,
        x: torch.Tensor,
        return_router_logits: bool = False,
        return_load_balancing_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        x: (batch_size, seq_len)

        Returns:
            x: (batch_size, seq_len, d_model)
            router_logits: (batch_size, seq_len, num_experts) if return_router_logits is True
            load_balancing_loss: (batch_size, seq_len) if return_load_balancing_loss is True
        """
        batch_size, seq_len = x.shape
        device = x.device
        assert seq_len <= self.max_seq_len, "Sequence length must be less than or equal to max_seq_len"

        # Embedding & positional encoding
        h = self.W_E(x) # (batch_size, seq_len, d_model)
        
        # attention mask
        attn_mask = self._causal_mask(seq_len, device)

        # Transformer routers, blocks & layer normalization
        all_router_logits = [] if return_router_logits else None
        all_load_balancing_loss = [] if return_load_balancing_loss else None

        for block in self.blocks:
            # MoE blocks return (h, router_logits, load_balancing_loss)
            if isinstance(block, LlamaStyleMoETransformerBlock):
                h, router_logits, load_balancing_loss = block(
                    h,
                    attn_mask = attn_mask,
                    return_router_logits=return_router_logits,
                    return_load_balancing_loss=return_load_balancing_loss,
                )
                if return_router_logits:
                    all_router_logits.append(router_logits)
                if return_load_balancing_loss:
                    all_load_balancing_loss.append(load_balancing_loss)
            else:
                # regular Transformer blocks return (h)
                h = block(
                    h,
                    attn_mask=attn_mask,
                )
        h = self.final_norm(h)
        logits = self.W_U(h)

        return (
            logits,
            all_router_logits if return_router_logits else None,
            all_load_balancing_loss if return_load_balancing_loss else None,
        )

if __name__ == "__main__":
    model = LlamaStyleMoETransformerLM(
        vocab_size=100,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4,
        max_seq_len=128,
        n_kv_heads=2,
        rope_base=10000.0,
        num_experts=16,
        top_k=2,
        moe_layer_indices=None,
        moe_load_balancing_loss=True,
    )