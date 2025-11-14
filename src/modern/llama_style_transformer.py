#src/modern/llama_style_transformer.py

# Author: Yiwei Huang
# Date: 2025-11-14



# Information about the script:
# This script implements a simple decoder-only Transformer language model with the following features:
# - RMSNorm instead of LayerNorm (pre-norm blocks).
# - No positional embedding table; instead we use RoPE (rotary positional embeddings) applied to Q and K.
# - SwiGLU FFN (gate_proj + up_proj + down_proj).
# - Optional grouped-query attention (GQA) via n_kv_heads < n_heads.
# - Still decoder-only, causal mask, and tied embeddings 𝑊𝐸=𝑊𝑈.



# Importing the necessary libraries
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- RMSNorm ---------





# --------- RoPE helpers ---------





# --------- LLaMA/Qwen-style Attention (RoPE + optional GQA) ---------





# --------- SwiGLU Feed-Forward ---------





# --------- LLaMA/Qwen-style Transformer Block ---------





# --------- Full LLaMA/Qwen-style Decoder-only LM ---------

class LlamaStyleTransformerLM(nn.Module):
    """

    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_layers: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # Token + positional embeddings
        self.W_E = nn.Embedding(vocab_size, d_model)
        self.W_pos = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)

        # Tied embedding / unembedding
        self.W_U = nn.Linear(d_model, vocab_size, bias=False)
        self.W_U.weight = self.W_E.weight # tie the weights of the embedding and unembedding layers

    def _causal_mask(
            self,
            seq_len: int,
            device: torch.device,
        ) -> torch.Tensor:
        """
        Create a causal mask for the attention mechanism.

        Args:
            seq_len: The length of the sequence
            device: The device to create the mask on

        Returns:
            A causal mask of shape (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(
            seq_len, seq_len, device=device
        ))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the TinyTransformerLM module.

        Args:
            x: Input tensor of shape (batch, seq_len)
            pos_ids: Optional positional IDs of shape (batch, seq_len)

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device
        assert seq_len <= self.max_seq_len, "Sequence length must be less than or equal to max_seq_len"

        # Embedding & positional encoding
        x_emb = self.W_E(x) # (batch, seq_len, d_model)
        if pos_ids is None:
            pos_ids = torch.arange(
                seq_len, device=device
            ).unsqueeze(0).expand(batch_size, seq_len)
        else:
            assert pos_ids.shape == (batch_size, seq_len), "Positional IDs must be of shape (batch, seq_len)"

        pos_emb = self.W_pos(pos_ids) # (batch, seq_len, d_model)
        h = x_emb + pos_emb # (batch, seq_len, d_model) -> initial hidden state

        # Causal mask
        attn_mask = self._causal_mask(seq_len, device)  # (1, 1, seq_len, seq_len)
        
        # Transformer blocks & layer normalization
        for block in self.blocks:
            h = block(h, attn_mask)    
        h = self.ln_final(h) # (batch, seq_len, d_model) -> final hidden state

        # Unembedding
        logits = self.W_U(h) # (batch, seq_len, vocab_size)
        return logits


if __name__ == "__main__":
    model = TinyTransformerLM(
        vocab_size=100,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4,
        dropout=0.0,
    )