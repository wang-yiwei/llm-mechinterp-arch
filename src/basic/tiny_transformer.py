#src/basic/tiny_transformer.py

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Simple multi-head self-attention module.

    Input:
        x: (batch, seq_len, d_model)
        attn_mask: optional attention mask with 1 for allowed, 0 for masked positions. Expected shape: (1, 1, seq_len, seq_len) or broadcastable to it.

    Output:
        a: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads # multiple heads by the dimension of each head
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_O = nn.Linear(d_model, d_model)


    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the MultiHeadAttention module.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attn_mask: Optional attention mask of shape (1, 1, seq_len, seq_len) or broadcastable to it
            return_attn_weights: Whether to return the attention weights

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, "Input dimension mismatch"

        # Project the input x into Q, K, V: (batch_size, seq_len, n_heads) -> (batch_size, seq_len, n_heads, d_head) -> (batch_size, n_heads, seq_len, d_head). 
        # We swap the position of the heads and the sequence length to perform the attention operation in a parallel manner.
        q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute the attention weights: (batch_size, n_heads, seq_len, seq_len)
        attn_weights = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)

        if attn_mask is not None:
            # attn_mask == 0 -> masked out position
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))
        
        # Apply the attention weights to the values: (batch_size, n_heads, seq_len, d_head) -> (batch_size, n_heads, seq_len, d_head)
        attn_weights = F.softmax(attn_weights, dim=-1) # (batch_size, n_heads, seq_len, seq_len)
        attn_weights = self.attn_dropout(attn_weights)

        # We perform the weighted sum of the values: (batch_size, n_heads, seq_len, d_head) -> (batch_size, seq_len, d_model)
        attn_weights = attn_weights @ v 

        # Concatenate the heads: (batch_size, seq_len, n_heads * d_head) -> (batch_size, seq_len, d_model)
        attn_weights = attn_weights.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Apply the output projection: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        attn_weights = self.W_O(attn_weights)

        return attn_weights