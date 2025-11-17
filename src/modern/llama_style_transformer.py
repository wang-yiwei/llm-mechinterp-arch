#src/modern/llama_style_transformer.py

# Author: Yiwei Wang
# Date: 2025-11-17



# Information about the script:
# This script implements a simple decoder-only Transformer language model with the following features:
# - RMSNorm instead of LayerNorm (pre-norm blocks).
# - No positional embedding table; instead we use RoPE (rotary positional embeddings) applied to Q and K.
# - SwiGLU FFN (gate_proj + up_proj + down_proj).
# - Optional grouped-query attention (GQA) via n_kv_heads < n_heads.
# - Still decoder-only, causal mask, and tied embeddings 𝑊𝐸=𝑊𝑈.



# Importing the necessary libraries
import math
from turtle import position
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- RMSNorm ---------

class RMSNorm(nn.Module):
    """
    RMSNorm is a normalization technique that scales the input 
    by the square root of the mean of the squares of the input.
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6
    )-> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(
        self,
        x: torch.Tensor,
    )-> torch.Tensor:
        """
        Forward pass for the RMSNorm module.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        rms_norm = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x_norm = x / rms_norm.sqrt() * self.weight
        return x_norm
    

# --------- RoPE helpers ---------

def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Builds a cache of RoPE rotation matrices (cos, sin pairs) for efficient computation.
    
    Args:
        max_seq_len: The maximum sequence length
        head_dim: The dimension of the heads
        base: The base of the RoPE
        device: The device to create the cache on
        dtype: The dtype of the cache

    Returns:
        A cache of RoPE rotation matrices (cos, sin pairs)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    device = device or torch.device("cpu")

    half_dim = head_dim // 2
    positions = torch.arange(
        max_seq_len,
        device=device,
        dtype=dtype,
    )  # (max_seq_len)
    freqs = torch.arange(
        half_dim,
        device=device,
        dtype=dtype,
    )  # (half_dim)
    inv_freqs = 1.0 / (base ** (freqs / half_dim))  # (half_dim)

    # outer product of positions and inverse frequencies
    angles = torch.einsum(
        "i,j->ij",
        positions,
        inv_freqs,
    )  # (max_seq_len, half_dim)
    cos = torch.cos(angles)  # (max_seq_len, half_dim)
    sin = torch.sin(angles)  # (max_seq_len, half_dim)
    return cos, sin


def apply_rope(
    q: torch.Tensor, # (batch, seq_len, n_heads, head_dim)
    k: torch.Tensor, # (batch, seq_len, n_heads, head_dim)
    cos: torch.Tensor, # (max_seq_len, head_dim // 2)
    sin: torch.Tensor, # (max_seq_len, head_dim // 2)
) -> torch.Tensor:
    """
    Apply RoPE to the query and key tensors.
    We treat dimensions in pairs (head_dim // 2) and apply the same RoPE rotation to each pair.
    Args:
        q: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, n_heads, head_dim)
        cos: Cosine tensor of shape (max_seq_len, head_dim // 2)
        sin: Sine tensor of shape (max_seq_len, head_dim // 2)

    Returns:
        Output tensors of shape (batch, seq_len, n_heads, head_dim)
    """
    batch_size, seq_len_q, n_heads_q, head_dim_q = q.shape
    _, seq_len_k, n_heads_k, head_dim_k = k.shape
    assert seq_len_q == seq_len_k, "Sequence lengths must match"
    assert n_heads_q == n_heads_k, "Number of heads must match"
    assert head_dim_q == head_dim_k, "Head dimensions must match"
    assert head_dim_q % 2 == 0, "Head dimension must be even for RoPE"

    half_dim = head_dim_q // 2
    cos = cos[:seq_len_q, :half_dim]
    sin = sin[:seq_len_q, :half_dim]  # (seq_len_q, half_dim)
    cos = cos[None, :, None, :, :]  # (1, 1, seq_len_q, half_dim)
    sin = sin[None, :, None, :, :]  # (1, 1, seq_len_q, half_dim)

    # split the heads into pairs and apply the RoPE rotation to each pair
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]

    # apply the RoPE rotation formula
    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd = q_even * sin + q_odd * cos
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd = k_even * sin + k_odd * cos

    # concatenate the even and odd parts back together
    q_out, k_out = torch.zeros_like(q), torch.zeros_like(k)
    q_out[..., 0::2] = q_rot_even
    q_out[..., 1::2] = q_rot_odd
    k_out[..., 0::2] = k_rot_even
    k_out[..., 1::2] = k_rot_odd
    return q_out, k_out



# --------- LLaMA/Qwen-style Attention (RoPE + optional GQA) ---------

class LlamaStyleAttention(nn.Module):
    """
    Multi-head self-attention module with:
    - RoPE applied to Q and K
    - optional grouped-query attention (GQA) by using fewer key/value heads than query heads
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert self.n_kv_heads <= self.n_heads, "n_kv_heads must be less than or equal to n_heads"
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

        # Q has full attention, K/V have grouped attention
        self.W_Q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.W_K = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.W_V = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # RoPE cache
        cos, sin = build_rope_cache(
            max_seq_len=max_seq_len,
            head_dim=self.d_head,
            base=rope_base,
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,        
    ) -> torch.Tensor:
        """
        Forward pass for the LlamaStyleAttention module.
        """

        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, "Input dimension mismatch"
        assert seq_len <= self.max_seq_len, "Sequence length must be less than or equal to max_seq_len"

        # Project the input x into Q, K, V
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_head) -> (batch_size, n_heads, seq_len, d_head). 
        # We swap the position of the heads and the sequence length to perform the attention operation in a parallel manner.
        q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K
        cos = self.rope_cos[:seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.rope_sin[:seq_len, :].to(dtype=x.dtype, device=x.device)
        q, k = apply_rope(
            q=q,
            k=k,
            cos=cos,
            sin=sin,
        )

        # Grouped attention: (batch_size, n_heads, seq_len, d_head) -> (batch_size, n_heads, seq_len, d_head)
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_rep, dim=1) # (batch_size, n_heads * n_kv_heads, seq_len, d_head)
            v = v.repeat_interleave(n_rep, dim=1) # (batch_size, n_heads * n_kv_heads, seq_len, d_head)

        # Compute the attention weights: (batch_size, n_heads, seq_len, seq_len)
        scores = q @ k.transpose(-2,-1) / math.sqrt(self.d_head)

        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask == 0, float("-inf")
            )
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # We perform the weighted sum of the values
        attn_weights = attn_weights @ v   # (batch_size, n_heads, seq_len, d_head)
        attn_weights = attn_weights.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # (batch_size, seq_len, d_model)
        attn_weights = self.W_O(attn_weights)
        attn_weights = self.proj_dropout(attn_weights)
        return attn_weights


# --------- SwiGLU Feed-Forward ---------

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network for Llama/Qwen-style MLPs:
        down_proj(up_proj(x) * silu(gate_proj(x)))

    Input:
        x: (batch, seq_len, d_model)

    Output:
        a: (batch, seq_len, d_model)
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        drop_out: float = 0.0,
    ) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff)
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(drop_out)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        gate = F.silu(self.gate_proj(x))  
        up = self.up_proj(x)
        hidden = up * gate
        hidden = self.down_proj(hidden)
        hidden = self.dropout(hidden)
        return hidden

# --------- LLaMA/Qwen-style Transformer Block ---------

class LlamaStyleTransformerBlock(nn.Module):
    """
    LLaMA/Qwen-style Transformer block with:
    - RoPE applied to Q and K
    - SwiGLU feed-forward network
    - optional grouped-query attention (GQA) via n_kv_heads < n_heads
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
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
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            drop_out=ffn_dropout,
        )
    

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the LlamaStyleTransformerBlock module.
        """
        # self-attention
        h = x
        x_norm = self.ln1(x)
        a = self.attn(x_norm, attn_mask)
        x = h + a  # residual connection

        # feed-forward
        h = x
        x_norm = self.ln2(x)
        f = self.ffn(x_norm)
        x = h + f  # residual connection
        return x



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
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        max_seq_len: int = 2048,
        n_kv_heads: Optional[int] = None,
        rope_base: float = 10000.0,
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

        # Token + positional embeddings
        self.W_E = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
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
                for _ in range(n_layers)
            ]
        )

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
        Create a causal mask for the attention mechanism.

        Args:
            seq_len: The length of the sequence
            device: The device to create the mask on

        Returns:
            A causal mask of shape (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(
            seq_len,
            seq_len,
            device=device,
            dtype=torch.bool,
        ))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the LlamaStyleTransformerLM module.

        Args:
            x: Input tensor of shape (batch, seq_len)

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device
        assert seq_len <= self.max_seq_len, "Sequence length must be less than or equal to max_seq_len"

        # Embedding & positional encoding
        x_emb = self.W_E(x) # (batch, seq_len, d_model)
        
        # Causal mask shared across all heads
        attn_mask = self._causal_mask(seq_len, device)  # (1, 1, seq_len, seq_len)
        
        # Transformer blocks & layer normalization
        for block in self.blocks:
            h = block(h, attn_mask)    
        
        # final layer normalization
        h = self.final_norm(h) # (batch, seq_len, d_model) -> final hidden state

        # Unembedding
        logits = self.W_U(h) # (batch, seq_len, vocab_size)

        return logits


if __name__ == "__main__":
    model = LlamaStyleTransformerLM(
        vocab_size=100,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4,
        max_seq_len=128,
        n_kv_heads=2,
    )