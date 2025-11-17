#src/basic/tiny_transformer.py

# Author: Yiwei Wang
# Date: 2025-11-14

# Importing the necessary libraries
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
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

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


class FeedForward(nn.Module):
    """
    Simple feed-forward network with GELU nonlinearity.

    Input:
        x: (batch, seq_len, d_model)

    Output:
        a: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.W_in = nn.Linear(d_model, d_ff)
        self.W_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FeedForward module.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        return self.W_out(F.gelu(self.W_in(x)))

class TransformerBlock(nn.Module):
    """
    Simple Transformer block with multi-head self-attention and feed-forward network.

    Input:
        x: (batch, seq_len, d_model)

    Output:
        a: (batch, seq_len, d_model)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the TransformerBlock module.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attn_mask: Optional attention mask of shape (1, 1, seq_len, seq_len) or broadcastable to it

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        
        # Self-attention
        h = x
        x_norm = self.ln1(x)
        a = self.attn(x_norm, attn_mask)
        x = h + a  # residual connection

        # Feed-forward
        h = x
        x_norm = self.ln2(x)
        f = self.ffn(x_norm)
        x = h + f  # residual connection

        return x


class TinyTransformerLM(nn.Module):
    """
    Simple decoder-only Transformer language model.

    Input:
        x: (batch, seq_len)

    Output:
        logits: (batch, seq_len, vocab_size)

    Components:
        - token + positional embeddings W_E
        - multi-head self-attention W_Q, W_K, W_V, W_O
        - LayerNorm + residual connections ln1, ln2
        - 2-layer FFN with GELU W_in, W_out
        - tied embedding / unembedding W_U
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