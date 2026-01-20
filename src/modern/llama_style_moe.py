#src/modern/llama_style_moe.py

# Author: Yiwei Wang
# Date: 2026-01-20



# Information about the script:
# This script implements a MoE FFN layer with the following features:


# Importing the necessary libraries
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# relative imports
from src.modern.llama_style_transformer import RMSNorm, SwiGLU, LlamaStyleAttention


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

    