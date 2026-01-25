#src/modern/llama_style_moe.py

# Author: Yiwei Wang
# Date: 2026-01-20



# Information about the script:
# This script implements a MoE FFN layer with the following features:


# Importing the necessary libraries
import math
from typing import Optional, List, Tuple

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
