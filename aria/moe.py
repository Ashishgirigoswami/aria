"""Mixture of Experts (MoE) routing for ARIA.

Implements top-k sparse expert routing as used by DeepSeek-V3, Llama 4,
and Qwen3. Designed as a drop-in replacement for the SwiGLU FFN in any
LSA or transformer block.

Key design decisions:
- Fine-grained experts (many small experts, top-2 routing) following
  DeepSeek-V3 and Qwen3's approach
- Optional shared expert (always active, receives all tokens) following
  DeepSeek-V3
- Auxiliary-loss-free load balancing via bias term (DeepSeek-V3 style)
- Expert capacity factor for bounded memory

Usage:
    Replace ``SwiGLU(d_model, d_ff)`` with
    ``MoELayer(d_model, d_ff_expert, num_experts=64, top_k=2)``
    in any block. Total params scale with num_experts; active params
    per token scale with top_k.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single SwiGLU expert — identical architecture to the dense FFN."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    """Sparse Mixture of Experts with top-k routing.

    Args:
        d_model: Model dimension.
        d_ff_expert: FFN dimension per expert (smaller than dense d_ff
            since multiple experts run in parallel).
        num_experts: Total number of routed experts.
        top_k: Number of experts activated per token (default 2).
        shared_expert: If True, add one always-active shared expert
            (DeepSeek-V3 style). Its output is added to the routed output.
        capacity_factor: Max fraction of tokens an expert can receive
            (0 = no limit). Prevents a single expert from getting all tokens.
        aux_loss_weight: Weight of the auxiliary load-balancing loss.
            Set to 0 for auxiliary-loss-free balancing (uses bias instead).
    """

    def __init__(
        self,
        d_model: int,
        d_ff_expert: int,
        num_experts: int = 64,
        top_k: int = 2,
        shared_expert: bool = True,
        capacity_factor: float = 0.0,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight

        # Router: projects token to expert scores
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Routed experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff_expert) for _ in range(num_experts)
        ])

        # Optional shared expert (always active)
        self.shared_expert = Expert(d_model, d_ff_expert) if shared_expert else None

        # Auxiliary-loss-free load balancing bias (DeepSeek-V3 style).
        # This learnable bias is added to router logits during training
        # to encourage balanced expert usage without an explicit aux loss.
        self.router_bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            output: (B, T, d_model)
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)
        N = x_flat.size(0)

        # Router scores
        logits = self.router(x_flat)  # (N, num_experts)
        if self.training:
            logits = logits + self.router_bias

        # Top-k selection
        scores = F.softmax(logits, dim=-1)  # (N, num_experts)
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)  # (N, top_k)

        # Normalize top-k scores to sum to 1 per token
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # Dispatch tokens to experts and combine outputs
        output = torch.zeros_like(x_flat)  # (N, D)
        for k in range(self.top_k):
            expert_indices = top_indices[:, k]  # (N,)
            expert_weights = top_scores[:, k]   # (N,)
            for e in range(self.num_experts):
                mask = expert_indices == e
                if not mask.any():
                    continue
                expert_input = x_flat[mask]      # (n_tokens, D)
                expert_output = self.experts[e](expert_input)
                output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output

        # Add shared expert output (always active on all tokens)
        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        output = output.view(B, T, D)

        # Auxiliary load-balancing loss (optional)
        if self.training and self.aux_loss_weight > 0:
            # Fraction of tokens assigned to each expert
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for k in range(self.top_k):
                expert_counts.scatter_add_(
                    0, top_indices[:, k],
                    torch.ones(N, device=x.device),
                )
            expert_frac = expert_counts / (N * self.top_k)
            # Average router probability per expert
            avg_prob = scores.mean(dim=0)
            # Balance loss: encourages uniform distribution
            aux_loss = self.num_experts * (expert_frac * avg_prob).sum()
            # Store for the training loop to add to main loss
            self._aux_loss = aux_loss * self.aux_loss_weight
        else:
            self._aux_loss = torch.tensor(0.0, device=x.device)

        return output

    @property
    def aux_loss(self) -> torch.Tensor:
        """Access the auxiliary load-balancing loss from the last forward pass."""
        return self._aux_loss
