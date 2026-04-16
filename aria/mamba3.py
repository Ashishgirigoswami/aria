"""Mamba-3 SSM block — pure PyTorch implementation for ARIA swap.

Replaces the Gated DeltaNet slot in LSA v2's interleaved hybrid. Follows
the ICLR 2026 paper (OpenReview HwCvaJOiCj) and Tri Dao's blog part 2.

Recurrence (SISO):
    alpha_t = exp(dt_t * A_t)
    beta_t  = (1 - lambda_t) * dt_t * exp(dt_t * A_t)
    gamma_t = lambda_t * dt_t
    h_t     = alpha_t * h_{t-1} + beta_t * (B_{t-1} x_{t-1}) + gamma_t * (B_t x_t)
    y_t     = C_t^T h_t

When lambda_t = 1 this collapses to Mamba-2 exponential-Euler. The
beta_t * B_{t-1}*x_{t-1} term is a data-dependent size-2 conv that
makes Mamba/Mamba-2's explicit conv1d redundant.

Complex-state trick (state-tracking / parity): instead of
torch.complex64 state, apply a cumulative RoPE angle to B and C before
the scan — mathematically equivalent, XLA-friendly.

Shapes per block:
    x : (B, T, D)           input
    B : (B, T, H, N)        selective state projection
    C : (B, T, H, N)        selective readout projection
    dt, lambda_ : (B, T, H) selective scalars (dt>0 via softplus; lambda in [0,1])
    h : (B, H, P, N)        recurrent state (P = headdim, N = d_state)

MIMO extension (rank R): x_mat in R^{P x R}, B/C in R^{N x R}, state
(P, N) shared. Not implemented yet — SISO first.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import RMSNorm


def apply_rope_cumangle(
    z: torch.Tensor,          # (B, T, H, N)  — either B or C projection
    cum_angle: torch.Tensor,  # (B, T, H)     — cumulative dt*theta
) -> torch.Tensor:
    """Apply cumulative-angle RoPE to pairs of channels in the last dim.

    z is viewed as (..., N/2, 2) and each pair is rotated by cum_angle[t].
    This is the real-valued equivalent of the complex-state formulation —
    no torch.complex needed.
    """
    B, T, H, N = z.shape
    assert N % 2 == 0, "state dim must be even for RoPE pairing"
    z_pair = z.view(B, T, H, N // 2, 2)
    cos = torch.cos(cum_angle)[..., None]  # (B, T, H, 1)
    sin = torch.sin(cum_angle)[..., None]
    z0 = z_pair[..., 0]
    z1 = z_pair[..., 1]
    out0 = z0 * cos - z1 * sin
    out1 = z0 * sin + z1 * cos
    return torch.stack([out0, out1], dim=-1).view(B, T, H, N)


def mamba3_scan_ref(
    x: torch.Tensor,         # (B, T, H, P)        input per head
    dt: torch.Tensor,        # (B, T, H)           step size (softplus-positive)
    A: torch.Tensor,          # (H,)                per-head decay (fixed, log-space)
    B_proj: torch.Tensor,    # (B, T, H, N)
    C_proj: torch.Tensor,    # (B, T, H, N)
    lambda_: torch.Tensor,   # (B, T, H)           trap coefficient in [0,1]
) -> torch.Tensor:           # (B, T, H, P)        output
    """Reference Python-loop Mamba-3 scan. Slow on large T, used for tests.

    State shape: (B, H, P, N).
    """
    Bsz, T, H, P = x.shape
    N = B_proj.size(-1)
    dtype = x.dtype
    device = x.device

    # Decay in log-space: alpha_t = exp(dt_t * A_log.neg()) — matches Mamba-2.
    A_neg = -torch.exp(A)  # (H,) negative reals
    h = torch.zeros(Bsz, H, P, N, dtype=dtype, device=device)
    y = torch.empty_like(x)

    # B_{t-1} x_{t-1} buffer (zero at t=0)
    Bx_prev = torch.zeros(Bsz, H, P, N, dtype=dtype, device=device)

    for t in range(T):
        dt_t = dt[:, t]                           # (B, H)
        alpha_t = torch.exp(dt_t * A_neg[None])   # (B, H)
        lam_t = lambda_[:, t]                      # (B, H)
        one_minus_lam = 1.0 - lam_t
        beta_t = one_minus_lam * dt_t * alpha_t   # (B, H)
        gamma_t = lam_t * dt_t                    # (B, H)

        # Current-step rank-1 write: B_t x_t → (B, H, P, N)
        Bx_t = x[:, t].unsqueeze(-1) * B_proj[:, t].unsqueeze(-2)

        h = (alpha_t[..., None, None] * h
             + beta_t[..., None, None] * Bx_prev
             + gamma_t[..., None, None] * Bx_t)

        # Readout y_t = (h · C_t).sum(N) → (B, H, P)
        y[:, t] = (h * C_proj[:, t].unsqueeze(-2)).sum(-1)

        Bx_prev = Bx_t

    return y


class Mamba3Block(nn.Module):
    """Single Mamba-3 SSM block (SISO). MIMO is a follow-up.

    Matches the API of ARIA's existing GatedDeltaNet block so the swap
    is a drop-in replacement inside LSAv2LanguageModel.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        n_heads: int = 8,
        headdim: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        theta_base: float = 10000.0,
        use_rope_state: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads
        self.headdim = headdim
        self.use_rope_state = use_rope_state

        d_inner = n_heads * headdim
        self.in_proj = nn.Linear(d_model, d_inner, bias=False)

        # Selective projections: B, C, dt, lambda, theta are data-dependent.
        # One linear from x → (B, C, dt, lambda, theta), then split.
        self.proj_selective = nn.Linear(
            d_model,
            2 * n_heads * d_state + 3 * n_heads,
            bias=False,
        )

        # Per-head A (fixed-per-head decay), stored in log-space.
        A_init = torch.arange(1, n_heads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A_init))

        # dt_bias so that softplus(dt_bias) ~ uniform in [dt_min, dt_max].
        # Draw target dt in log-space, then inverse-softplus it.
        log_u = torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        dt_target = torch.exp(log_u).clamp(min=1e-4)
        # inv_softplus(y) = y + log(-expm1(-y))  — valid only for y > 0
        self.dt_bias = nn.Parameter(dt_target + torch.log(-torch.expm1(-dt_target)))

        self.theta_base = theta_base
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.rms_norm_out = RMSNorm(d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)."""
        Bsz, T, _ = x.shape
        H = self.n_heads
        P = self.headdim
        N = self.d_state

        u = self.in_proj(x).view(Bsz, T, H, P)  # (B, T, H, P)

        sel = self.proj_selective(x)
        B_proj, C_proj, scalars = torch.split(
            sel,
            [H * N, H * N, 3 * H],
            dim=-1,
        )
        B_proj = B_proj.view(Bsz, T, H, N)
        C_proj = C_proj.view(Bsz, T, H, N)
        dt_raw, lam_raw, theta_raw = torch.split(
            scalars.view(Bsz, T, 3, H), 1, dim=2
        )
        dt = F.softplus(dt_raw.squeeze(2) + self.dt_bias)           # (B, T, H)
        lambda_ = torch.sigmoid(lam_raw.squeeze(2))                  # (B, T, H) in [0,1]
        theta = theta_raw.squeeze(2)                                  # (B, T, H)

        if self.use_rope_state:
            # Cumulative angle = cumsum(dt * theta, t)
            cum_angle = torch.cumsum(dt * theta, dim=1)  # (B, T, H)
            B_proj = apply_rope_cumangle(B_proj, cum_angle)
            C_proj = apply_rope_cumangle(C_proj, cum_angle)

        y = mamba3_scan_ref(u, dt, self.A_log, B_proj, C_proj, lambda_)
        y = y.reshape(Bsz, T, H * P)
        y = self.rms_norm_out(y)
        return self.out_proj(y)


__all__ = ["Mamba3Block", "mamba3_scan_ref", "apply_rope_cumangle"]
