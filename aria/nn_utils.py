"""Shared neural network utilities: RoPE, RMSNorm, SwiGLU FFN.

Kept intentionally small and dependency-free. Both LSA and baseline transformer
use these identical components so any performance difference is attributable to
the attention mechanism, not peripheral choices.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich 2019).

    Drops the mean-subtraction from LayerNorm, keeping only RMS rescaling.
    ~10-15% faster than LayerNorm with equivalent empirical performance.
    Standard in LLaMA, Mistral, Gemma, DeepSeek.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer 2020).

    FFN(x) = (W1 x) * SiLU(W2 x); final projection W3.
    Inner dim conventionally (2/3) * 4 * d_model so the three-matrix variant
    has ~same parameter count as a standard two-matrix 4d FFN.

    Standard in LLaMA, Mistral, PaLM, Gemma, all major 2024-25 LLMs.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


def precompute_rope(dim: int, max_seq_len: int, base: float = 10000.0,
                    device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos / sin tables.

    dim: the head dimension (must be even)
    Returns: cos, sin of shape (max_seq_len, dim/2)
    """
    assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q and k.

    q, k: (B, T, H, Dh)  — Dh must be even
    cos, sin: (T, Dh/2)  — sliced to current sequence length

    Rotation is applied pairwise on adjacent dimensions:
        (x_even, x_odd) -> (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)
    """
    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # cos/sin broadcast: (1, T, 1, Dh/2)
        cos_b = cos[None, :, None, :]
        sin_b = sin[None, :, None, :]
        rx1 = x1 * cos_b - x2 * sin_b
        rx2 = x1 * sin_b + x2 * cos_b
        # Interleave back: (..., Dh)
        out = torch.stack([rx1, rx2], dim=-1).flatten(-2)
        return out

    return rotate(q), rotate(k)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
