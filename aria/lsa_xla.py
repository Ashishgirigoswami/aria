"""Layered State Attention for torch_xla (TPU) backends.

Functionally identical to ``aria.lsa`` but with two differences:

1. The SSM scan is a plain Python ``for`` loop with no ``@torch.jit.script``
   decorator. ``torch.jit.script`` traces into TorchScript IR which conflicts
   with XLA's lazy-tensor tracing; XLA compiles the whole graph into a single
   HLO program anyway, so wrapping the loop in jit.script gives nothing and
   breaks compilation on TPU.

2. Everything else (LSAAttention, LSABlock, LSALanguageModel) is re-exported
   unchanged from ``aria.lsa``. Only the scan is overridden.

Use this module from TPU training scripts. The GPU path still uses
``aria.lsa`` with the JIT-compiled scan.
"""

from __future__ import annotations

import torch

from . import lsa as _lsa_gpu
from .lsa import (  # re-export for import convenience on TPU
    LSAAttention,
    LSABlock,
    LSALanguageModel,
)


def ssm_scan_xla(A: torch.Tensor, Bg: torch.Tensor,
                 state_input: torch.Tensor) -> torch.Tensor:
    """Sequential causal SSM scan, plain Python, XLA-friendly.

    Args:
        A: (B, T, d_state) per-token decay gate in [0, 1].
        Bg: (B, T, d_state) per-token write gate in [-1, 1].
        state_input: (B, T, d_state) what to write at each step.

    Returns:
        states: (B, T, d_state) per-position causal states.
    """
    B, T, D = A.shape
    s = torch.zeros(B, D, device=A.device, dtype=A.dtype)
    outputs = []
    for t in range(T):
        s = A[:, t] * s + Bg[:, t] * state_input[:, t]
        outputs.append(s)
    return torch.stack(outputs, dim=1)


# Monkey-patch the GPU module's scan symbol at import time so that any
# ``LSAAttention.forward`` path reaches the XLA-safe scan instead of the
# jit.script'd one. This is imported by scripts/train_xla.py before any
# LSA model is instantiated.
_lsa_gpu.ssm_scan_jit = ssm_scan_xla

__all__ = [
    "ssm_scan_xla",
    "LSAAttention",
    "LSABlock",
    "LSALanguageModel",
]
