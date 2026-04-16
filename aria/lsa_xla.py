"""Layered State Attention for torch_xla (TPU) backends.

Uses ``torch_xla.experimental.scan`` to compile the SSM recurrence as an
XLA ``While`` op instead of unrolling T iterations into T graph nodes.
This reduces compilation time from ~30 min to ~1 min for T=512.

Also patches the v2 Gated DeltaNet scan with an XLA-friendly version.

Use this module from TPU training scripts. Import it BEFORE constructing
any LSA model — it monkey-patches the scan functions at import time.
"""

from __future__ import annotations

import torch

from . import lsa as _lsa_gpu

# Try to import the XLA scan; fall back to plain loop if not available
# (e.g., running on CPU for testing).
try:
    from torch_xla.experimental.scan import scan as xla_scan
    HAS_XLA_SCAN = True
except ImportError:
    HAS_XLA_SCAN = False


# ---------------------------------------------------------------------------
# v1: SSM scan (vector state) — used by aria.lsa.LSAAttention
# ---------------------------------------------------------------------------

def ssm_scan_xla(A: torch.Tensor, Bg: torch.Tensor,
                 state_input: torch.Tensor) -> torch.Tensor:
    """Sequential causal SSM scan using torch_xla.experimental.scan.

    XLA compiles ONE iteration body and reuses it via a While loop,
    instead of unrolling T iterations into T graph nodes. This cuts
    compilation from ~30 min to ~1 min for T=512.

    Falls back to a plain Python loop if torch_xla.experimental.scan
    is not available (e.g., on CPU for testing).

    Args:
        A: (B, T, d_state) per-token decay gate in [0, 1].
        Bg: (B, T, d_state) per-token write gate in [-1, 1].
        state_input: (B, T, d_state) what to write at each step.

    Returns:
        states: (B, T, d_state) per-position causal states.
    """
    B, T, D = A.shape

    if HAS_XLA_SCAN:
        # scan expects inputs as (T, ...) and a step function:
        #   step(carry, x) -> (new_carry, output)
        # Transpose from (B, T, D) to (T, B, D) for the scan dimension.
        xs = (A.transpose(0, 1), Bg.transpose(0, 1), state_input.transpose(0, 1))
        init = torch.zeros(B, D, device=A.device, dtype=A.dtype)

        def step(s: torch.Tensor, x: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
            a, bg, u = x
            s_new = a * s + bg * u
            return s_new, s_new  # (carry, output)

        _, outputs = xla_scan(step, init, xs)
        # outputs: (T, B, D) -> (B, T, D)
        return outputs.transpose(0, 1)
    else:
        # Fallback: plain Python loop (slow but correct).
        s = torch.zeros(B, D, device=A.device, dtype=A.dtype)
        out_list = []
        for t in range(T):
            s = A[:, t] * s + Bg[:, t] * state_input[:, t]
            out_list.append(s)
        return torch.stack(out_list, dim=1)


# ---------------------------------------------------------------------------
# v2: Gated DeltaNet scan (matrix state) — used by aria.lsa_v2
# ---------------------------------------------------------------------------

def gated_delta_rule_xla(
    q: torch.Tensor,        # (B, H, T, K)
    k: torch.Tensor,        # (B, H, T, K)
    v: torch.Tensor,        # (B, H, T, V)
    g: torch.Tensor,        # (B, H, T)
    beta: torch.Tensor,     # (B, H, T)
    chunk_size: int = 64,   # ignored on XLA; kept for API compat
) -> torch.Tensor:
    """Gated DeltaNet scan using torch_xla.experimental.scan.

    Same math as gated_delta_rule_ref / chunked_gated_delta_rule_torch,
    but compiles one iteration body instead of unrolling T graph nodes.
    State is fp32 regardless of input dtype.
    """
    B, H, T, K = q.shape
    V = v.size(-1)

    if HAS_XLA_SCAN:
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        g_f = g.float()
        b_f = beta.float()

        # Transpose time dim to position 0: (T, B, H, K/V)
        q_t = q_f.permute(2, 0, 1, 3)    # (T, B, H, K)
        k_t = k_f.permute(2, 0, 1, 3)    # (T, B, H, K)
        v_t = v_f.permute(2, 0, 1, 3)    # (T, B, H, V)
        g_t = g_f.permute(2, 0, 1)       # (T, B, H)
        b_t = b_f.permute(2, 0, 1)       # (T, B, H)

        xs = (q_t, k_t, v_t, g_t, b_t)
        init_S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)

        def step(S: torch.Tensor, x: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
            q_i, k_i, v_i, g_i, b_i = x
            # Decay
            alpha = torch.exp(g_i)                               # (B, H)
            S_pre = S * alpha[:, :, None, None]
            # Delta-rule correction
            sTk = torch.einsum("bhkv,bhk->bhv", S_pre, k_i)     # (B, H, V)
            v_new = b_i[:, :, None] * (v_i - sTk)               # (B, H, V)
            # Rank-1 write
            S_new = S_pre + torch.einsum("bhk,bhv->bhkv", k_i, v_new)
            # Readout
            o_i = torch.einsum("bhkv,bhk->bhv", S_new, q_i)     # (B, H, V)
            return S_new, o_i

        _, outputs = xla_scan(step, init_S, xs)
        # outputs: (T, B, H, V) -> (B, H, T, V)
        return outputs.permute(1, 2, 0, 3).to(q.dtype)
    else:
        # Fallback: use the chunked scan from lsa_v2
        from .lsa_v2 import chunked_gated_delta_rule_torch
        return chunked_gated_delta_rule_torch(q, k, v, g, beta, chunk_size)


# ---------------------------------------------------------------------------
# Monkey-patch at import time
# ---------------------------------------------------------------------------

# Patch v1 SSM scan
_lsa_gpu.ssm_scan_jit = ssm_scan_xla

# Patch v2 Gated DeltaNet scan (if lsa_v2 is importable)
try:
    from . import lsa_v2 as _lsa_v2
    _lsa_v2.chunked_gated_delta_rule_torch = gated_delta_rule_xla
    # Also set it as the fallback so GatedDeltaRecurrence.forward uses it
    # when HAS_FLA is False (which is always on TPU).
except ImportError:
    pass


# Re-export for convenience
from .lsa import (
    LSAAttention,
    LSABlock,
    LSALanguageModel,
)

__all__ = [
    "ssm_scan_xla",
    "gated_delta_rule_xla",
    "LSAAttention",
    "LSABlock",
    "LSALanguageModel",
]
