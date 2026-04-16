"""Layered State Attention for torch_xla (TPU) backends.

Uses ``torch_xla.experimental.scan`` for O(1)-compilation scans, wrapped
in a custom ``torch.autograd.Function`` so that backward gradients flow
correctly (the raw scan primitive does not support autograd on torch_xla 2.6).

Forward scan: s_t = A_t * s_{t-1} + Bg_t * u_t
Backward scan (reverse): accumulates dL/ds_t via carry = ds * A_t

Both compile as a single XLA While op — one iteration body, reused T times.
Compilation: ~30s instead of ~30min. Step time: ~0.75s instead of ~22s.
"""

from __future__ import annotations

import torch

from . import lsa as _lsa_gpu

try:
    from torch_xla.experimental.scan import scan as xla_scan
    HAS_XLA_SCAN = True
except ImportError:
    HAS_XLA_SCAN = False


# ---------------------------------------------------------------------------
# v1: SSM scan with autograd-compatible XLA scan
# ---------------------------------------------------------------------------

class _SSMScanFunction(torch.autograd.Function):
    """Custom autograd for SSM scan on XLA.

    Forward:  s_t = A_t * s_{t-1} + Bg_t * u_t
    Backward: reverse scan computes dL/ds_t, then pointwise grads for A, Bg, u.
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, Bg: torch.Tensor,
                state_input: torch.Tensor) -> torch.Tensor:
        B, T, D = A.shape

        if HAS_XLA_SCAN:
            xs = (A.transpose(0, 1), Bg.transpose(0, 1),
                  state_input.transpose(0, 1))
            init = torch.zeros(B, D, device=A.device, dtype=A.dtype)

            def fwd_step(s, x):
                a, bg, u = x
                s_new = a * s + bg * u
                return s_new, s_new

            _, outputs = xla_scan(fwd_step, init, xs)
            states = outputs.transpose(0, 1)  # (B, T, D)
        else:
            s = torch.zeros(B, D, device=A.device, dtype=A.dtype)
            out_list = []
            for t in range(T):
                s = A[:, t] * s + Bg[:, t] * state_input[:, t]
                out_list.append(s)
            states = torch.stack(out_list, dim=1)

        ctx.save_for_backward(A, Bg, state_input, states)
        return states

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        A, Bg, state_input, states = ctx.saved_tensors
        B, T, D = A.shape

        # s_prev[t] = states[t-1] for t>0, else 0
        s_prev = torch.cat([
            torch.zeros(B, 1, D, device=A.device, dtype=A.dtype),
            states[:, :-1],
        ], dim=1)  # (B, T, D)

        if HAS_XLA_SCAN:
            # Reverse scan to compute dL/ds[t] for each t.
            # Recurrence (backward): ds[t] = grad_output[t] + carry
            #                         carry_new = ds[t] * A[t]
            go_flip = grad_output.flip(1).transpose(0, 1)  # (T, B, D) reversed
            a_flip = A.flip(1).transpose(0, 1)              # (T, B, D) reversed
            init = torch.zeros(B, D, device=A.device, dtype=A.dtype)

            def bwd_step(carry, x):
                go_t, a_t = x
                ds = go_t + carry
                new_carry = ds * a_t
                return new_carry, ds

            _, ds_flip = xla_scan(bwd_step, init, (go_flip, a_flip))
            ds_all = ds_flip.transpose(0, 1).flip(1)  # (B, T, D) forward order
        else:
            # Plain Python reverse loop fallback
            ds_list = [None] * T
            carry = torch.zeros(B, D, device=A.device, dtype=A.dtype)
            for t in range(T - 1, -1, -1):
                ds = grad_output[:, t] + carry
                ds_list[t] = ds
                carry = ds * A[:, t]
            ds_all = torch.stack(ds_list, dim=1)

        # Pointwise gradients
        dA = ds_all * s_prev
        dBg = ds_all * state_input
        du = ds_all * Bg

        return dA, dBg, du


def ssm_scan_xla(A: torch.Tensor, Bg: torch.Tensor,
                 state_input: torch.Tensor) -> torch.Tensor:
    """SSM scan with XLA-optimized forward AND backward.

    Both forward and backward use ``torch_xla.experimental.scan`` via a
    custom ``torch.autograd.Function``, so compilation is O(1) in sequence
    length and gradients flow correctly.
    """
    return _SSMScanFunction.apply(A, Bg, state_input)


# ---------------------------------------------------------------------------
# Monkey-patch at import time
# ---------------------------------------------------------------------------

_lsa_gpu.ssm_scan_jit = ssm_scan_xla

# Patch v2 Gated DeltaNet — for now, v2 on TPU uses the chunked PyTorch
# scan (not the XLA scan), because the delta-rule backward is more complex
# (matrix state). The v1 scan fix above is the priority.
try:
    from . import lsa_v2 as _lsa_v2
    # v2 already has chunked_gated_delta_rule_torch as fallback; leave it.
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
    "LSAAttention",
    "LSABlock",
    "LSALanguageModel",
]
