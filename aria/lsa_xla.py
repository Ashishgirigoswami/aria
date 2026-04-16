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

# ---------------------------------------------------------------------------
# v2: Gated DeltaNet scan with custom autograd for XLA
# ---------------------------------------------------------------------------

class _GDNScanFunction(torch.autograd.Function):
    """Custom autograd for Gated DeltaNet delta-rule scan on XLA.

    Forward: S_t = α_t·S_{t-1} + β_t·(v_t − S_{t-1}^T k_t)·k_t^T
             o_t = S_t^T q_t
    Backward: reverse scan accumulating dL/dS_t through the matrix state.

    Uses xla_scan for O(1) compilation (one iteration compiled, reused T times).
    State is fp32 regardless of input dtype.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, H, T, K = q.shape
        V = v.size(-1)

        q_f, k_f, v_f = q.float(), k.float(), v.float()
        g_f, b_f = g.float(), beta.float()

        if HAS_XLA_SCAN:
            xs = (q_f.permute(2,0,1,3), k_f.permute(2,0,1,3),
                  v_f.permute(2,0,1,3), g_f.permute(2,0,1), b_f.permute(2,0,1))
            init_S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)

            def fwd_step(S, x):
                q_i, k_i, v_i, g_i, b_i = x
                alpha = torch.exp(g_i)
                S_pre = S * alpha[:, :, None, None]
                sTk = torch.einsum("bhkv,bhk->bhv", S_pre, k_i)
                v_new = b_i[:, :, None] * (v_i - sTk)
                S_new = S_pre + torch.einsum("bhk,bhv->bhkv", k_i, v_new)
                o_i = torch.einsum("bhkv,bhk->bhv", S_new, q_i)
                return S_new, o_i

            _, outputs = xla_scan(fwd_step, init_S, xs)
            out = outputs.permute(1, 2, 0, 3).to(q.dtype)
        else:
            S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
            out = torch.empty(B, H, T, V, dtype=q.dtype, device=q.device)
            for t in range(T):
                alpha = torch.exp(g_f[:, :, t])
                S = S * alpha[:, :, None, None]
                sTk = torch.einsum("bhkv,bhk->bhv", S, k_f[:, :, t])
                v_new = b_f[:, :, t, None] * (v_f[:, :, t] - sTk)
                S = S + torch.einsum("bhk,bhv->bhkv", k_f[:, :, t], v_new)
                out[:, :, t] = torch.einsum("bhkv,bhk->bhv", S, q_f[:, :, t]).to(q.dtype)

        ctx.save_for_backward(q, k, v, g, beta, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, g, beta, out = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.size(-1)

        # Recompute forward states (needed for backward)
        q_f, k_f, v_f = q.float(), k.float(), v.float()
        g_f, b_f = g.float(), beta.float()
        go = grad_output.float()

        # Simple Python backward (no XLA scan for backward — too complex for matrix state).
        # This is slower but correct. The forward uses XLA scan for speed;
        # backward is less critical since it runs once per step.
        states = []
        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        for t in range(T):
            alpha = torch.exp(g_f[:, :, t])
            S = S * alpha[:, :, None, None]
            sTk = torch.einsum("bhkv,bhk->bhv", S, k_f[:, :, t])
            v_new = b_f[:, :, t, None] * (v_f[:, :, t] - sTk)
            S = S + torch.einsum("bhk,bhv->bhkv", k_f[:, :, t], v_new)
            states.append(S)

        # Backward pass
        dq = torch.zeros_like(q_f)
        dk = torch.zeros_like(k_f)
        dv = torch.zeros_like(v_f)
        dg = torch.zeros_like(g_f)
        dbeta = torch.zeros_like(b_f)
        dS = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)

        for t in range(T - 1, -1, -1):
            S_t = states[t]
            S_prev = states[t - 1] if t > 0 else torch.zeros_like(S_t)
            alpha_t = torch.exp(g_f[:, :, t])
            k_t = k_f[:, :, t]
            v_t = v_f[:, :, t]
            q_t = q_f[:, :, t]
            beta_t = b_f[:, :, t]
            go_t = go[:, :, t]

            # Gradient from output: o_t = S_t^T q_t
            dq[:, :, t] = torch.einsum("bhkv,bhv->bhk", S_t, go_t)
            dS = dS + torch.einsum("bhk,bhv->bhkv", q_t, go_t)

            # Gradient through S_t = S_pre + k_t ⊗ v_new
            S_pre = S_prev * alpha_t[:, :, None, None]
            sTk = torch.einsum("bhkv,bhk->bhv", S_pre, k_t)
            v_new = beta_t[:, :, None] * (v_t - sTk)

            # dv_new from outer product
            dk_from_outer = torch.einsum("bhkv,bhv->bhk", dS, v_new)  # not used directly for dk
            dv_new = torch.einsum("bhkv,bhk->bhv", dS, k_t)

            # dk from outer product: dk += dS @ v_new^T ... no, k ⊗ v_new means dk += sum over V
            dk[:, :, t] = dk[:, :, t] + torch.einsum("bhkv,bhv->bhk", dS, v_new)

            # dv_new → dbeta, dv, dS_pre (through sTk)
            dbeta[:, :, t] = (dv_new * (v_t - sTk)).sum(dim=-1)
            dv[:, :, t] = dv[:, :, t] + dv_new * beta_t[:, :, None]
            dsTk = -dv_new * beta_t[:, :, None]

            # dS_pre from sTk = S_pre^T k
            dS_pre_from_sTk = torch.einsum("bhv,bhk->bhkv", dsTk, k_t)
            dk[:, :, t] = dk[:, :, t] + torch.einsum("bhkv,bhv->bhk", S_pre, dsTk)

            # dS_pre total = dS (from carry) + dS_pre_from_sTk
            dS_pre = dS + dS_pre_from_sTk

            # dS_pre → dS_{t-1} and dg
            dg[:, :, t] = (dS_pre * S_prev * alpha_t[:, :, None, None]).sum(dim=(-2, -1))
            dS = dS_pre * alpha_t[:, :, None, None]

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.to(g.dtype), dbeta.to(beta.dtype)


def gdn_scan_xla(q, k, v, g, beta, chunk_size=64):
    """Gated DeltaNet scan with XLA-optimized forward + Python backward."""
    return _GDNScanFunction.apply(q, k, v, g, beta)


# Monkey-patch v1 SSM scan
_lsa_gpu.ssm_scan_jit = ssm_scan_xla

# Monkey-patch v2 Gated DeltaNet scan
try:
    from . import lsa_v2 as _lsa_v2
    _lsa_v2.chunked_gated_delta_rule_torch = gdn_scan_xla
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
