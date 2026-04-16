"""Smoke tests for Mamba-3 block."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aria.mamba3 import Mamba3Block, mamba3_scan_ref, apply_rope_cumangle


def test_shapes():
    torch.manual_seed(0)
    blk = Mamba3Block(d_model=128, d_state=16, n_heads=4, headdim=32)
    x = torch.randn(2, 64, 128)
    y = blk(x)
    assert y.shape == x.shape


def test_lambda_one_collapses_to_mamba2():
    """When lambda_t = 1 for all t, Mamba-3 should reduce to Mamba-2 form:
    h_t = alpha_t * h_{t-1} + dt_t * B_t * x_t.
    (gamma_t = dt, beta_t = 0 when lambda=1.)
    """
    torch.manual_seed(1)
    B, T, H, P, N = 1, 8, 2, 4, 8
    x = torch.randn(B, T, H, P)
    dt = torch.rand(B, T, H) * 0.05 + 0.01
    A = torch.log(torch.arange(1, H + 1, dtype=torch.float32))
    B_proj = torch.randn(B, T, H, N)
    C_proj = torch.randn(B, T, H, N)
    lam1 = torch.ones(B, T, H)
    y_m3 = mamba3_scan_ref(x, dt, A, B_proj, C_proj, lam1)

    # Mamba-2 form: h_t = alpha_t h_{t-1} + dt B_t x_t
    A_neg = -torch.exp(A)
    h = torch.zeros(B, H, P, N)
    y_m2 = torch.empty_like(x)
    for t in range(T):
        alpha_t = torch.exp(dt[:, t] * A_neg[None])
        Bx = x[:, t].unsqueeze(-1) * B_proj[:, t].unsqueeze(-2)
        h = alpha_t[..., None, None] * h + dt[:, t][..., None, None] * Bx
        y_m2[:, t] = (h * C_proj[:, t].unsqueeze(-2)).sum(-1)

    assert torch.allclose(y_m3, y_m2, rtol=1e-5, atol=1e-5), \
        f"max diff {(y_m3 - y_m2).abs().max().item()}"


def test_rope_cumangle_shape():
    z = torch.randn(2, 16, 4, 8)
    ang = torch.randn(2, 16, 4)
    out = apply_rope_cumangle(z, ang)
    assert out.shape == z.shape
    # Zero angle must be identity
    out0 = apply_rope_cumangle(z, torch.zeros_like(ang))
    assert torch.allclose(out0, z, rtol=1e-6, atol=1e-6)


def test_gradient_flow():
    """Gradients must reach the inputs and parameters."""
    torch.manual_seed(2)
    blk = Mamba3Block(d_model=64, d_state=8, n_heads=2, headdim=32)
    x = torch.randn(1, 16, 64, requires_grad=True)
    y = blk(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert blk.A_log.grad is not None
    assert blk.in_proj.weight.grad is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
