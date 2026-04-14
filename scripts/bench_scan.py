"""Benchmark SSM scan correctness + speed.

Compares the Python-loop reference against the JIT-scripted version.
Verifies numerical equivalence then measures wall-time speedup.

Usage:
    python scripts/bench_scan.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aria.lsa import ssm_scan_jit


def ssm_scan_reference(A: torch.Tensor, Bg: torch.Tensor,
                       state_input: torch.Tensor) -> torch.Tensor:
    """Pure-Python reference implementation. Slow but obviously correct."""
    B, T, D = A.shape
    states = torch.empty(B, T, D, device=A.device, dtype=A.dtype)
    s = torch.zeros(B, D, device=A.device, dtype=A.dtype)
    for t in range(T):
        s = A[:, t] * s + Bg[:, t] * state_input[:, t]
        states[:, t] = s
    return states


def test_correctness(device: torch.device) -> None:
    """Assert JIT version matches Python reference to float32 precision."""
    torch.manual_seed(0)
    B, T, D = 4, 128, 64
    A = torch.sigmoid(torch.randn(B, T, D, device=device))
    Bg = torch.tanh(torch.randn(B, T, D, device=device))
    x = torch.randn(B, T, D, device=device)

    ref = ssm_scan_reference(A, Bg, x)
    jit = ssm_scan_jit(A, Bg, x)

    max_diff = (ref - jit).abs().max().item()
    print(f"correctness: max |ref - jit| = {max_diff:.2e}")
    assert max_diff < 1e-5, f"JIT scan diverges from reference: {max_diff}"
    print("correctness: OK")


def bench(fn, A, Bg, x, n_warmup: int = 5, n_runs: int = 20) -> float:
    """Return median wall time in milliseconds."""
    for _ in range(n_warmup):
        _ = fn(A, Bg, x)
    if A.device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = fn(A, Bg, x)
        if A.device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def test_speed(device: torch.device) -> None:
    print()
    print(f"{'shape (B, T, D)':<22}{'reference':>14}{'jit':>14}{'speedup':>12}")
    print("-" * 62)
    for (B, T, D) in [(4, 64, 64), (16, 128, 128), (16, 256, 128), (32, 256, 128)]:
        torch.manual_seed(1)
        A = torch.sigmoid(torch.randn(B, T, D, device=device))
        Bg = torch.tanh(torch.randn(B, T, D, device=device))
        x = torch.randn(B, T, D, device=device)
        ref_ms = bench(ssm_scan_reference, A, Bg, x, n_warmup=2, n_runs=5)
        jit_ms = bench(ssm_scan_jit, A, Bg, x)
        speedup = ref_ms / jit_ms
        shape_str = f"({B}, {T}, {D})"
        print(f"{shape_str:<22}{ref_ms:>12.2f}ms{jit_ms:>12.2f}ms{speedup:>11.2f}x")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    test_correctness(device)
    test_speed(device)


if __name__ == "__main__":
    main()
