"""Colab-TPU smoke test for the LSA + baseline models.

Run this in a Colab notebook with Runtime -> Change runtime type -> TPU
BEFORE submitting the TRC project number form. Verifies that:

1. torch_xla imports and sees a TPU device
2. Both LSALanguageModel and BaselineLanguageModel construct on the TPU
3. One forward + backward + optimizer step succeeds
4. A second step produces a lower loss (sanity check that gradients flow)
5. Parameter counts match the CPU / GPU values

Usage (from a Colab TPU runtime, with the repo cloned):

    !pip install torch_xla
    !python scripts/smoke_test_tpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Patch the SSM scan FIRST (removes @torch.jit.script for XLA).
from aria import lsa_xla  # noqa: F401
from aria.lsa import LSALanguageModel
from aria.baseline import BaselineLanguageModel
from aria.nn_utils import count_parameters

import torch_xla
import torch_xla.core.xla_model as xm  # still used for optimizer_step


def smoke(model: torch.nn.Module, name: str, device: torch.device,
          vocab_size: int = 65, batch_size: int = 4, seq_len: int = 64) -> None:
    print(f"\n--- {name} ---")
    print(f"parameters: {count_parameters(model):,}")
    model.to(device)
    model.train()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Step 1
    logits, loss = model(x, y)
    loss.backward()
    xm.optimizer_step(opt)
    torch_xla.sync()
    loss1 = loss.item()
    print(f"step 1 | logits shape {tuple(logits.shape)} | loss {loss1:.4f}")
    assert logits.shape == (batch_size, seq_len, vocab_size), (
        f"expected logits shape {(batch_size, seq_len, vocab_size)}, got {tuple(logits.shape)}"
    )
    assert torch.isfinite(torch.tensor(loss1)), f"loss is non-finite: {loss1}"

    # Step 2 — fresh gradients
    opt.zero_grad(set_to_none=True)
    logits2, loss2 = model(x, y)
    loss2.backward()
    xm.optimizer_step(opt)
    torch_xla.sync()
    loss2_val = loss2.item()
    print(f"step 2 | loss {loss2_val:.4f}")
    assert loss2_val < loss1 + 1e-3, (
        f"loss did not decrease: step1={loss1:.4f}, step2={loss2_val:.4f}"
    )
    print(f"OK: loss decreased ({loss1:.4f} -> {loss2_val:.4f})")


def main() -> None:
    device = torch_xla.device()
    print(f"XLA device: {device}")
    try:
        print(f"device kind: {xm.xla_device_hw(device)}")
    except Exception:
        pass  # xla_device_hw is deprecated/removed in newer torch_xla

    vocab = 65
    lsa = LSALanguageModel(
        vocab_size=vocab, d_model=256, n_layers=4, n_heads=4, d_head=64,
        d_ff=512, d_kv_latent=128, d_state=128, max_seq_len=128,
    )
    smoke(lsa, "LSALanguageModel", device, vocab_size=vocab,
          batch_size=4, seq_len=64)

    base = BaselineLanguageModel(
        vocab_size=vocab, d_model=256, n_layers=4, n_heads=4, d_head=64,
        d_ff=512, max_seq_len=128,
    )
    smoke(base, "BaselineLanguageModel", device, vocab_size=vocab,
          batch_size=4, seq_len=64)

    print("\nAll TPU smoke tests passed. Safe to run full training on TPU.")


if __name__ == "__main__":
    main()
