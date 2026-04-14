"""Quick smoke test: forward + backward + step on both models.

Ensures the code actually runs before you commit to a 3000-step training run.
Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aria.lsa import LSALanguageModel
from aria.baseline import BaselineLanguageModel
from aria.nn_utils import count_parameters


def build_from_config(config_path: str, vocab_size: int = 65):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name")
    model_cfg["vocab_size"] = vocab_size
    cls = LSALanguageModel if name == "lsa" else BaselineLanguageModel
    return cls(**model_cfg), name


def smoke(model: torch.nn.Module, name: str, device: torch.device,
          batch_size: int = 4, seq_len: int = 128) -> None:
    print(f"\n--- {name} ---")
    print(f"parameters: {count_parameters(model):,}")
    model.to(device)
    model.train()

    x = torch.randint(0, 65, (batch_size, seq_len), device=device)
    y = torch.randint(0, 65, (batch_size, seq_len), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Forward
    t0 = time.time()
    logits, loss = model(x, y)
    t_fwd = time.time() - t0
    print(f"forward   OK | logits shape {tuple(logits.shape)} | loss {loss.item():.4f} | {t_fwd*1000:.1f} ms")
    assert logits.shape == (batch_size, seq_len, 65)
    assert torch.isfinite(loss)

    # Backward
    t0 = time.time()
    loss.backward()
    t_bwd = time.time() - t0
    print(f"backward  OK | {t_bwd*1000:.1f} ms")

    # Optimizer step
    t0 = time.time()
    opt.step()
    opt.zero_grad()
    t_step = time.time() - t0
    print(f"opt step  OK | {t_step*1000:.1f} ms")

    # Second step (checks repeated training works)
    logits2, loss2 = model(x, y)
    loss2.backward()
    opt.step()
    print(f"2nd step  OK | loss {loss2.item():.4f}")

    # Memory snapshot
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"peak VRAM : {mem_mb:.1f} MB")
        torch.cuda.reset_peak_memory_stats()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for config_path in ["configs/baseline_tiny.yaml", "configs/lsa_tiny.yaml"]:
        model, name = build_from_config(str(ROOT / config_path))
        smoke(model, name, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
