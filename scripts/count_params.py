"""Sanity check: confirm LSA and baseline have matched parameter counts.

A fair A/B comparison requires the two models to have nearly identical parameter
counts. Run this before training to verify.

Usage:
    python scripts/count_params.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aria.lsa import LSALanguageModel
from aria.baseline import BaselineLanguageModel
from aria.nn_utils import count_parameters


def count_from_config(config_path: str, vocab_size: int = 128) -> tuple[int, str]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name")
    model_cfg["vocab_size"] = vocab_size
    cls = LSALanguageModel if name == "lsa" else BaselineLanguageModel
    model = cls(**model_cfg)
    return count_parameters(model), name


def format_count(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def main() -> None:
    configs = [
        "configs/lsa_tiny.yaml",
        "configs/baseline_tiny.yaml",
    ]
    results = []
    for path in configs:
        full_path = ROOT / path
        if not full_path.exists():
            print(f"MISSING: {path}")
            continue
        n, name = count_from_config(str(full_path))
        results.append((path, name, n))
        print(f"{path}")
        print(f"  model: {name}")
        print(f"  parameters: {n:,} ({format_count(n)})")
        print()

    if len(results) == 2:
        n_a, n_b = results[0][2], results[1][2]
        diff = abs(n_a - n_b)
        pct = 100 * diff / max(n_a, n_b)
        print("-" * 50)
        print(f"Parameter delta: {diff:,} ({pct:.1f}%)")
        if pct < 5:
            print("OK: models are matched within 5%.")
        elif pct < 15:
            print("WARN: >5% parameter mismatch. Tune d_ff or d_kv_latent to close the gap.")
        else:
            print("FAIL: >15% mismatch. Comparison will NOT be fair.")


if __name__ == "__main__":
    main()
