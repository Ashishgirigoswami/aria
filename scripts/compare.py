"""Train LSA and baseline back-to-back, print side-by-side comparison.

Usage:
    python scripts/compare.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_training(config_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"Training: {config_path}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", config_path],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {config_path}")


def load_summary(checkpoint_dir: str) -> dict:
    path = Path(checkpoint_dir) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    configs = [
        ("configs/baseline_tiny.yaml", "./checkpoints/baseline_tiny"),
        ("configs/lsa_tiny.yaml",      "./checkpoints/lsa_tiny"),
    ]

    # Train both (you can also run these separately if you prefer)
    for config_path, _ in configs:
        run_training(config_path)

    # Load summaries
    summaries = []
    for _, ckpt_dir in configs:
        summaries.append(load_summary(ckpt_dir))

    # Print comparison table
    print("\n")
    print("=" * 70)
    print("Phase 0 comparison — LSA vs matched baseline")
    print("=" * 70)
    print()
    print(f"{'metric':<25} {'baseline':>20} {'lsa':>20}")
    print("-" * 70)

    b, l = summaries[0], summaries[1]

    def row(label: str, b_val, l_val, fmt: str = "{:.4f}"):
        bs = fmt.format(b_val) if b_val is not None else "-"
        ls = fmt.format(l_val) if l_val is not None else "-"
        print(f"{label:<25} {bs:>20} {ls:>20}")

    row("parameters", b["n_params"], l["n_params"], "{:,}")
    row("best val loss", b["best_val_loss"], l["best_val_loss"])
    row("best val ppl", b["best_val_ppl"], l["best_val_ppl"], "{:.2f}")
    row("training wall time (s)", b["elapsed_seconds"], l["elapsed_seconds"], "{:.1f}")

    # Delta summary
    print()
    ppl_delta = l["best_val_ppl"] - b["best_val_ppl"]
    ppl_pct = 100 * ppl_delta / b["best_val_ppl"]
    print(f"LSA perplexity delta vs baseline: {ppl_delta:+.2f} ({ppl_pct:+.1f}%)")
    if ppl_delta <= 0:
        print("  LSA matches or beats baseline at same parameter count.")
    elif ppl_pct < 5:
        print("  LSA within 5% of baseline — acceptable for the memory savings.")
    else:
        print("  LSA underperforms baseline by >5%. Tune before scaling up.")

    print()
    print("Next steps:")
    print("  1. Verify LSA actually uses less KV memory (expected: yes by design)")
    print("  2. Scale context length to 512, 1024 and re-compare")
    print("  3. If results hold, apply to TPU Research Cloud for 100M scale")
    print("  4. Draft arxiv paper: 'Layered State Attention'")


if __name__ == "__main__":
    main()
