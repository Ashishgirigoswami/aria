"""Run baseline + LSA back-to-back for a single seed.

Usage:
    python scripts/ablate_seed.py 43
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_training(config_path: str) -> None:
    print(f"\n{'='*60}\nTraining: {config_path}\n{'='*60}\n", flush=True)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", config_path],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {config_path}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/ablate_seed.py <seed>")
        sys.exit(1)
    seed = sys.argv[1]
    configs = [
        f"configs/baseline_seed{seed}.yaml",
        f"configs/lsa_seed{seed}.yaml",
    ]
    for cfg in configs:
        if not (ROOT / cfg).exists():
            print(f"Missing config: {cfg}")
            sys.exit(1)
    for cfg in configs:
        run_training(cfg)
    print(f"\nSeed {seed} ablation complete.")


if __name__ == "__main__":
    main()
