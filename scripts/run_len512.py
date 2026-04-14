"""Run both len-512 configs sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main() -> None:
    for cfg in ["configs/baseline_len512.yaml", "configs/lsa_len512.yaml"]:
        print(f"\n=== Training {cfg} ===", flush=True)
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", cfg],
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            raise RuntimeError(f"Training failed: {cfg}")
    print("\nlen-512 ablation complete.")


if __name__ == "__main__":
    main()
