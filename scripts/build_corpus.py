"""Build a large-scale sharded mmap corpus for ARIA pretraining.

Usage:
    python scripts/build_corpus.py --config configs/aria_v3_1b_tpu_spmd.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aria.corpus import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override data.output_dir from the YAML.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest_path = build_corpus(cfg, output_dir=args.output_dir)
    print(f"Corpus manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
