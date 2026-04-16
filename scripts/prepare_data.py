"""One-off data preparation: download + BPE-tokenize a dataset into cached .bin files.

Run this once before launching parallel training processes so they share the same
cached tokenization instead of racing to build it.

Usage:
    python scripts/prepare_data.py --dataset wikitext-103 --max-tokens 50000000
    python scripts/prepare_data.py --dataset tinyshakespeare
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aria.data import build_bpe_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BPE-tokenized dataset cache.")
    parser.add_argument("--dataset", default="wikitext-103",
                        choices=["wikitext-103", "tinyshakespeare", "fineweb-edu"])
    parser.add_argument("--cache-dir", default="./data")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=50_000_000)
    args = parser.parse_args()

    print(f"Preparing {args.dataset} (max_tokens={args.max_tokens:,})", flush=True)
    train_ds, val_ds, tok = build_bpe_datasets(
        dataset=args.dataset,
        cache_dir=args.cache_dir,
        seq_len=args.seq_len,
        max_train_tokens=args.max_tokens,
    )
    bin_dir = Path(args.cache_dir) / "bpe_tokens" / args.dataset
    print(f"Done. Cached to {bin_dir}/train.bin + val.bin", flush=True)
    print(f"Train: {len(train_ds.data):,} tokens | Val: {len(val_ds.data):,} tokens",
          flush=True)


if __name__ == "__main__":
    main()
