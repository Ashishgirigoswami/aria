"""Main training entry point.

Usage:
    python scripts/train.py --config configs/lsa_tiny.yaml
    python scripts/train.py --config configs/baseline_tiny.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running from project root without installing the package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aria.data import build_datasets, build_bpe_datasets, RandomWindowSampler
from aria.lsa import LSALanguageModel
from aria.lsa_v2 import LSAv2LanguageModel
from aria.baseline import BaselineLanguageModel
from aria.nn_utils import count_parameters
from aria.trainer import Trainer, TrainConfig, resolve_device


MODEL_REGISTRY = {
    "lsa": LSALanguageModel,
    "lsa_v2": LSAv2LanguageModel,
    "baseline": BaselineLanguageModel,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, vocab_size: int) -> torch.nn.Module:
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model_cfg["vocab_size"] = vocab_size   # override config vocab with actual tokenizer
    cls = MODEL_REGISTRY[model_name]
    return cls(**model_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--override-device", default=None,
                        help="Override device (auto/cuda/cpu)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    device_str = args.override_device or cfg["training"]["device"]
    device = resolve_device(device_str)

    # --- Data ---
    tokenizer_kind = cfg["data"].get("tokenizer", "char")
    if tokenizer_kind == "bpe":
        train_ds, val_ds, tokenizer = build_bpe_datasets(
            dataset=cfg["data"].get("dataset", "tinyshakespeare"),
            cache_dir=cfg["data"]["cache_dir"],
            seq_len=cfg["data"]["seq_len"],
            train_split=cfg["data"]["train_split"],
            max_train_tokens=cfg["data"].get("max_train_tokens"),
        )
    else:
        train_ds, val_ds, tokenizer = build_datasets(
            cache_dir=cfg["data"]["cache_dir"],
            seq_len=cfg["data"]["seq_len"],
            train_split=cfg["data"]["train_split"],
        )
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    train_sampler = RandomWindowSampler(train_ds, cfg["training"]["batch_size"], device)
    val_sampler = RandomWindowSampler(val_ds, cfg["training"]["batch_size"], device)

    # --- Model ---
    model = build_model(cfg, vocab_size=tokenizer.vocab_size)
    n_params = count_parameters(model)
    print(f"Model: {cfg['model']['name']}, parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # --- Training ---
    train_cfg_dict = dict(cfg["training"])
    train_cfg_dict["checkpoint_dir"] = cfg["logging"]["checkpoint_dir"]
    train_cfg_dict["run_name"] = cfg["logging"]["run_name"]
    train_cfg_dict["log_every"] = cfg["logging"]["log_every"]

    # Remove training-only fields not in TrainConfig
    train_cfg_dict = {k: v for k, v in train_cfg_dict.items()
                      if k in TrainConfig.__dataclass_fields__}
    train_cfg = TrainConfig(**train_cfg_dict)

    trainer = Trainer(model, train_sampler, val_sampler, train_cfg)

    # Optional resume — explicit flag or auto-detect latest checkpoint
    resume_path = args.resume
    if resume_path is None:
        auto = Path(train_cfg.checkpoint_dir) / "final.pt"
        if auto.exists():
            resume_path = str(auto)
            print(f"Auto-detected resume checkpoint: {resume_path}")
    if resume_path:
        trainer.load_checkpoint(resume_path)

    result = trainer.fit()

    # --- Save summary ---
    summary = {
        "run_name": cfg["logging"]["run_name"],
        "model": cfg["model"]["name"],
        "n_params": n_params,
        "best_val_loss": result["best_val_loss"],
        "best_val_ppl": float(np.exp(result["best_val_loss"])),
        "elapsed_seconds": result["elapsed_seconds"],
        "history": result["history"],
    }
    summary_path = Path(cfg["logging"]["checkpoint_dir"]) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
