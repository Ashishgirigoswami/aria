"""Train pure Mamba-3 stack on Kaggle T4x2. CUDA-native, no torch_xla.

Resumes from /kaggle/working/checkpoints/.../latest.pt on every restart so
Kaggle's 9-hour session limit doesn't lose progress. Save a notebook output,
attach it as a Dataset input on the next session, and this script resumes.

Usage (from Kaggle notebook):
    %cd /kaggle/working/aria
    !python scripts/train_mamba3_kaggle.py --config configs/mamba3_150m_kaggle.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aria.data import build_bpe_datasets, RandomWindowSampler
from aria.mamba3_model import Mamba3LanguageModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wsd_lr(step: int, warmup: int, max_steps: int,
           base_lr: float, min_lr: float, decay_start: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    decay_step = int(max_steps * decay_start)
    if step < decay_step:
        return base_lr
    if step >= max_steps:
        return min_lr
    progress = (step - decay_step) / max(1, max_steps - decay_step)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    train_ds, val_ds, tok = build_bpe_datasets(
        dataset=cfg["data"].get("dataset", "wikitext-103"),
        cache_dir=cfg["data"]["cache_dir"],
        seq_len=cfg["data"]["seq_len"],
        train_split=cfg["data"]["train_split"],
        max_train_tokens=cfg["data"].get("max_train_tokens"),
    )
    print(f"Vocab: {tok.vocab_size} | train: {len(train_ds):,} | val: {len(val_ds):,}")

    device = torch.device("cuda:0")
    train_sampler = RandomWindowSampler(train_ds, cfg["training"]["batch_size"], device)
    val_sampler = RandomWindowSampler(val_ds, cfg["training"]["batch_size"], device)

    model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
    model_cfg["vocab_size"] = tok.vocab_size
    model = Mamba3LanguageModel(**model_cfg)

    dtype = torch.float16 if cfg["training"].get("dtype", "fp16") == "fp16" else torch.bfloat16
    model = model.to(device).to(dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Mamba-3 {n_params/1e6:.2f}M params on {device} ({dtype})")

    if torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler() if dtype == torch.float16 else None

    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    resume_path = args.resume
    if resume_path is None:
        for name in ("latest.pt", "best.pt"):
            cand = ckpt_dir / name
            if cand.exists():
                resume_path = str(cand)
                break
    start_step = 0
    best_val = float("inf")
    if resume_path and Path(resume_path).exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        (model.module if hasattr(model, "module") else model).load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        best_val = state.get("best_val_loss", float("inf"))
        print(f"Resumed at step {start_step}, best_val={best_val:.4f}")

    history = {"step": [], "train_loss": [], "val_loss": []}
    t0 = time.time()

    def save_ckpt(tag: str) -> None:
        torch.save({
            "model": (model.module if hasattr(model, "module") else model).state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "step": step + 1,
            "best_val_loss": best_val,
        }, ckpt_dir / f"{tag}.pt")

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        losses: list[float] = []
        for _ in range(cfg["training"]["eval_iters"]):
            x, y = val_sampler.sample()
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(x, y)
            losses.append(loss.mean().item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    tc = cfg["training"]
    for step in range(start_step, tc["max_steps"]):
        lr = wsd_lr(step, tc["warmup_steps"], tc["max_steps"],
                    tc["lr"], tc["min_lr"], tc.get("wsd_decay_start", 0.8))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(tc["grad_accum_steps"]):
            x, y = train_sampler.sample()
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(x, y)
                loss = loss.mean() / tc["grad_accum_steps"]
            total_loss += loss.item() * tc["grad_accum_steps"]
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        step_loss = total_loss / tc["grad_accum_steps"]

        if (step + 1) % cfg["logging"]["log_every"] == 0:
            dt = time.time() - t0
            rate = (step + 1 - start_step) / dt if dt > 0 else 0
            print(f"step {step+1}/{tc['max_steps']} loss={step_loss:.4f} "
                  f"lr={lr:.6f} {rate:.2f} it/s")

        if (step + 1) % tc["ckpt_every"] == 0:
            save_ckpt("latest")

        if (step + 1) % tc["eval_every"] == 0 or (step + 1) == tc["max_steps"]:
            val_loss = evaluate()
            history["step"].append(step + 1)
            history["train_loss"].append(step_loss)
            history["val_loss"].append(val_loss)
            print(f"  step {step+1} val_loss={val_loss:.4f} "
                  f"val_ppl={math.exp(val_loss):.2f}")
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt("best")

    save_ckpt("final")
    summary = {
        "n_params": n_params,
        "best_val_loss": best_val,
        "elapsed_seconds": time.time() - t0,
        "history": history,
    }
    (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. best_val={best_val:.4f}, elapsed={summary['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
