"""Minimal training harness.

Features:
- AdamW with weight decay excluded from biases and norms
- Cosine LR schedule with linear warmup
- Gradient accumulation
- Optional mixed precision (off by default — Pascal GPUs lack tensor cores)
- Periodic eval with perplexity reporting
- Checkpointing (best val loss)
- No external deps beyond torch + tqdm

Deliberately small. Replace with a more capable harness in Phase 1+.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .data import RandomWindowSampler


@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 3000
    eval_every: int = 200
    eval_iters: int = 50
    warmup_steps: int = 100
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    amp: bool = False
    device: str = "auto"
    seed: int = 42
    log_every: int = 20
    checkpoint_dir: str = "./checkpoints/run"
    run_name: str = "run"
    lr_schedule: str = "cosine"       # "cosine" or "wsd"
    wsd_decay_start: float = 0.8      # fraction of max_steps where decay begins


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def split_weight_decay_params(model: nn.Module, weight_decay: float) -> list[dict]:
    """AdamW param groups: decay applied only to 2D weights (Linear, Embedding).

    Biases, LayerNorm/RMSNorm weights, and 1D params are excluded from decay.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_lr(step: int, warmup: int, max_steps: int,
              base_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def wsd_lr(step: int, warmup: int, max_steps: int,
           base_lr: float, min_lr: float, decay_start: float = 0.8) -> float:
    """Warmup-Stable-Decay schedule (WSD).

    Used by frontier labs (DeepSeek, Qwen, Nemotron) as a replacement for
    cosine. Key advantage: no need to pre-commit total compute budget during
    the stable phase. The decay phase produces sharp loss drops that match
    or beat cosine final loss.

    Three phases:
      1. Warmup: linear ramp from 0 to base_lr over warmup steps
      2. Stable: constant base_lr (the bulk of training)
      3. Decay: cosine decay from base_lr to min_lr

    ``decay_start`` is the fraction of max_steps where the decay phase begins
    (default 0.8 = last 20% of training is decay).
    """
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    decay_step = int(max_steps * decay_start)
    if step < decay_step:
        return base_lr
    if step >= max_steps:
        return min_lr
    progress = (step - decay_step) / max(1, max_steps - decay_step)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, model: nn.Module, train_sampler: RandomWindowSampler,
                 val_sampler: RandomWindowSampler, config: TrainConfig):
        self.model = model
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.cfg = config

        self.device = resolve_device(config.device)
        self.model.to(self.device)

        param_groups = split_weight_decay_params(model, config.weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

        self.scaler = GradScaler(enabled=config.amp and self.device.type == "cuda")
        self.step = 0
        self.best_val_loss = float("inf")

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        losses = []
        for _ in range(self.cfg.eval_iters):
            x, y = self.val_sampler.sample()
            with autocast(device_type=self.device.type,
                          enabled=self.cfg.amp and self.device.type == "cuda"):
                _, loss = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        mean_loss = sum(losses) / len(losses)
        return {"val_loss": mean_loss, "val_ppl": math.exp(mean_loss)}

    def save_checkpoint(self, tag: str, extra: dict[str, Any] | None = None) -> None:
        path = Path(self.cfg.checkpoint_dir) / f"{tag}.pt"
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "best_val_loss": self.best_val_loss,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume from a checkpoint. Restores model, optimizer, step, best loss.

        Required for Kaggle-style interrupted sessions where training must
        pick up from where it stopped. Call after optimizer construction but
        before ``fit``.
        """
        path = Path(path)
        if not path.exists():
            print(f"  no checkpoint at {path}, starting from scratch")
            return
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state.get("step", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        print(f"  resumed from {path} at step {self.step}, "
              f"best val loss {self.best_val_loss:.4f}")

    def train_step(self) -> float:
        """One optimizer step = grad_accum_steps micro-batches."""
        self.model.train()
        total_loss = 0.0

        if self.cfg.lr_schedule == "wsd":
            lr = wsd_lr(self.step, self.cfg.warmup_steps, self.cfg.max_steps,
                        self.cfg.lr, self.cfg.min_lr, self.cfg.wsd_decay_start)
        else:
            lr = cosine_lr(self.step, self.cfg.warmup_steps, self.cfg.max_steps,
                           self.cfg.lr, self.cfg.min_lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        for _ in range(self.cfg.grad_accum_steps):
            x, y = self.train_sampler.sample()
            with autocast(device_type=self.device.type,
                          enabled=self.cfg.amp and self.device.type == "cuda"):
                _, loss = self.model(x, y)
                loss = loss / self.cfg.grad_accum_steps
            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.cfg.grad_accum_steps

        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return total_loss / self.cfg.grad_accum_steps

    def fit(self) -> dict[str, Any]:
        print(f"Training {self.cfg.run_name} on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        history = {"train_loss": [], "val_loss": [], "val_ppl": [], "step": []}
        t0 = time.time()

        pbar = tqdm(range(self.cfg.max_steps), desc=self.cfg.run_name)
        for _ in pbar:
            loss = self.train_step()
            self.step += 1

            if self.step % self.cfg.log_every == 0:
                pbar.set_postfix(loss=f"{loss:.4f}")

            if self.step % self.cfg.eval_every == 0 or self.step == self.cfg.max_steps:
                metrics = self.evaluate()
                history["train_loss"].append(loss)
                history["val_loss"].append(metrics["val_loss"])
                history["val_ppl"].append(metrics["val_ppl"])
                history["step"].append(self.step)
                tqdm.write(
                    f"  step {self.step:6d} | train {loss:.4f} | "
                    f"val {metrics['val_loss']:.4f} | val ppl {metrics['val_ppl']:.2f}"
                )
                if metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = metrics["val_loss"]
                    self.save_checkpoint("best")

        self.save_checkpoint("final")
        elapsed = time.time() - t0
        print(f"Training complete in {elapsed:.1f}s. Best val loss: {self.best_val_loss:.4f}")
        return {
            "history": history,
            "best_val_loss": self.best_val_loss,
            "elapsed_seconds": elapsed,
        }
