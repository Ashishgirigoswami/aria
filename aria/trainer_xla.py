"""TPU training harness using torch_xla.

Functionally mirrors ``aria.trainer.Trainer`` but adapted for XLA devices:

- Device comes from ``xm.xla_device()`` instead of ``torch.device("cuda")``.
- Optimizer updates go through ``xm.optimizer_step(optimizer)`` so XLA
  synchronises the distributed gradient reduction on multi-core TPU pods.
- ``xm.mark_step()`` is called after each logical step to fence the lazy
  tensor graph and trigger XLA compilation/execution.
- No ``torch.cuda.amp.GradScaler``. TPUs run bf16 natively without needing
  loss scaling; we rely on the model's internal ``to(bfloat16)`` conversion
  plus the ``XLA_USE_BF16=1`` env var that Colab / TPU VMs set by default.
- Checkpoint saving uses ``xm.save`` to serialise XLA tensors correctly.

Everything else (cosine schedule, param group split, eval loop structure)
is copied from the GPU trainer to keep the learning curves directly
comparable.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

import torch_xla

# torch_xla is optional at import time so the file still parses on machines
# without it (the GPU trainer remains usable).
try:
    import torch_xla.core.xla_model as xm
except ImportError:  # pragma: no cover - handled at runtime
    xm = None

from .data import RandomWindowSampler


@dataclass
class XLATrainConfig:
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 20000
    eval_every: int = 500
    eval_iters: int = 100
    warmup_steps: int = 500
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 50
    checkpoint_dir: str = "./checkpoints/xla_run"
    run_name: str = "xla_run"


def _require_xla() -> None:
    if xm is None:
        raise RuntimeError(
            "torch_xla is not installed. Run the TPU trainer on a TPU-enabled "
            "environment (Colab TPU runtime, TPU VM, or `pip install torch_xla`)."
        )


def split_weight_decay_params(model: nn.Module, weight_decay: float) -> list[dict]:
    """Same param-group split as the GPU trainer. 2D weights get decay;
    biases and 1D norm scales do not."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_lr(step: int, warmup: int, max_steps: int,
              base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


class XLATrainer:
    """Minimal TPU trainer. Single-core for now (TPU v*-1 slice).

    Multi-core pod training via ``xmp.spawn`` should be added in a follow-up
    once the single-core path is verified on Colab TPU.
    """

    def __init__(self, model: nn.Module,
                 train_sampler: RandomWindowSampler,
                 val_sampler: RandomWindowSampler,
                 config: XLATrainConfig):
        _require_xla()
        self.model = model
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.cfg = config

        self.device = torch_xla.device()
        self.model.to(self.device)

        # Re-tie weights after .to(device) — XLA's .to() can break Parameter
        # sharing, causing lm_head and token_emb to get independent copies.
        # Unconditionally re-assign if the model was built with tie_weights.
        if hasattr(model, 'lm_head') and hasattr(model, 'token_emb'):
            if model.lm_head.weight is not model.token_emb.weight:
                model.lm_head.weight = model.token_emb.weight

        param_groups = split_weight_decay_params(model, config.weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

        self.step = 0
        self.best_val_loss = float("inf")
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        for _ in range(self.cfg.eval_iters):
            x, y = self.val_sampler.sample()
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            torch_xla.sync()
            losses.append(loss.item())
        self.model.train()
        mean_loss = sum(losses) / len(losses)
        return {"val_loss": mean_loss, "val_ppl": math.exp(mean_loss)}

    def save_checkpoint(self, tag: str) -> None:
        path = Path(self.cfg.checkpoint_dir) / f"{tag}.pt"
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "best_val_loss": self.best_val_loss,
        }
        xm.save(state, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            print(f"  no checkpoint at {path}, starting from scratch")
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state.get("step", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        print(f"  resumed from {path} at step {self.step}, "
              f"best val loss {self.best_val_loss:.4f}")

    def train_step(self) -> float:
        self.model.train()
        total_loss = 0.0

        lr = cosine_lr(self.step, self.cfg.warmup_steps, self.cfg.max_steps,
                       self.cfg.lr, self.cfg.min_lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(self.cfg.grad_accum_steps):
            x, y = self.train_sampler.sample()
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            loss = loss / self.cfg.grad_accum_steps
            # Sync after forward to materialize the computation graph before
            # backward. Without this, the combined forward+backward graph can
            # produce numerical issues with the custom SSM scan autograd.
            torch_xla.sync()
            total_loss += loss.item() * self.cfg.grad_accum_steps
            loss.backward()
            torch_xla.sync()

        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        xm.optimizer_step(self.optimizer)
        torch_xla.sync()
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
