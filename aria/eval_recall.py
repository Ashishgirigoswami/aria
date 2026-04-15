"""Synthetic recall evaluation for LSA and baseline language models.

The Phase 0 regularization finding was measured in language-modeling validation
loss. The hybrid-architecture literature (arXiv 2507.06457) warns that LM loss
is nearly flat across different hybrid configurations at scale — recall metrics
(passkey, needle-in-a-haystack) are the discriminating signal.

This module provides a **passkey retrieval task** that works on any base LM
(no instruction tuning required) by measuring loss on the repeated passkey
tokens rather than requiring generation.

The task construction:

    [ prefix_filler  tokens ]
    [ marker_A       "="   ]                # marks where the passkey is written
    [ passkey        tokens ]                # e.g. 5 random digit tokens
    [ middle_filler  tokens ]                # the retrieval distance
    [ marker_A       "="   ]                # same marker reappears
    [ passkey        tokens ]                # <-- loss is measured here

The model must attend back across the middle_filler to recover the passkey
at the second marker. Since the passkey is random, the only way to get low
loss on those final tokens is to retrieve them from earlier in the context.
Average cross-entropy on the final passkey tokens is the recall score.

Reported as a grid over (context_length, passkey_depth) so we can see where
the model's recall breaks down — exactly the same shape as a NIAH heatmap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PasskeyConfig:
    """Configuration for the synthetic passkey retrieval task."""

    context_lengths: tuple[int, ...] = (128, 256, 512, 1024)
    depths: tuple[float, ...] = (0.1, 0.5, 0.9)  # where in the context the passkey is placed
    passkey_len: int = 8                          # how many tokens the passkey spans
    n_trials: int = 16                            # samples per (length, depth) cell
    seed: int = 20260415                          # deterministic task construction


def build_passkey_batch(
    cfg: PasskeyConfig,
    vocab_size: int,
    context_length: int,
    depth: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch of passkey retrieval samples at one (length, depth) cell.

    Returns:
        input_ids: (n_trials, context_length) int64
        target_ids: (n_trials, context_length) int64 — shifted by one, with -100
            on every position EXCEPT the final ``passkey_len`` tokens, so that
            ``F.cross_entropy(ignore_index=-100)`` computes loss only on the
            recalled passkey.
        passkey_positions: (n_trials,) int64 — start index of the first passkey
            for diagnostics.
    """
    n = cfg.n_trials
    k = cfg.passkey_len
    # Need room for: first_passkey (k) + middle_filler + second_passkey (k).
    # The second passkey must end at position context_length - 1.
    # The first passkey starts at int(depth * (context_length - 2*k)).
    usable = context_length - 2 * k
    assert usable > 0, f"context_length {context_length} too short for passkey_len {k}"
    first_start = int(depth * usable)
    second_start = context_length - k

    # Random filler — draw from a safe subset of the vocab to avoid special tokens.
    # GPT-2 BPE vocab is 50257; token 0 is '!', token 50256 is '<|endoftext|>'.
    # Avoid the last 256 tokens to dodge reserved/special territory.
    filler_hi = min(vocab_size, 50000)
    input_ids = rng.integers(0, filler_hi, size=(n, context_length), dtype=np.int64)

    # Draw passkey tokens from a DIFFERENT slice so they are "rare" relative to filler,
    # making accidental prediction unlikely.
    passkey_lo = max(0, vocab_size - 10_000)
    passkey_hi = vocab_size
    passkeys = rng.integers(passkey_lo, passkey_hi, size=(n, k), dtype=np.int64)

    # Write the passkey into both positions.
    input_ids[:, first_start : first_start + k] = passkeys
    input_ids[:, second_start : second_start + k] = passkeys

    # Build target_ids: shift by one for next-token prediction, then mask out
    # everything except the second passkey's interior.
    target_ids = np.full_like(input_ids, fill_value=-100)
    # The token at position p is predicted FROM the token at position p-1,
    # so the input position that produces "passkey[i]" is second_start + i - 1.
    # We want predictions at input positions [second_start-1 .. second_start+k-2],
    # predicting targets [passkey[0] .. passkey[k-1]].
    # Equivalently: target_ids[:, second_start-1 : second_start+k-1] = passkeys.
    tgt_start = second_start - 1
    tgt_end = second_start + k - 1
    assert tgt_end <= context_length - 1
    target_ids[:, tgt_start:tgt_end] = passkeys

    return (
        torch.from_numpy(input_ids),
        torch.from_numpy(target_ids),
        torch.from_numpy(np.full(n, first_start, dtype=np.int64)),
    )


@torch.no_grad()
def passkey_eval(
    model: torch.nn.Module,
    vocab_size: int,
    device: torch.device,
    cfg: PasskeyConfig = PasskeyConfig(),
) -> dict[str, Any]:
    """Run passkey retrieval over a grid of (context_length, depth).

    Returns a nested dict::

        {
            "config": {...},
            "grid": {
                "<context_length>": {
                    "<depth>": {
                        "loss": float,            # mean CE on passkey tokens
                        "top1_accuracy": float,   # fraction of passkey tokens the
                                                  # model ranked as top-1
                    }
                }
            },
            "summary": {
                "mean_loss": float,
                "mean_top1": float,
            }
        }

    Lower loss / higher top-1 on longer contexts at deeper positions indicates
    better retrieval.
    """
    model.eval()
    rng = np.random.default_rng(cfg.seed)
    grid: dict[str, dict[str, dict[str, float]]] = {}
    all_losses: list[float] = []
    all_top1: list[float] = []

    max_seq_len = getattr(model, "max_seq_len", None)

    for L in cfg.context_lengths:
        if max_seq_len is not None and L > max_seq_len:
            continue
        grid[str(L)] = {}
        for d in cfg.depths:
            x, y, _ = build_passkey_batch(cfg, vocab_size, L, d, rng)
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)  # ignore the built-in loss; we need masked CE
            # Flatten and compute masked cross-entropy on the passkey positions.
            V = logits.size(-1)
            flat_logits = logits.view(-1, V)
            flat_targets = y.view(-1)
            mask = flat_targets != -100
            if not mask.any():
                continue
            ce = F.cross_entropy(
                flat_logits[mask], flat_targets[mask], reduction="mean"
            ).item()

            # Top-1 accuracy on passkey positions.
            preds = flat_logits[mask].argmax(dim=-1)
            top1 = (preds == flat_targets[mask]).float().mean().item()

            grid[str(L)][str(d)] = {"loss": ce, "top1_accuracy": top1}
            all_losses.append(ce)
            all_top1.append(top1)

    return {
        "config": {
            "context_lengths": list(cfg.context_lengths),
            "depths": list(cfg.depths),
            "passkey_len": cfg.passkey_len,
            "n_trials": cfg.n_trials,
            "seed": cfg.seed,
            "vocab_size": vocab_size,
        },
        "grid": grid,
        "summary": {
            "mean_loss": float(np.mean(all_losses)) if all_losses else float("nan"),
            "mean_top1": float(np.mean(all_top1)) if all_top1 else float("nan"),
        },
    }


def save_recall_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a recall report as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))


# Convenience entry-point: run recall on a trained model from its checkpoint.
def run_from_checkpoint(
    checkpoint_path: str | Path,
    model_factory: Callable[[], torch.nn.Module],
    vocab_size: int,
    out_path: str | Path | None = None,
    cfg: PasskeyConfig = PasskeyConfig(),
) -> dict[str, Any]:
    """Load a checkpoint, run recall eval, optionally save the report.

    model_factory must return a fresh instance of the same architecture class
    (LSALanguageModel or BaselineLanguageModel) that produced the checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = model_factory().to(device)
    model.load_state_dict(state)
    report = passkey_eval(model, vocab_size, device, cfg)
    report["checkpoint"] = str(checkpoint_path)
    if out_path is not None:
        save_recall_report(report, out_path)
    return report
