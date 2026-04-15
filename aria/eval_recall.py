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
    """Configuration for the synthetic passkey retrieval task.

    Vocab slicing notes — important for a meaningful metric:

    - ``filler_range`` and ``passkey_range`` both draw from the COMMON region
      of the BPE vocabulary. GPT-2's last ~10k tokens are Unicode fragments
      and rare byte pairs that wikitext-trained models assign ~0 probability
      to; drawing passkeys from that region produces losses *above* the
      uniform-prediction floor because the model's training prior actively
      fights the correct answer. Stick to indices < ~30_000 for natural tokens.

    - ``run_control``, if True, additionally measures loss at the same
      positions with a RANDOM unseen token substituted for the second passkey.
      The retrieval signal is the delta:  control_loss - retrieval_loss.
      This isolates in-context induction from the model's generic token prior.
    """

    context_lengths: tuple[int, ...] = (128, 256, 512, 1024)
    depths: tuple[float, ...] = (0.1, 0.5, 0.9)  # where in the context the passkey is placed
    passkey_len: int = 8                          # how many tokens the passkey spans
    n_trials: int = 16                            # samples per (length, depth) cell
    seed: int = 20260415                          # deterministic task construction
    filler_range: tuple[int, int] = (0, 30_000)   # BPE ids for filler tokens
    passkey_range: tuple[int, int] = (0, 30_000)  # BPE ids for passkey tokens
    run_control: bool = True                      # also measure the no-retrieval baseline


def build_passkey_batch(
    cfg: PasskeyConfig,
    vocab_size: int,
    context_length: int,
    depth: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch of passkey retrieval samples at one (length, depth) cell.

    Returns:
        input_retrieve: (n_trials, context_length) int64 — the passkey is
            written at BOTH the first_start and second_start positions, so
            the model can retrieve it in-context.
        input_control:  (n_trials, context_length) int64 — identical filler
            except the second_start position is filled with a FRESH random
            token that does not appear earlier in the sequence. Provides the
            no-retrieval baseline.
        target_ids:     (n_trials, context_length) int64 — shifted-by-one
            targets masked to only score the passkey_len tokens at the
            second_start region. Shared between both inputs (the "correct"
            answer is always the retrieval passkey; loss on ``input_control``
            measures how well the model predicts that passkey without having
            seen it).
        passkey_positions: (n_trials,) int64 — diagnostic.
    """
    n = cfg.n_trials
    k = cfg.passkey_len
    usable = context_length - 2 * k
    assert usable > 0, f"context_length {context_length} too short for passkey_len {k}"
    first_start = int(depth * usable)
    second_start = context_length - k

    f_lo, f_hi = cfg.filler_range
    p_lo, p_hi = cfg.passkey_range
    f_hi = min(f_hi, vocab_size)
    p_hi = min(p_hi, vocab_size)
    assert f_lo < f_hi and p_lo < p_hi

    # Base filler sequence.
    base = rng.integers(f_lo, f_hi, size=(n, context_length), dtype=np.int64)

    # Passkey tokens — drawn from the natural vocab range.
    passkeys = rng.integers(p_lo, p_hi, size=(n, k), dtype=np.int64)

    # Control tokens — different random tokens at the second_start position,
    # used only in ``input_control``. This gives us a "no prior occurrence"
    # baseline per sample.
    control_tokens = rng.integers(p_lo, p_hi, size=(n, k), dtype=np.int64)

    input_retrieve = base.copy()
    input_retrieve[:, first_start : first_start + k] = passkeys
    input_retrieve[:, second_start : second_start + k] = passkeys

    input_control = base.copy()
    input_control[:, first_start : first_start + k] = passkeys
    input_control[:, second_start : second_start + k] = control_tokens

    # Target: always the retrieval passkey at the second_start position.
    target_ids = np.full_like(base, fill_value=-100)
    tgt_start = second_start - 1
    tgt_end = second_start + k - 1
    assert tgt_end <= context_length - 1
    target_ids[:, tgt_start:tgt_end] = passkeys

    return (
        torch.from_numpy(input_retrieve),
        torch.from_numpy(input_control),
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
    all_control: list[float] = []
    all_deltas: list[float] = []

    def _masked_ce_and_top1(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        logits, _ = model(x)
        V = logits.size(-1)
        flat_logits = logits.view(-1, V)
        flat_targets = y.view(-1)
        mask = flat_targets != -100
        ce = F.cross_entropy(
            flat_logits[mask], flat_targets[mask], reduction="mean"
        ).item()
        preds = flat_logits[mask].argmax(dim=-1)
        top1 = (preds == flat_targets[mask]).float().mean().item()
        return ce, top1

    for L in cfg.context_lengths:
        if max_seq_len is not None and L > max_seq_len:
            continue
        grid[str(L)] = {}
        for d in cfg.depths:
            xr, xc, y, _ = build_passkey_batch(cfg, vocab_size, L, d, rng)
            xr = xr.to(device)
            xc = xc.to(device)
            y = y.to(device)

            retrieval_ce, retrieval_top1 = _masked_ce_and_top1(xr, y)
            cell: dict[str, float] = {
                "loss": retrieval_ce,
                "top1_accuracy": retrieval_top1,
            }

            if cfg.run_control:
                control_ce, _ = _masked_ce_and_top1(xc, y)
                # Delta > 0 means retrieval helps: the model predicts the
                # passkey more confidently when it has seen it earlier in
                # context than when it has not. This is the induction signal.
                delta = control_ce - retrieval_ce
                cell["control_loss"] = control_ce
                cell["retrieval_delta"] = delta
                all_control.append(control_ce)
                all_deltas.append(delta)

            grid[str(L)][str(d)] = cell
            all_losses.append(retrieval_ce)
            all_top1.append(retrieval_top1)

    summary: dict[str, float] = {
        "mean_retrieval_loss": float(np.mean(all_losses)) if all_losses else float("nan"),
        "mean_top1": float(np.mean(all_top1)) if all_top1 else float("nan"),
    }
    if cfg.run_control and all_control:
        summary["mean_control_loss"] = float(np.mean(all_control))
        summary["mean_retrieval_delta"] = float(np.mean(all_deltas))

    return {
        "config": {
            "context_lengths": list(cfg.context_lengths),
            "depths": list(cfg.depths),
            "passkey_len": cfg.passkey_len,
            "n_trials": cfg.n_trials,
            "seed": cfg.seed,
            "vocab_size": vocab_size,
            "filler_range": list(cfg.filler_range),
            "passkey_range": list(cfg.passkey_range),
            "run_control": cfg.run_control,
        },
        "grid": grid,
        "summary": summary,
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
