"""Maximal Update Parametrization (muP) for ARIA hybrid models.

muP allows transferring hyperparameters (LR, init scale) from a small proxy
model to a larger target model. Validated at scale by Cerebras, EleutherAI,
and used implicitly by most frontier labs.

Key idea: scale weight init and per-layer LR multipliers so that the
magnitude of activations and updates stays constant across model widths.

For a standard transformer with width ``d_model``:
- Embedding init: std = 1 (not scaled)
- Hidden layer weights: std = 1 / sqrt(d_model)
- Output layer weight: std = 1 / d_model
- LR for hidden layers: base_lr * (base_width / d_model)
- LR for embedding: base_lr (not scaled)
- LR for output: base_lr * (base_width / d_model)

For our HYBRID model, the SSM projections (W_A, W_B, W_in_state, W_state_k,
W_state_v) are treated as hidden-layer weights. The MLA down-projection
(w_kv_down) is an embedding-like bottleneck; the up-projections are hidden.

Usage:
    from aria.mup import apply_mup

    model = LSALanguageModel(...)
    apply_mup(model, base_width=128, target_width=model.d_model)
    optimizer = get_mup_optimizer(model, base_lr=3e-4, base_width=128)

References:
- "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot HP Transfer"
  (arXiv 2203.03466, Greg Yang et al.)
- Cerebras muP Practitioner's Guide
- EleutherAI muP Guide
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def _classify_param(name: str, param: nn.Parameter) -> str:
    """Classify a parameter as 'embedding', 'hidden', or 'output' for muP scaling.

    - embedding: token_emb.weight (input embedding)
    - output: lm_head.weight (output projection, tied or not)
    - hidden: everything else (attention, FFN, SSM, MoE)
    """
    if "token_emb" in name:
        return "embedding"
    if "lm_head" in name:
        return "output"
    return "hidden"


def apply_mup_init(model: nn.Module, base_width: int, target_width: int) -> None:
    """Apply muP-correct weight initialization.

    Scales init std based on the ratio of base_width to target_width so that
    activations have consistent magnitude across widths.

    Args:
        model: The model to initialize.
        base_width: The d_model of the proxy model where HP were tuned.
        target_width: The d_model of the actual model being trained.
    """
    ratio = base_width / target_width  # < 1 when scaling up

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        ptype = _classify_param(name, param)

        if ptype == "embedding":
            # Embedding: standard init, not scaled by width
            nn.init.normal_(param, mean=0.0, std=1.0)
        elif ptype == "output":
            # Output: scale down by width for stable logits
            nn.init.normal_(param, mean=0.0, std=ratio)
        else:
            # Hidden: standard muP init
            fan_in = param.shape[-1] if param.dim() >= 2 else param.shape[0]
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(param, mean=0.0, std=std)


def get_mup_param_groups(
    model: nn.Module,
    base_lr: float,
    base_width: int,
    target_width: int,
    weight_decay: float = 0.1,
) -> list[dict[str, Any]]:
    """Create optimizer param groups with muP-correct per-layer LR scaling.

    Hidden-layer parameters get LR scaled by (base_width / target_width).
    Embedding and output parameters keep the base LR.
    1D parameters (biases, norms) get no weight decay.

    Returns param groups suitable for AdamW.
    """
    width_ratio = base_width / target_width

    groups: dict[str, dict[str, Any]] = {
        "embedding_decay": {"params": [], "lr": base_lr, "weight_decay": weight_decay},
        "embedding_no_decay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        "hidden_decay": {"params": [], "lr": base_lr * width_ratio, "weight_decay": weight_decay},
        "hidden_no_decay": {"params": [], "lr": base_lr * width_ratio, "weight_decay": 0.0},
        "output_decay": {"params": [], "lr": base_lr * width_ratio, "weight_decay": weight_decay},
        "output_no_decay": {"params": [], "lr": base_lr * width_ratio, "weight_decay": 0.0},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        ptype = _classify_param(name, param)
        has_decay = param.dim() >= 2
        key = f"{ptype}_{'decay' if has_decay else 'no_decay'}"
        groups[key]["params"].append(param)

    # Filter out empty groups
    return [g for g in groups.values() if g["params"]]


def get_mup_optimizer(
    model: nn.Module,
    base_lr: float,
    base_width: int,
    target_width: int | None = None,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """Create an AdamW optimizer with muP-correct LR scaling.

    Convenience wrapper around get_mup_param_groups.
    If target_width is None, infers it from model.d_model.
    """
    if target_width is None:
        target_width = getattr(model, "d_model", base_width)

    param_groups = get_mup_param_groups(
        model, base_lr, base_width, target_width, weight_decay,
    )
    return torch.optim.AdamW(param_groups, betas=betas)


def mup_summary(model: nn.Module, base_width: int, base_lr: float = 3e-4) -> str:
    """Print a summary of muP scaling for a model."""
    target_width = getattr(model, "d_model", base_width)
    ratio = base_width / target_width

    lines = [
        f"muP Summary: base_width={base_width}, target_width={target_width}, ratio={ratio:.4f}",
        f"  Embedding LR: {base_lr:.6f}",
        f"  Hidden LR:    {base_lr * ratio:.6f}",
        f"  Output LR:    {base_lr * ratio:.6f}",
        "",
    ]

    counts = {"embedding": 0, "hidden": 0, "output": 0}
    for name, param in model.named_parameters():
        if param.requires_grad:
            ptype = _classify_param(name, param)
            counts[ptype] += param.numel()

    for ptype, count in counts.items():
        lines.append(f"  {ptype}: {count:,} params")

    return "\n".join(lines)
