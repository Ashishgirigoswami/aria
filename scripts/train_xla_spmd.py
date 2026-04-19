"""Multi-host TPU training via GSPMD (torch_xla 2.9+).

Canonical 2.9 launcher for multi-host TPU pods (v4-32 = 4 hosts × 4 chips = 16
devices). Unlike xmp.spawn — which double-initializes PJRT on v4 and aborts
with "Check failed: reporting_closure_ == nullptr" — SPMD runs as a single
python3 process per host. PJRT auto-joins each host's process into a global
ICI mesh; the XLA compiler shards the model program and handles all-reduce
internally. No explicit gradient sync needed.

Launch (from laptop, fans out to every worker in parallel):
    gcloud compute tpus tpu-vm ssh aria-v4-32-spot \\
        --zone=us-central2-b --worker=all \\
        --command="cd ~/aria && PJRT_DEVICE=TPU XLA_USE_SPMD=1 \\
                   python3 scripts/train_xla_spmd.py \\
                     --config configs/aria_v1_160m_multihost.yaml \\
                     2>&1 | tee -a logs/spmd_w\\$TPU_WORKER_ID.log"

Effective batch = batch_size * grad_accum * global_device_count (all handled
by SPMD sharding on dim-0 of the input).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def _pull_from_gcs(remote: str, local: Path) -> bool:
    """Best-effort ``gsutil cp`` of a single object. Returns True on success.

    Used to sync checkpoints across TPU-VM hosts via the GCS mirror, which
    do not share a local filesystem. Silent no-op if gsutil is missing or
    the remote object does not exist — callers must treat a False return
    as "start from random init".
    """
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["gsutil", "-q", "cp", remote, str(local)],
            capture_output=True, text=True, timeout=600,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and local.exists()

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# torch_xla + SPMD setup MUST come before any XLA tensor is allocated.
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

xr.use_spmd()  # Enables SPMD mode — single process per host, mesh auto-formed.

# Import model/data modules AFTER SPMD init.
from aria import lsa_xla  # noqa: F401  (patches SSM scan for XLA)
from aria.lsa import LSALanguageModel
from aria.lsa_v2 import LSAv2LanguageModel
from aria.baseline import BaselineLanguageModel
from aria.data import build_datasets, build_bpe_datasets, RandomWindowSampler
from aria.nn_utils import count_parameters


MODEL_REGISTRY = {
    "lsa": LSALanguageModel,
    "lsa_v2": LSAv2LanguageModel,
    "baseline": BaselineLanguageModel,
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _wsd_lr(step: int, warmup: int, max_steps: int,
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


def _cosine_lr(step: int, warmup: int, max_steps: int,
               base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_model(cfg: dict, vocab_size: int) -> nn.Module:
    mcfg = dict(cfg["model"])
    name = mcfg.pop("name")
    mcfg["vocab_size"] = vocab_size
    return MODEL_REGISTRY[name](**mcfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    _set_seed(cfg["training"]["seed"])

    device = xm.xla_device()
    process_idx = xr.process_index()       # 0..n_hosts-1
    process_count = xr.process_count()     # n_hosts
    global_devices = xr.global_runtime_device_count()  # 16 on v4-32
    is_master = (process_idx == 0)

    # Build 1D mesh along 'data' axis → pure data parallelism.
    # (For FSDP/TP extend to 2D: mesh_shape=(dp, mp).)
    device_ids = np.arange(global_devices)
    mesh = Mesh(device_ids, (global_devices,), ("data",))
    xs.set_global_mesh(mesh)

    if is_master:
        print(f"[spmd] process {process_idx}/{process_count}, "
              f"global_devices={global_devices}, mesh=({global_devices},)",
              flush=True)

    # Data — each host builds the cache from disk (pre-tokenize guaranteed by
    # the separate pretokenize step run before training).
    tok_kind = cfg["data"].get("tokenizer", "bpe")
    if tok_kind == "bpe":
        train_ds, val_ds, tok = build_bpe_datasets(
            dataset=cfg["data"].get("dataset", "wikitext-103"),
            cache_dir=cfg["data"]["cache_dir"],
            seq_len=cfg["data"]["seq_len"],
            train_split=cfg["data"]["train_split"],
            max_train_tokens=cfg["data"].get("max_train_tokens"),
        )
    else:
        train_ds, val_ds, tok = build_datasets(
            cache_dir=cfg["data"]["cache_dir"],
            seq_len=cfg["data"]["seq_len"],
            train_split=cfg["data"]["train_split"],
        )

    # Global batch = per-step_batch (config) * grad_accum * global_devices.
    # Our sampler returns a global batch; SPMD shards along dim 0 onto
    # the mesh's 'data' axis, so each chip sees batch/global_devices rows.
    global_batch = cfg["training"]["batch_size"] * global_devices
    train_sampler = RandomWindowSampler(train_ds, global_batch, device)
    val_sampler = RandomWindowSampler(val_ds, global_batch, device)

    model = _build_model(cfg, vocab_size=tok.vocab_size).to(device)
    # Re-tie weights post-.to() — XLA can break tied Parameters.
    if hasattr(model, "lm_head") and hasattr(model, "token_emb"):
        if model.lm_head.weight is not model.token_emb.weight:
            model.lm_head.weight = model.token_emb.weight

    # Replicate all weights across the data mesh (pure DP).
    # Weights: replicate → gradients auto-all-reduce at compile time.
    for p in model.parameters():
        if p.dim() > 0:
            xs.mark_sharding(p, mesh, (None,) * p.dim())

    n_params = count_parameters(model)
    if is_master:
        print(f"[spmd] model {cfg['model']['name']} {n_params/1e6:.2f}M, "
              f"global_batch={global_batch}, per-chip_batch="
              f"{cfg['training']['batch_size']}", flush=True)
        print(f"[spmd] vocab={tok.vocab_size}, "
              f"train tokens={len(train_ds.data):,}, "
              f"val tokens={len(val_ds.data):,}", flush=True)

    tc = cfg["training"]
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]),
    )

    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    # mkdir on ALL workers — xm.save() writes from every rank even though only
    # master persists to disk. Missing dir crashes non-master ranks mid-training.
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step, best_val = 0, float("inf")
    resume_path = args.resume
    if resume_path is None:
        for name in ("latest.pt", "best.pt"):
            cand = ckpt_dir / name
            if cand.exists():
                resume_path = str(cand)
                break

    # Multi-host TPU VMs do NOT share a filesystem. After a pod restart, only
    # the worker that last held rank-0 has latest.pt; every other host starts
    # from random init unless we explicitly sync. Pull from GCS if it's the
    # only place the checkpoint actually lives.
    gcs_mirror = os.environ.get("ARIA_GCS_CKPT_PREFIX") or (
        f"gs://aria-trc-ckpts/{cfg['logging']['run_name']}"
    )
    if resume_path is None and gcs_mirror:
        remote = f"{gcs_mirror}/latest.pt"
        local = ckpt_dir / "latest.pt"
        if _pull_from_gcs(remote, local):
            resume_path = str(local)
            if is_master:
                print(f"[spmd] pulled resume ckpt from {remote}", flush=True)

    if resume_path and Path(resume_path).exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))
        best_val = float(state.get("best_val_loss", float("inf")))
        if is_master:
            print(f"[spmd] resumed from {resume_path} step {start_step}, "
                  f"best_val={best_val:.4f}", flush=True)

    def evaluate() -> float:
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for _ in range(tc["eval_iters"]):
                x, y = val_sampler.sample()
                xs.mark_sharding(x, mesh, ("data", None))
                xs.mark_sharding(y, mesh, ("data", None))
                _, loss = model(x, y)
                torch_xla.sync()
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    def save_ckpt(tag: str, step_val: int) -> None:
        # xm.save gathers replicated params and writes once on master.
        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step_val,
            "best_val_loss": best_val,
        }
        xm.save(state, str(ckpt_dir / f"{tag}.pt"))

    history: dict[str, list] = {"step": [], "train_loss": [], "val_loss": []}
    lr_sched = tc.get("lr_schedule", "cosine")
    t0 = time.time()
    model.train()

    for step in range(start_step, tc["max_steps"]):
        lr = (_wsd_lr(step, tc["warmup_steps"], tc["max_steps"],
                      tc["lr"], tc["min_lr"], tc.get("wsd_decay_start", 0.8))
              if lr_sched == "wsd" else
              _cosine_lr(step, tc["warmup_steps"], tc["max_steps"],
                         tc["lr"], tc["min_lr"]))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(tc["grad_accum_steps"]):
            x, y = train_sampler.sample()
            xs.mark_sharding(x, mesh, ("data", None))
            xs.mark_sharding(y, mesh, ("data", None))
            _, loss = model(x, y)
            loss = loss / tc["grad_accum_steps"]
            torch_xla.sync()
            total_loss += loss.item() * tc["grad_accum_steps"]
            loss.backward()
            torch_xla.sync()

        if tc.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        xm.mark_step()

        step_loss = total_loss / tc["grad_accum_steps"]
        cur = step + 1

        if is_master and cur % cfg["logging"]["log_every"] == 0:
            dt = time.time() - t0
            rate = (cur - start_step) / dt if dt > 0 else 0
            print(f"step {cur}/{tc['max_steps']} loss={step_loss:.4f} "
                  f"lr={lr:.6f} {rate:.2f} it/s", flush=True)

        if tc.get("ckpt_every", 0) > 0 and cur % tc["ckpt_every"] == 0:
            save_ckpt("latest", cur)

        if cur % tc["eval_every"] == 0 or cur == tc["max_steps"]:
            val_loss = evaluate()
            if is_master:
                history["step"].append(cur)
                history["train_loss"].append(step_loss)
                history["val_loss"].append(val_loss)
                print(f"  step {cur} val_loss={val_loss:.4f} "
                      f"val_ppl={math.exp(val_loss):.2f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt("best", cur)

    save_ckpt("final", tc["max_steps"])
    if is_master:
        summary = {
            "n_params": n_params,
            "global_devices": global_devices,
            "best_val_loss": best_val,
            "elapsed_seconds": time.time() - t0,
            "history": history,
        }
        (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[spmd] done. best_val={best_val:.4f}, "
              f"elapsed={summary['elapsed_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
