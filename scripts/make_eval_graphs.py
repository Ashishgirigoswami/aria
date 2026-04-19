"""Generate publication-ready figures for the README:

1. ``figures/loss_curves_131m_160m.png`` — training-loss trajectories for the
   131M wikitext run (from summary.json val history) and the 160M FineWeb-Edu
   run (from spmd.log train-loss prints).
2. ``figures/eval_comparison_160m_vs_pythia.png`` — grouped bar chart of
   acc/acc_norm across 6 standard 0-shot benchmarks for ARIA-131M, ARIA-160M,
   and Pythia-160M (reference).

Inputs:
    docs/eval_160m/*.json           — ARIA-160M per-task eval outputs
    runs/aria_v1_150m_t256_v6eu/eval_v1_131m.json — ARIA-131M eval
    ckpts-pulled/aria_v1_131m_repro/summary.json  — 131M training history
    docs/eval_160m/spmd_train.log                 — 160M training log

Run once after all eval JSONs land.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------- Eval number loaders ----------

def load_160m_eval() -> dict[str, dict[str, float]]:
    """Merge all per-task ARIA-160M JSONs into one flat dict."""
    out: dict[str, dict[str, float]] = {}
    for p in sorted((ROOT / "docs" / "eval_160m").glob("*.json")):
        data = json.loads(p.read_text())
        for task, metrics in data.get("tasks", {}).items():
            clean = {k.split(",")[0]: v for k, v in metrics.items()
                     if not k.endswith("_stderr")}
            out[task] = clean
    return out


def load_131m_eval() -> dict[str, dict[str, float]]:
    path = ROOT / "runs" / "aria_v1_150m_t256_v6eu" / "eval_v1_131m.json"
    data = json.loads(path.read_text())
    out: dict[str, dict[str, float]] = {}
    for task, metrics in data.get("tasks", {}).items():
        clean = {k.split(",")[0]: v for k, v in metrics.items()
                 if not k.endswith("_stderr")}
        out[task] = clean
    return out


# Reference Pythia-160M numbers (from EleutherAI's lm-eval leaderboard + paper).
PYTHIA_160M = {
    "hellaswag": {"acc": 0.285, "acc_norm": 0.336},
    "arc_easy": {"acc": 0.395, "acc_norm": 0.377},
    "arc_challenge": {"acc": 0.199, "acc_norm": 0.253},
    "piqa": {"acc": 0.606, "acc_norm": 0.605},
    "winogrande": {"acc": 0.508},
    "lambada_openai": {"acc": 0.373, "perplexity": 18.0},
    "openbookqa": {"acc": 0.168, "acc_norm": 0.294},
}


# ---------- Training curves ----------

def parse_131m_history(path: Path) -> tuple[list[int], list[float], list[float]]:
    """Return (step, val_loss, train_loss) lists from the 131M summary.json."""
    data = json.loads(path.read_text())
    h = data.get("history", {})
    return h.get("step", []), h.get("val_loss", []), h.get("train_loss", [])


_STEP_RE = re.compile(r"step (\d+)/\d+ loss=([\d.]+) lr=([\d.]+)")


def parse_160m_train_log(path: Path) -> tuple[list[int], list[float], list[float]]:
    """Extract (step, train_loss, lr) from the 160M SPMD training log."""
    steps: list[int] = []
    losses: list[float] = []
    lrs: list[float] = []
    if not path.exists():
        return steps, losses, lrs
    with path.open("r", errors="ignore") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if not m:
                continue
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            lrs.append(float(m.group(3)))
    return steps, losses, lrs


# ---------- Plots ----------

def plot_loss_curves() -> None:
    s131_step, s131_val, s131_train = parse_131m_history(
        ROOT / "ckpts-pulled" / "aria_v1_131m_repro" / "summary.json"
    )
    s160_step, s160_train, _ = parse_160m_train_log(
        ROOT / "docs" / "eval_160m" / "spmd_train.log"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    if s131_step:
        ax.plot(s131_step, s131_train, "o-", label="ARIA-131M train (wikitext-103)",
                color="tab:blue", alpha=0.6)
        ax.plot(s131_step, s131_val, "s--", label="ARIA-131M val (wikitext-103)",
                color="tab:blue")
    if s160_step:
        # Subsample for legibility (30K points is too dense).
        stride = max(1, len(s160_step) // 300)
        ax.plot(s160_step[::stride], s160_train[::stride], "-",
                label="ARIA-160M train (FineWeb-Edu)",
                color="tab:orange", alpha=0.8, linewidth=1.0)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("ARIA training loss — 131M wikitext vs 160M FineWeb-Edu")
    ax.legend()
    ax.grid(alpha=0.3)
    out = FIG_DIR / "loss_curves_131m_160m.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def plot_eval_comparison() -> None:
    aria160 = load_160m_eval()
    aria131 = load_131m_eval()

    # Six benchmarks with aligned headline metric.
    tasks = [
        ("hellaswag", "acc_norm"),
        ("arc_easy", "acc_norm"),
        ("arc_challenge", "acc_norm"),
        ("piqa", "acc_norm"),
        ("winogrande", "acc"),
        ("lambada_openai", "acc"),
    ]

    labels = [f"{t}\n({m})" for t, m in tasks]

    def pull(d: dict, task: str, metric: str) -> float:
        vals = d.get(task, {})
        # Fall back to "acc" if acc_norm missing (winogrande).
        return float(vals.get(metric, vals.get("acc", 0.0)))

    aria131_vals = [pull(aria131, t, m) for t, m in tasks]
    aria160_vals = [pull(aria160, t, m) for t, m in tasks]
    pythia_vals = [PYTHIA_160M.get(t, {}).get(m, PYTHIA_160M.get(t, {}).get("acc", 0.0))
                   for t, m in tasks]

    import numpy as np
    x = np.arange(len(tasks))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, aria131_vals, width, label="ARIA-131M (wikitext, 50M tok)",
           color="tab:blue", alpha=0.85)
    ax.bar(x, aria160_vals, width, label="ARIA-160M (FWE, 500M tok)",
           color="tab:orange", alpha=0.9)
    ax.bar(x + width, pythia_vals, width, label="Pythia-160M (Pile, 300B tok)",
           color="tab:gray", alpha=0.7)

    # Random baseline at 25% for arc/hellaswag/openbookqa, 50% for winogrande/piqa.
    random_baseline = [0.25, 0.25, 0.25, 0.50, 0.50, 0.00]
    for i, rb in enumerate(random_baseline):
        if rb > 0:
            ax.hlines(rb, i - 0.4, i + 0.4, colors="black", linestyles="dotted",
                      linewidth=1)

    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (0-shot)")
    ax.set_ylim(0, 0.7)
    ax.set_title("0-shot benchmark accuracy — dotted lines = random baseline")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    out = FIG_DIR / "eval_comparison_160m_vs_pythia.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_loss_curves()
    plot_eval_comparison()
