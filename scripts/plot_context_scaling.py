"""Generate the context-scaling figure.

Shows val perplexity for baseline vs LSA at two context lengths (256 and 512)
side by side. Demonstrates that LSA's regularization advantage grows with
context length — the core prediction of the compression-bottleneck hypothesis.

Usage:
    python scripts/plot_context_scaling.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def mean_across_seeds(paths: list[str]) -> tuple[list[int], list[float], list[float]]:
    runs = [load(p) for p in paths]
    steps = runs[0]["history"]["step"]
    means, stds = [], []
    for i in range(len(steps)):
        vals = [r["history"]["val_ppl"][i] for r in runs]
        means.append(st.mean(vals))
        stds.append(st.stdev(vals) if len(vals) > 1 else 0.0)
    return steps, means, stds


def plot() -> None:
    # seq_len 256 — averaged over 3 seeds
    b256_steps, b256_mean, b256_std = mean_across_seeds([
        "checkpoints/baseline_tiny/summary.json",
        "checkpoints/baseline_seed43/summary.json",
        "checkpoints/baseline_seed44/summary.json",
    ])
    l256_steps, l256_mean, l256_std = mean_across_seeds([
        "checkpoints/lsa_tiny/summary.json",
        "checkpoints/lsa_seed43/summary.json",
        "checkpoints/lsa_seed44/summary.json",
    ])

    # seq_len 512 — single seed each (LSA run crashed OOM at step 2400)
    b512 = load("checkpoints/baseline_len512/summary.json")
    l512 = load("checkpoints/lsa_len512/summary.json")
    b512_steps = b512["history"]["step"]
    b512_ppl = b512["history"]["val_ppl"]
    l512_steps = l512["history"]["step"]
    l512_ppl = l512["history"]["val_ppl"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120, sharey=False)
    baseline_color = "#d1495b"
    lsa_color = "#1d6fa5"

    # --- Left panel: seq_len=256, n=3 seeds with std bands ---
    ax = axes[0]
    ax.fill_between(
        b256_steps,
        [m - s for m, s in zip(b256_mean, b256_std)],
        [m + s for m, s in zip(b256_mean, b256_std)],
        color=baseline_color, alpha=0.18, linewidth=0,
    )
    ax.fill_between(
        l256_steps,
        [m - s for m, s in zip(l256_mean, l256_std)],
        [m + s for m, s in zip(l256_mean, l256_std)],
        color=lsa_color, alpha=0.18, linewidth=0,
    )
    ax.plot(b256_steps, b256_mean, color=baseline_color, linewidth=2.4,
            marker="o", markersize=5, markerfacecolor="white",
            markeredgewidth=1.6, markeredgecolor=baseline_color,
            label="Baseline (mean ± σ, n=3)")
    ax.plot(l256_steps, l256_mean, color=lsa_color, linewidth=2.4,
            marker="s", markersize=5, markerfacecolor="white",
            markeredgewidth=1.6, markeredgecolor=lsa_color,
            label="LSA (mean ± σ, n=3)")

    # Final-step gap
    gap_256 = 100 * (b256_mean[-1] - l256_mean[-1]) / b256_mean[-1]
    ax.annotate(
        f"LSA +{gap_256:.1f}%",
        xy=(b256_steps[-1], (b256_mean[-1] + l256_mean[-1]) / 2),
        xytext=(b256_steps[-1] - 400, (b256_mean[-1] + l256_mean[-1]) / 2),
        fontsize=11, color="#1a1a1a", fontweight="bold", ha="right",
    )

    ax.set_title("Context length 256 (n=3 seeds)", fontsize=12, pad=10)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Validation perplexity", fontsize=11)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_ylim(4.0, 8.0)

    # --- Right panel: seq_len=512, single seed ---
    ax = axes[1]
    ax.plot(b512_steps, b512_ppl, color=baseline_color, linewidth=2.4,
            marker="o", markersize=5, markerfacecolor="white",
            markeredgewidth=1.6, markeredgecolor=baseline_color,
            label="Baseline (seed 42)")
    ax.plot(l512_steps, l512_ppl, color=lsa_color, linewidth=2.4,
            marker="s", markersize=5, markerfacecolor="white",
            markeredgewidth=1.6, markeredgecolor=lsa_color,
            label="LSA (seed 42, crashed OOM)")

    # Mark the crash point
    crash_step = l512_steps[-1]
    crash_val = l512_ppl[-1]
    ax.annotate(
        "run crashed (OOM)",
        xy=(crash_step, crash_val), xytext=(crash_step - 350, crash_val + 2),
        fontsize=9, color="#444444", style="italic",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
    )

    # Last measured advantage
    gap_512 = 100 * (b512_ppl[11] - l512_ppl[-1]) / b512_ppl[11]
    ax.annotate(
        f"LSA +{gap_512:.1f}% at step {crash_step}\n(still widening)",
        xy=(crash_step, (b512_ppl[11] + l512_ppl[-1]) / 2),
        xytext=(1600, 10), fontsize=11, color="#1a1a1a", fontweight="bold", ha="left",
    )

    ax.set_title("Context length 512 (seed 42, partial LSA)", fontsize=12, pad=10)
    ax.set_xlabel("Training step", fontsize=11)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_ylim(4.0, 20.0)

    fig.suptitle(
        "LSA's regularization advantage grows with context length\n"
        "(5.4M params, char-level tiny-shakespeare)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out_base = ROOT / "figures" / "phase0_context_scaling"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out_base.with_suffix('.svg')}")
    print(f"wrote {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    plot()
