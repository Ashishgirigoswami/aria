# ARIA — Layered State Attention

A hybrid attention + selective-state 154M-parameter language model, trained
from scratch on Google TPU Research Cloud free-tier. On 0-shot evaluations
against Pythia-160M (same parameter scale, 600× more training data), ARIA
shows **one clear win (ARC-Easy, +6pp), two within-error matches
(ARC-Challenge, WinoGrande), and three clear trails (HellaSwag, PIQA,
LAMBADA).**

![0-shot benchmark comparison](figures/eval_comparison_160m_vs_pythia.png)

## Headline result (0-shot, lm-eval-harness 0.4.11)

| Task                     | Metric      | ARIA-131M  | **ARIA-160M**  | Pythia-160M | Δ to Pythia | Verdict |
|--------------------------|-------------|:----------:|:--------------:|:-----------:|:-----------:|:-------:|
| Training tokens          | —           | 50M wikitext | **500M FWE** | 300B Pile  | — | — |
| **ARC-Easy**             | acc         | 33.25%     | **45.62%**     | 39.50%      | **+6.12pp** | ✅ WIN |
| **ARC-Easy**             | acc_norm    | 32.32%     | **41.20%**     | 37.70%      | **+3.50pp** | ✅ WIN |
| WinoGrande               | acc         | 48.46%     | **50.51%**     | 50.80%      | -0.29pp (±1.4) | ≈ MATCH |
| ARC-Challenge            | acc         | 18.60%     | **19.80%**     | 19.90%      | -0.10pp (±1.2) | ≈ MATCH |
| ARC-Challenge            | acc_norm    | 23.46%     | **24.40%**     | 25.30%      | -0.90pp (±1.3) | ≈ MATCH |
| OpenBookQA               | acc         | —          | 16.40%         | 16.80%      | -0.40pp (±1.7) | ≈ MATCH |
| PIQA                     | acc         | 53.37%     | **59.79%**     | 60.60%      | -0.81pp (±1.1) | ⚠ trails slightly |
| PIQA                     | acc_norm    | 49.18%     | **58.38%**     | 60.50%      | -2.12pp (±1.2) | ❌ trails |
| OpenBookQA               | acc_norm    | —          | 27.80%         | 29.40%      | -1.60pp (±2.0) | ⚠ trails slightly |
| HellaSwag                | acc         | 26.23%     | **27.63%**     | 28.50%      | -0.87pp (±0.4) | ❌ trails |
| HellaSwag                | acc_norm    | 26.62%     | **28.52%**     | 33.60%      | **-5.08pp**  | ❌ trails |
| LAMBADA                  | acc         | 5.72%      | **14.57%**     | 37.30%      | **-22.73pp** | ❌ trails (data-limited) |
| LAMBADA                  | perplexity  | 8058       | **1008** (8× better than 131M)* | **18** | 56× worse | ❌ trails (*see caveat) |

`± value` = 1 standard error of the difference (Welch). "MATCH" = within
±2 standard errors; "WIN" / "trails" = reliably different.

**What the numbers do and don't say:**
- ✅ ARC-Easy is a **genuine win** — +6pp at 600× less training data is real.
- ≈ ARC-Challenge, WinoGrande, OpenBookQA (acc) are **within statistical error**
  of Pythia. Not a "win," not a "loss."
- ❌ HellaSwag acc_norm (-5pp), PIQA acc_norm (-2pp), LAMBADA (22pp / 56× ppl)
  are **real gaps**. These are the data-hungry benchmarks (story completion,
  narrative coherence) where 500M tokens is demonstrably not enough.
- ⚠ **LAMBADA caveat**: the current `loglikelihood_rolling` in
  [`scripts/eval_harness.py`](./scripts/eval_harness.py) truncates inputs to
  `max_length=256` instead of doing proper sliding-window scoring. LAMBADA
  samples average ~70 tokens so truncation rarely bites there, but the
  perplexity number should be treated as indicative, not authoritative,
  until the wrapper is fixed. See [issue tracker](#known-issues) below.

Raw eval JSONs: [`docs/eval_160m/`](./docs/eval_160m/).
Per-task headline metric plotted above; dotted lines = random baseline.

## Provenance (what artifact → what claim)

Every number in the headline table maps to exactly one (checkpoint, config,
eval JSON, date) tuple:

| Model          | Checkpoint                              | Config                                      | Eval JSON                              | Date       |
|----------------|-----------------------------------------|---------------------------------------------|----------------------------------------|------------|
| ARIA-131M      | `aria_v1_150m_t256/best.pt` (TPU v6e-4) | `configs/aria_v1_150m_t256.yaml`            | `runs/.../eval_v1_131m.json`           | 2026-04-18 |
| ARIA-160M      | `aria_v1_160m_multihost/final.pt` (v4-32 SPMD) | `configs/aria_v1_160m_multihost.yaml` | `docs/eval_160m/eval_batch_*.json` + `eval_winogrande_openbookqa.json` | 2026-04-19 |
| Pythia-160M    | `EleutherAI/pythia-160m` (public)       | — (from HF model card)                      | EleutherAI published leaderboard       | 2023       |

Note: an earlier document ([`docs/TRAINING_RESULTS_131M.md`](./docs/TRAINING_RESULTS_131M.md))
reports **val_ppl 29.01, elapsed 82,771s** for the ORIGINAL 131M run on
TPU v6e-4 (2026-04-17). The later ARIA-131M reproducibility run on TPU v4-8
(`ckpts-pulled/aria_v1_131m_repro/summary.json`) reports **val_ppl 29.37,
elapsed 155,982s** — a separate, independent 2026-04-18 run that confirmed
the original within ~1% ppl on different hardware. Both are intentional; the
repro run is documented separately to demonstrate reproducibility across
TPU generations, not to replace the original result.

## Training curves

![Training loss — 131M vs 160M](figures/loss_curves_131m_160m.png)

ARIA-131M (wikitext-103) plateaus around train-loss 3.3 by step 8K —
wikitext is a narrow corpus and 50M tokens saturates the model. ARIA-160M
(FineWeb-Edu, 10× more and more-diverse tokens) keeps descending through
30K steps, finishing at train-loss 3.44 / val-loss 3.52 (ppl 33.9 on FWE).

## Architecture

Layered State Attention fuses three memory-efficient attention ideas into a
single fused operation:

1. **Shared MLA low-rank KV**, inspired by DeepSeek-V2 MLA
   ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)):
   `c_kv = W_down · x` with `dim(c_kv) << dim(x)`, then independent K/V
   up-projections from the same latent.
2. **Selective SSM state**, inspired by Mamba-2
   ([arXiv:2405.21060](https://arxiv.org/abs/2405.21060)):
   content-dependent recurrent summary `s_t = A(x_t)·s_{t-1} + B(x_t)·u_t`,
   where `u_t` reads from the **same** `c_kv` latent (shared bottleneck).
3. **Joint softmax** over the local causal KV window *and* a virtual token
   reconstructed from the current SSM state — one softmax, not two.

Two novelty claims (independently verified against arXiv + Google Scholar +
Semantic Scholar + patents): shared-latent-drives-both-paths and
joint-softmax-fusion. Deeper architecture notes:
[`aria/lsa.py`](./aria/lsa.py) (authoritative reference implementation).

Block ratio: 3:1 LSA:full-attention interleave (per Qwen3-Next validation).
Effective long-range memory per token: `O(W + d_state)` instead of `O(N)`.

## Training recipe

| Parameter      | ARIA-131M (wikitext)  | ARIA-160M (FWE)        |
|----------------|-----------------------|------------------------|
| Params         | 131,074,944           | 154,470,912            |
| d_model        | 768                   | 768                    |
| Layers         | 12                    | 15                     |
| Heads          | 12 (d_head=64)        | 12 (d_head=64)         |
| d_kv_latent    | 384 (2× compression)  | 384                    |
| d_state        | 192 (SSM state dim)   | 192                    |
| seq_len        | 256                   | 256                    |
| Batch (global) | 64                    | 64                     |
| LR (WSD)       | 6e-4 → 6e-5           | 6e-4 → 6e-5            |
| Steps          | 8,000                 | 30,000                 |
| Hardware       | TPU v6e-4 single-core | TPU v4-32 SPMD 16-chip |
| Wallclock      | 23 h                  | 16.3 h                 |

Training scripts:
- Single-core TPU: [`scripts/train_xla.py`](./scripts/train_xla.py)
- **Multi-host SPMD** (recommended for v4-32+): [`scripts/train_xla_spmd.py`](./scripts/train_xla_spmd.py).
  Uses `xr.use_spmd()` + `Mesh` + `mark_sharding` — **not** `xmp.spawn`
  (which crashes on torch_xla 2.9 + v4). Deploy notes:
  [`docs/MULTIHOST_DEPLOY.md`](./docs/MULTIHOST_DEPLOY.md).

## Quick start

```bash
git clone https://github.com/Ashishgirigoswami/aria.git
cd aria
pip install -r requirements.txt

# Reproduce ARIA-131M training (~23 h on v6e-4, ~9 h on a single H100):
python scripts/train_xla.py --config configs/aria_v1_150m_t256.yaml

# Reproduce ARIA-160M on a TPU v4-32 pod (8-host multi-host SPMD):
# Run on every worker in parallel:
gcloud compute tpus tpu-vm ssh <pod> --zone=<zone> --worker=all \
  --command="cd ~/aria && PJRT_DEVICE=TPU XLA_USE_SPMD=1 \
             python3 scripts/train_xla_spmd.py \
               --config configs/aria_v1_160m_multihost.yaml"

# Evaluate a checkpoint against the 6-benchmark lm-eval suite:
python scripts/eval_harness.py \
  --ckpt checkpoints/aria_v1_160m_multihost/final.pt \
  --config configs/aria_v1_160m_multihost.yaml \
  --tasks arc_easy,arc_challenge,piqa,winogrande,hellaswag,lambada_openai \
  --device xla --max-length 256 --pad-to-max --batch-size 16 \
  --output eval_160m.json
```

## What this is (and isn't)

- ✅ **A working 154M-parameter hybrid LM**, trained from scratch on 500M
  FineWeb-Edu tokens, beating Pythia-160M on **1 of 7** benchmarks (ARC-Easy,
  +6pp) and within statistical error on **3 more** (ARC-Challenge,
  WinoGrande, OpenBookQA acc), while trailing clearly on the data-hungry
  ones (HellaSwag, PIQA, LAMBADA).
- ✅ **Architectural novelty**: shared MLA latent driving both KV path and
  SSM input stream; joint softmax fusion over local KV and virtual state
  token. Both claims independently verified against prior work.
- ✅ **Reproducible**: `summary.json` + `final.pt` + per-task eval JSONs in
  [`docs/eval_160m/`](./docs/eval_160m/); exact configs under
  [`configs/`](./configs/); every training commit tagged in git history.
- ❌ **Not a frontier model**. Pythia-160M outperforms on HellaSwag and
  LAMBADA; Qwen3-0.6B and larger models dominate every benchmark.
- ❌ **Not instruction-tuned**. Numbers are raw base-model 0-shot. No SFT,
  RLHF, or DPO applied yet.
- ❌ **Not on HuggingFace Hub yet** — model card + weights upload in
  progress.

## Directory layout

```
.
├── README.md                   (this file)
├── LICENSE                     (MIT)
├── requirements.txt
│
├── aria/                       (core package)
│   ├── lsa.py                  (LSA attention + LM for CUDA)
│   ├── lsa_xla.py              (XLA-compatible scan wrapper for TPU)
│   ├── lsa_v2.py               (matrix-state Gated DeltaNet variant)
│   ├── lsa_mamba3.py           (LSA + Mamba-3 hybrid)
│   ├── baseline.py             (matched causal transformer)
│   ├── mamba3_model.py         (pure Mamba-3 baseline for Kaggle)
│   ├── data.py                 (BPE/char datasets + FineWeb-Edu streaming)
│   ├── trainer.py              (GPU AdamW trainer with resume)
│   ├── trainer_xla.py          (single-core TPU trainer)
│   └── nn_utils.py             (RMSNorm, SwiGLU, RoPE — shared)
│
├── configs/
│   ├── aria_v1_150m_t256.yaml         (131M wikitext)
│   ├── aria_v1_160m_fwe_500m.yaml     (160M single-host)
│   ├── aria_v1_160m_multihost.yaml    (160M v4-32 SPMD — main result)
│   ├── aria_v1_1b_multihost.yaml      (1B scale-up plan)
│   └── …
│
├── scripts/
│   ├── train.py                  (GPU)
│   ├── train_xla.py              (single-core TPU)
│   ├── train_xla_spmd.py         (multi-host TPU SPMD — what trained 160M)
│   ├── eval_harness.py           (lm-eval wrapper with batched pad_to_max)
│   ├── slice_keeper.py           (holds PJRT slot during partial evals)
│   ├── make_eval_graphs.py       (regenerates figures/ from JSONs)
│   └── …
│
├── figures/
│   ├── eval_comparison_160m_vs_pythia.png
│   ├── loss_curves_131m_160m.png
│   └── phase0_learning_curves.{svg,png}
│
├── docs/
│   ├── EVAL_RESULTS_131M.md            (pre-print writeup)
│   ├── TRAINING_RESULTS_131M.md        (131M recipe + tables)
│   ├── MULTIHOST_DEPLOY.md             (v4-32 + SPMD runbook)
│   ├── AI_BUILDABLES_2026.md           (post-160M research survey)
│   ├── HF_ARCHITECTURE_SURVEY_APR2026.md (what's on HF trending)
│   └── eval_160m/                      (per-task JSON results)
│
└── ckpts-pulled/                 (local copies of summary.json from TPU runs)
```

## Citation

```bibtex
@misc{goswami2026aria160m,
  author = {Ashish Giri Goswami},
  title  = {ARIA-160M: A hybrid attention+SSM language model trained from
            scratch on 500M FineWeb-Edu tokens, competitive with Pythia-160M
            at 600× less training data},
  year   = {2026},
  note   = {CargoHive Technologies. https://github.com/Ashishgirigoswami/aria},
}
```

## Known issues

- **`loglikelihood_rolling` truncates instead of sliding.** The current eval
  wrapper ([`scripts/eval_harness.py`](./scripts/eval_harness.py)) scores
  only `ids[:max_length]` of each document rather than doing proper
  sliding-window scoring across the full sequence. Affects any benchmark
  whose metric passes through `loglikelihood_rolling` — specifically
  WikiText perplexity, and LAMBADA documents longer than 256 tokens
  (LAMBADA averages ~70 tokens so the 1008 ppl number is probably not
  materially wrong, but it isn't strictly rigorous). A sliding-window
  rewrite is on the backlog; until then, treat LAMBADA ppl as indicative
  not authoritative. All accuracy-based benchmarks (loglikelihood of
  short continuations, scored on full short inputs) are unaffected.
- **No HuggingFace Hub upload yet.** Model weights + config + card are
  staged at `ckpts-pulled/aria_v1_160m_multihost/`; Hub push is next.
- **No instruction tuning.** All reported numbers are base-model 0-shot.
  SFT/DPO on Tulu-3 + UltraFeedback is the planned follow-up.

## License

MIT. See [LICENSE](./LICENSE).
