# ARIA Training Roadmap — Complete Pipeline

> Last updated: 2026-04-26
> Status: Phase 2 (160M) complete, Phase 3 (1B) in progress

---

## Executive Summary

ARIA is a hybrid attention + selective-state language model family targeting
competitive performance at 100-600× less training data than frontier models.
This document outlines the complete training pipeline from pretraining through
post-training (SFT, RLHF, DPO).

**Current state:**
- ✅ Phase 0: 131M wikitext (complete)
- ✅ Phase 1: 160M FineWeb-Edu (complete)
- 🔄 Phase 2: 1B multi-source corpus (in progress)
- ⏳ Phase 3: SFT + RLHF (planned)

---

## Phase 0: Foundation (Complete)

### What We Have

| Component | Status | Location |
|-----------|--------|----------|
| 131M model | ✅ Complete | `checkpoints/aria_v1_150m_t256/best.pt` |
| Training script | ✅ Complete | `scripts/train_xla.py` |
| Evaluation script | ✅ Complete | `scripts/eval_harness.py` |
| WikiText-103 data | ✅ Complete | `data/wikitext-103/` |

### Results

| Metric | Value |
|--------|-------|
| Parameters | 131,074,944 |
| Training tokens | 50M (WikiText-103) |
| Training time | 23h (TPU v6e-4) |
| Val perplexity | 29.01 |
| ARC-Easy acc | 33.25% |

---

## Phase 1: 160M FineWeb-Edu (Complete)

### What We Have

| Component | Status | Location |
|-----------|--------|----------|
| 160M model | ✅ Complete | `checkpoints/aria_v1_160m_multihost/final.pt` |
| Multi-host SPMD trainer | ✅ Complete | `scripts/train_xla_spmd.py` |
| FineWeb-Edu data | ✅ Complete | `data/fineweb-edu/` |
| Evaluation results | ✅ Complete | `docs/eval_160m/` |

### Results

| Metric | Value |
|--------|-------|
| Parameters | 154,470,912 |
| Training tokens | 500M (FineWeb-Edu) |
| Training time | 16.3h (TPU v4-32 SPMD) |
| Val perplexity | 33.9 |
| ARC-Easy acc | 45.62% (+6pp vs Pythia) |
| WinoGrande acc | 50.51% (≈ Pythia) |
| ARC-Challenge acc | 19.80% (≈ Pythia) |

### Tokenizer Improvements (Phase 2a)

| Task | Full Tokenizer | Slim Tokenizer | Δ |
|------|---------------|----------------|---|
| Lambada perplexity | 243,613 | **69.95** | **-99.97%** |
| Lambada acc | 3.30% | **29.17%** | **+25.87%** |
| SciQ acc | 42.50% | **71.80%** | **+29.30%** |
| ARC Easy acc | 26.35% | **42.76%** | **+16.41%** |

---

## Phase 2: 1B Multi-Source Corpus (In Progress)

### What We Have

| Component | Status | Location |
|-----------|--------|----------|
| 1B model architecture | ✅ Complete | `configs/aria_v3_1b_model.yaml` |
| TPU SPMD config | ✅ Complete | `configs/aria_v3_1b_tpu_spmd.yaml` |
| Corpus pipeline | ✅ Complete | `aria/corpus.py`, `scripts/build_corpus.py` |
| TPU runbook | ✅ Complete | `docs/TPU_V6E_1B_SPMD.md` |
| Tests | ✅ Complete | `tests/test_1b_recipe.py` |

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | 1,017,700,000 |
| d_model | 2048 |
| Layers | 20 (14 LSA-SSM + 6 full-attention) |
| Heads | 16Q / 4KV GQA |
| d_head | 128 |
| d_kv_latent | 512 |
| d_state | 256 |
| seq_len | 2048 |
| Full-attn layers | [3, 6, 9, 12, 15, 18] |
| RoPE base | 500,000 |
| Norm eps | 1e-6 |

### Data Mix (SmolLM2-aligned)

| Source | Weight | Tokens | Rationale |
|--------|--------|--------|-----------|
| FineWeb-Edu (threshold=3) | 55% | 55B | Educational quality |
| DCLM-baseline | 25% | 25B | Modern web corpus |
| OpenWebMath | 10% | 10B | Math reasoning |
| Stack-Edu | 7% | 7B | Code + reasoning |
| Cosmopedia v2 | 3% | 3B | Synthetic knowledge |

**Total:** 100B tokens (100× overtraining)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 3e-4 → 3e-5 (warmup-stable-cosine) |
| Warmup steps | 8,000 |
| Stable steps | 48,000 |
| Batch size | 2 (per-chip) |
| Grad accum | 4 |
| Effective batch | 1,048,576 tokens/step |
| Total steps | ~95,500 |
| Checkpoint interval | 200 steps / 10 min wall-clock |
| Eval interval | Every 2.5B tokens |

### What We Need to Create

| Component | Status | Priority |
|-----------|--------|----------|
| Build corpus | ⏳ Pending | **High** |
| Run training | ⏳ Pending | **High** |
| Evaluation pipeline | ⏳ Pending | Medium |
| Training monitoring | ⏳ Pending | Medium |

### Commands

```bash
# 1. Build corpus (run once)
cd ~/aria
python scripts/build_corpus.py --config configs/aria_v3_1b_tpu_spmd.yaml

# 2. Sync to all TPU workers
QUEUE=aria-v6e-64-us
ZONE=us-east1-d
gcloud compute tpus tpu-vm scp --recurse ~/aria "${QUEUE}:~/aria" \
  --zone="$ZONE" --worker=all

# 3. Install dependencies
gcloud compute tpus tpu-vm ssh "$QUEUE" --zone="$ZONE" --worker=all \
  --command "cd ~/aria && pip install -q -r requirements.txt"

# 4. Launch training
gcloud compute tpus tpu-vm ssh "$QUEUE" --zone="$ZONE" --worker=all \
  --command "cd ~/aria && PJRT_DEVICE=TPU XLA_USE_SPMD=1 \
             python3 scripts/train_xla_spmd.py \
             --config configs/aria_v3_1b_tpu_spmd.yaml \
             2>&1 | tee -a logs/aria_v3_1b_w\$TPU_WORKER_ID.log"
```

### Expected Results

| Metric | Target |
|--------|--------|
| Training time | ~7-10 days (v6e-64) |
| Val perplexity | < 20 |
| ARC-Easy acc | > 50% |
| MMLU (5-shot) | > 35% |

---

## Phase 3: Supervised Fine-Tuning (Planned)

### What We Have

| Component | Status | Location |
|-----------|--------|----------|
| Base model | ⏳ Pending | `checkpoints/aria_v3_1b_tpu_spmd/final.pt` |
| SFT data | ❌ Missing | — |
| SFT trainer | ❌ Missing | — |

### What We Need to Create

| Component | Priority | Est. Effort |
|-----------|----------|-------------|
| SFT training script | **High** | 2-3 days |
| Data loading pipeline | **High** | 1-2 days |
| Evaluation script | Medium | 1 day |
| Chat template | Medium | 0.5 day |

### SFT Data Sources

| Dataset | Tokens | Quality | Notes |
|---------|--------|---------|-------|
| Tulu-3-SFT | ~10B | High | Instruction-following |
| UltraFeedback | ~5B | High | Preference pairs |
| OpenHermes-2.5 | ~5B | High | Multi-turn conversations |
| ShareGPT | ~3B | Medium | Real conversations |
| CodeAlpaca | ~1B | Medium | Code instructions |

**Total target:** 20-30B tokens for SFT

### SFT Configuration

| Parameter | Value |
|-----------|-------|
| Base model | ARIA-1B (Phase 2) |
| Learning rate | 5e-6 (lower than pretraining) |
| Batch size | 64 |
| Seq length | 2048 |
| Warmup steps | 500 |
| Total steps | 50,000 |
| Weight decay | 0.01 |
| Gradient checkpointing | True |

### SFT Script Structure

```python
# scripts/train_sft.py
"""
Supervised Fine-Tuning for ARIA.

Usage:
    python scripts/train_sft.py \
        --base_ckpt checkpoints/aria_v3_1b_tpu_spmd/final.pt \
        --sft_data HuggingFaceH4/ultrafeedback_binarized \
        --output checkpoints/aria_1b_sft \
        --device tpu
"""
```

### Expected SFT Results

| Metric | Target |
|--------|--------|
| MT-Bench | > 7.0 |
| AlpacaEval | > 80% |
| HumanEval | > 30% |
| GSM8K | > 40% |

---

## Phase 4: Reward Model Training (Planned)

### What We Have

| Component | Status | Location |
|-----------|--------|----------|
| SFT model | ⏳ Pending | `checkpoints/aria_1b_sft/final.pt` |
| RM data | ❌ Missing | — |
| RM trainer | ❌ Missing | — |

### What We Need to Create

| Component | Priority | Est. Effort |
|-----------|----------|-------------|
| RM training script | **High** | 2-3 days |
| Preference data loader | **High** | 1-2 days |
| RM evaluation | Medium | 1 day |

### RM Data Sources

| Dataset | Pairs | Quality | Notes |
|---------|-------|---------|-------|
| UltraFeedback | ~1M | High | Preference pairs |
| HH-RLHF | ~500K | High | Human preferences |
| OpenAssistant | ~300K | High | Multi-turn |
| WebGPT | ~200K | High | Web search |

**Total target:** 2M preference pairs

### RM Configuration

| Parameter | Value |
|-----------|-------|
| Base model | ARIA-1B-SFT |
| Learning rate | 1e-5 |
| Batch size | 32 |
| Seq length | 512 (shorter for RM) |
| Warmup steps | 100 |
| Total steps | 10,000 |
| Loss | Binary cross-entropy |

### RM Script Structure

```python
# scripts/train_rm.py
"""
Reward Model Training for ARIA.

Usage:
    python scripts/train_rm.py \
        --base_ckpt checkpoints/aria_1b_sft/final.pt \
        --preference_data HuggingFaceH4/ultrafeedback_binarized \
        --output checkpoints/aria_1b_rm \
        --device tpu
"""
```

---

## Phase 5: RLHF / DPO Training (Planned)

### What We Have

| Component | Status | Location |
|-----------|--------|----------|
| SFT model | ⏳ Pending | `checkpoints/aria_1b_sft/final.pt` |
| Reward model | ⏳ Pending | `checkpoints/aria_1b_rm/final.pt` |
| DPO trainer | ❌ Missing | — |

### What We Need to Create

| Component | Priority | Est. Effort |
|-----------|----------|-------------|
| DPO training script | **High** | 3-4 days |
| PPO training script | Medium | 4-5 days |
| KL divergence monitoring | Medium | 1 day |
| Evaluation pipeline | Medium | 1 day |

### DPO vs PPO

| Aspect | DPO | PPO |
|--------|-----|-----|
| Complexity | Low | High |
| Memory | 2× model | 4× model |
| Training speed | Fast | Slow |
| Stability | High | Medium |
| **Recommendation** | ✅ **Start with DPO** | Later optimization |

### DPO Configuration

| Parameter | Value |
|-----------|-------|
| Base model | ARIA-1B-SFT |
| Reward model | ARIA-1B-RM |
| Learning rate | 1e-6 |
| Batch size | 32 |
| Seq length | 512 |
| Beta (KL penalty) | 0.1 |
| Warmup steps | 100 |
| Total steps | 20,000 |

### DPO Script Structure

```python
# scripts/train_dpo.py
"""
Direct Preference Optimization for ARIA.

Usage:
    python scripts/train_dpo.py \
        --base_ckpt checkpoints/aria_1b_sft/final.pt \
        --rm_ckpt checkpoints/aria_1b_rm/final.pt \
        --preference_data HuggingFaceH4/ultrafeedback_binarized \
        --output checkpoints/aria_1b_dpo \
        --device tpu
"""
```

### Expected DPO Results

| Metric | Target |
|--------|--------|
| MT-Bench | > 8.0 |
| AlpacaEval | > 85% |
| GSM8K | > 50% |
| HumanEval | > 40% |

---

## Phase 6: Evaluation & Benchmarking (Planned)

### What We Have

| Component | Status | Location |
|-----------|--------|----------|
| lm-eval harness | ✅ Complete | `scripts/eval_harness.py` |
| 0-shot benchmarks | ✅ Complete | `docs/eval_160m/` |

### What We Need to Create

| Component | Priority | Est. Effort |
|-----------|----------|-------------|
| Few-shot evaluation | Medium | 1 day |
| Chat evaluation | **High** | 2-3 days |
| Code evaluation | Medium | 1 day |
| Math evaluation | Medium | 1 day |
| Safety evaluation | Medium | 1 day |

### Evaluation Suite

| Category | Benchmarks |
|----------|------------|
| **Language** | MMLU, HellaSwag, PIQA, ARC |
| **Chat** | MT-Bench, AlpacaEval, Chatbot Arena |
| **Code** | HumanEval, MBPP, CodeContests |
| **Math** | GSM8K, MATH, OpenWebMath |
| **Reasoning** | BigBench-Hard, TruthfulQA |
| **Safety** | ToxiGen, AdvGLUE, SafetyBench |

---

## Phase 7: Deployment & Serving (Planned)

### What We Have

| Component | Status | Location |
|-----------|--------|----------|
| Model weights | ⏳ Pending | GCS bucket |
| Config files | ✅ Complete | `configs/` |

### What We Need to Create

| Component | Priority | Est. Effort |
|-----------|----------|-------------|
| HuggingFace Hub upload | **High** | 1 day |
| vLLM serving | Medium | 2-3 days |
| API server | Medium | 2-3 days |
| Docker container | Medium | 1 day |
| Documentation | Medium | 1 day |

### Serving Options

| Option | Pros | Cons |
|--------|------|------|
| vLLM | Fast inference | GPU only |
| TGI | Easy deployment | GPU only |
| Custom XLA | TPU native | More work |
| **Recommendation** | vLLM for GPU, custom XLA for TPU |

---

## Resource Requirements

### Compute

| Phase | Hardware | Duration | Cost |
|-------|----------|---------|------|
| Phase 2 (1B pretraining) | v6e-64 (spot) | 7-10 days | Free (TRC) |
| Phase 3 (SFT) | v6e-32 (spot) | 2-3 days | Free (TRC) |
| Phase 4 (RM) | v6e-16 (spot) | 1-2 days | Free (TRC) |
| Phase 5 (DPO) | v6e-32 (spot) | 3-4 days | Free (TRC) |

### Storage

| Phase | Storage | Size |
|-------|---------|------|
| Phase 2 corpus | Local + GCS | ~5TB |
| Phase 2 checkpoints | GCS | ~50GB |
| Phase 3+ checkpoints | GCS | ~100GB |

---

## Timeline

| Phase | Start | End | Status |
|-------|-------|-----|--------|
| Phase 0 | 2026-04-15 | 2026-04-18 | ✅ Complete |
| Phase 1 | 2026-04-18 | 2026-04-19 | ✅ Complete |
| Phase 2a (tokenizer) | 2026-04-20 | 2026-04-25 | ✅ Complete |
| Phase 2b (1B prep) | 2026-04-25 | 2026-04-26 | ✅ Complete |
| Phase 2c (1B train) | 2026-04-26 | 2026-05-05 | 🔄 In Progress |
| Phase 3 (SFT) | 2026-05-06 | 2026-05-12 | ⏳ Planned |
| Phase 4 (RM) | 2026-05-13 | 2026-05-16 | ⏳ Planned |
| Phase 5 (DPO) | 2026-05-17 | 2026-05-23 | ⏳ Planned |
| Phase 6 (Eval) | 2026-05-24 | 2026-05-28 | ⏳ Planned |
| Phase 7 (Deploy) | 2026-05-29 | 2026-06-02 | ⏳ Planned |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TPU preemption | High | GCS checkpoint sync, auto-resume |
| Memory overflow (1B) | Medium | Gradient checkpointing, smaller batch |
| Data quality issues | Medium | Manual inspection, deduplication |
| Training instability | Medium | LR warmup, gradient clipping |
| Evaluation failures | Low | Multiple eval runs, averaging |

---

## Next Steps

### Immediate (This Week)

1. **Build 1B corpus** — Run `scripts/build_corpus.py`
2. **Launch 1B training** — Start on v6e-64 queue
3. **Monitor training** — Set up logging + alerts

### Short-term (Next 2 Weeks)

1. **Complete 1B pretraining** — Target 100B tokens
2. **Evaluate 1B model** — Full benchmark suite
3. **Prepare SFT data** — Download and preprocess

### Medium-term (Next Month)

1. **Implement SFT** — Training script + pipeline
2. **Train reward model** — Preference data
3. **Implement DPO** — Direct preference optimization

---

## References

- [SmolLM2 paper](https://arxiv.org/html/2502.02737v1) — Data mixing
- [FineWeb paper](https://arxiv.org/html/2406.17557v1) — Data quality
- [DPO paper](https://arxiv.org/abs/2305.18290) — Direct preference optimization
- [TPU SPMD guide](https://cloud.google.com/tpu/docs/spmd) — Multi-host training
- [TRC documentation](https://sites.research.google/trc/) — TPU Research Cloud
