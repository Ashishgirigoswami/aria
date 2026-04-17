# Post-training recipes for ARIA 131M (Apr 2026 sweep)

Source literature: SmolLM2, ORPO, Magpie papers + HF alignment-handbook.

## Recommended: SmolLM2-135M recipe (SFT → DPO)
- Paper: arXiv 2502.02737 (SmolLM2, Feb 2025) + alignment-handbook
- Two-stage, proven at 135M scale
- TPU/T4 compatible (no custom kernels)

### Stage A: SFT
- Dataset: `HuggingFaceTB/smol-smoltalk` (460k rows, Apache-2.0)
- Config: lr=1e-3, eff_batch=16, 2 epochs, seq=2048 (T4) or 8192 (TPU), cosine + 10% warmup, bf16, grad checkpointing
- Time: 8-14h on single T4
- LOC: ~50 using TRL `SFTTrainer`

### Stage B: DPO
- Dataset: `HuggingFaceH4/ultrafeedback_binarized` (~62k pairs)
- Config: lr=1e-6, eff_batch=16, 2 epochs, max_len=1024, beta=0.5, bf16
- Time: 3-5h on T4
- LOC: ~50 using TRL `DPOTrainer`

### Fallback: ORPO (single-stage)
- Paper: arXiv 2403.07691 (EMNLP 2024)
- Single loss combining NLL + log-odds ratio
- **No reference model needed** → half the memory
- Config: lr=8e-6, beta=0.1, bs=2 × accum=8, 2 epochs
- LOC: ~30 using TRL `ORPOTrainer`
- Validated at OPT-125M / OPT-350M scale

### Smoke test: Magpie-Ultra SFT-only
- Paper: arXiv 2406.08464 (ICLR 2025)
- Dataset: `argilla/magpie-ultra-v1.0` (~50k filtered pairs)
- Finishes in 1-2h on T4
- LOC: ~20

## Implementation requirements before running
1. Add ChatML special tokens to tokenizer: `<|im_start|>`, `<|im_end|>`
2. Use `tokenizer.apply_chat_template` 
3. Verify our vocab handles new tokens (resize embedding)

## Skipped
- KTO — evidence thin <1B, validated mostly 1-30B
- SimPO — validated 7-9B
- LIMA-1000 — assumes knowledge-rich base (ARIA is under-pretrained)
- Claude/GPT-4 distillation — redundant with public Magpie

## Decision matrix for ARIA 131M

| Path | When | Effort | Expected outcome |
|---|---|---|---|
| SFT + DPO | Best model, have T4/TPU for 12-18h | Medium | ChatML-able, preference-aligned |
| ORPO only | Simplest preference training | Low | Similar quality, less memory |
| Magpie SFT | Smoke test, fastest | Trivial | Validates chat template works |

## Path integration plan
After ARIA 131M pretraining completes:
1. Smoke test: Magpie SFT on Kaggle T4 (2h)
2. If pretrain loss sane: full SmolLM2 recipe (SFT + DPO, ~15h)
3. Compare to SmolLM2-135M-Instruct on same benchmarks

## Sources
- [SmolLM2 arXiv](https://arxiv.org/abs/2502.02737)
- [alignment-handbook smollm2 recipe](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2)
- [ORPO arXiv](https://arxiv.org/abs/2403.07691)
- [Magpie arXiv](https://arxiv.org/abs/2406.08464)
- [smol-smoltalk dataset](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
- [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [magpie-ultra-v1.0](https://huggingface.co/datasets/argilla/magpie-ultra-v1.0)
