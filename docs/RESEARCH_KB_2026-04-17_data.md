# FineWeb-Edu + data mixing recipes for ARIA 150M-1B (Apr 2026 sweep)

## Key findings

### 1. SmolLM2-style multi-source edu mix (RECOMMENDED)
- 60% FineWeb-Edu + 40% DCLM web + code/math/Cosmopedia annealing
- Beats single-source FineWeb-Edu by 4-7pts MMLU at 1.7B
- Source: arXiv 2502.02737 (SmolLM2, Feb 2025)

### 2. Overtrain 40-60× parameters (not Chinchilla 20×)
- For inference-served <1B models: overtraining 20-40× Chinchilla
- TinyLlama 1.1B @ 3T tokens (2700:1), SmolLM2-1.7B @ 11T (6470:1), Qwen3-0.6B @ 60000:1
- Source: Beyond Chinchilla-Optimal arXiv 2401.00448

### 3. edu_score=3 threshold + MinHash dedup
- edu_score>=3 at 1.82B/350BT: MMLU 33→37, ARC 46→57
- edu_score>=4 hurts common-sense benchmarks
- Use FineWeb-Edu pre-filtered (1.3T tokens, already MinHash-deduped)

## Token targets for ARIA

| Model | Chinchilla 20x | SmolLM2-aligned |
|---|---|---|
| 150M | 3B | 40-60B |
| 400M | 8B | 2-4T |
| 1B | 20B | 2-3T (staged) |

## Concrete recipe for ARIA 1B (2.5T tokens, 4-stage WSD)

### Stage 1 (0-1.5T, 60%) — broad learning
- 54% FineWeb-Edu (threshold=3, sample-350BT looped)
- 36% DCLM-baseline
- 10% StarCoder v2 (Stack-Edu threshold=3)

### Stage 2 (1.5-2.0T, 20%) — continue + math
- 45% FineWeb-Edu + 30% DCLM + 20% Stack-Edu + 5% OpenWebMath

### Stage 3 anneal (2.0-2.25T, 10%)
- 35% FineWeb-Edu + 25% DCLM + 20% Stack-Edu + 15% FineMath-4+ + 5% Cosmopedia v2

### Stage 4 decay (2.25-2.5T, 10%) — high-quality only
- 30% FineWeb-Edu + 20% DCLM + 24% Stack-Edu + 14% math (FineMath4+, OpenWebMath, InfiWebMath3+, AugGSM8K) + 8% Cosmopedia v2 + 4% Wikipedia

## Implementation requirements
- Weighted dataloader: ~150 LOC
- MinHash dedup: already in FineWeb-Edu
- 13-gram decontamination vs GSM8K/MATH/MMLU/ARC/HellaSwag (~80 LOC, apply to math + Cosmopedia only)
- Storage: ~5-6 TB for 2.5T tokens tokenized (use streaming)

## Data sources
- `HuggingFaceFW/fineweb-edu` (1.3T, threshold≥3)
- `mlfoundations/dclm-baseline-1.0` (3.8T)
- `bigcode/the-stack-v2` + `HuggingFaceTB/stack-edu` (threshold=3)
- `open-web-math/open-web-math`, `math-ai/FineMath`, `infimm/InfiWebMath`
- `HuggingFaceTB/cosmopedia-v2`
- `wikimedia/wikipedia`

## Sources
- [SmolLM2 paper](https://arxiv.org/html/2502.02737v1)
- [FineWeb paper](https://arxiv.org/html/2406.17557v1)
- [FineWeb-Edu card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)
- [TinyLlama](https://arxiv.org/html/2401.02385v2)
- [OLMo-2](https://arxiv.org/pdf/2501.00656)
- [LFM2 Tech Report](https://arxiv.org/html/2511.23404v1)
