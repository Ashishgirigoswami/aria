# Synthetic pretraining data for ARIA 400M-1B (Apr 2026 sweep)

## TOP 3 synthetic corpora

### 1. Nemotron-CC-v2 / v2.1 (NVIDIA, Sep 2025)
- Qwen3-30B-rephrased CommonCrawl + specialized math/code
- Size: 6.6T base + 2.5T v2.1 delta
- HF: `nvidia/Nemotron-CC-v2`, `nvidia/Nemotron-CC-v2.1`, `Nemotron-Pretraining-SFT-v1`
- Evidence: Backbone of Nemotron-3 Nano 4B/30B-A3B. Beats DCLM on MMLU/ARC
- LOC to integrate: ~15 (datasets + parquet packer)
- Cost to extend: ~$0.15/1M tokens via Qwen3-30B-A3B
- **License: NVIDIA DAA — commercial OK**

### 2. BeyondWeb (DatologyAI, Aug 2025)
- Paper: arXiv 2508.10975
- Claim: Targeted rephrasing → highest-density synthetic published
- Evidence: 3B @180BT on BeyondWeb beats 8B @180BT on Cosmopedia; +5.1pp over Cosmopedia, +2.6pp over Nemotron-CC-HQ at 8B (14 benchmarks), 7.7× speedup vs raw web
- Size: Dataset NOT released — paper + recipe only
- LOC to integrate: ~150 (rephraser worker + prompts + dedup)
- Cost: ~$0.20/1M via Qwen3-8B rephraser (8B is saturation point)
- License: Paper; self-generated output is yours

### 3. Persona-Hub + OpenR1-Math-220k (reasoning slice)
- Persona-Hub: arXiv 2406.20094 — 1B personas, ~50B token potential
- OpenR1-Math-220k: 500M tokens of DeepSeek-R1 traces (Apache-2.0)
- HF: `proj-persona/PersonaHub`, `open-r1/OpenR1-Math-220k`
- Evidence: Drove Qwen2.5-Math and SmolLM3 SFT gains
- LOC: ~40 (instruction formatter + thinking-tag packer)
- Cost: ~$0.40/1M via R1-distill-70B
- License: PersonaHub CC-BY-NC 4.0 (non-commercial!); OpenR1-Math Apache-2.0

## Scaling-fraction reference

| Model | Synthetic % of pretrain | Source |
|---|---|---|
| SmolLM3-3B | 3-8% | Cosmopedia v2 inside web slice |
| OLMo-2 | ~0% | Mid-training stage-2 adds ~1% math |
| Qwen3-small | 10-15% | Qwen-Math + Qwen-Code style (undisclosed exact) |

**Sweet spot**: ~30% rephrased tokens (arXiv 2510.01631). Textbook-style collapses past 33%. Generator saturates at 8B.

## RECOMMENDED MIX for ARIA 400M-1B

```
75% FineWeb-Edu          (base web, threshold≥3)
15% Nemotron-CC-v2 HQ    (rephrased web, commercial-OK)
 5% OpenR1-Math + persona-math (reasoning slice)
 5% StarCoder v2 / Stack-Edu (code)
```

## Skip
- Pure Cosmopedia v2 past 5% — collapse risk at 1B
- Persona-Hub if commercial use (CC-BY-NC 4.0 license)

## Sources
- [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2)
- [Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1)
- [BeyondWeb paper](https://arxiv.org/abs/2508.10975)
- [Persona-Hub](https://arxiv.org/abs/2406.20094)
- [OpenR1-Math](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [Demystifying Synthetic Data EMNLP'25](https://arxiv.org/abs/2510.01631)
- [SmolLM3 blog](https://huggingface.co/blog/smollm3)
- [OLMo-2 paper](https://arxiv.org/pdf/2501.00656)
