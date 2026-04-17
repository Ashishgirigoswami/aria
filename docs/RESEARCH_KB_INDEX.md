# ARIA Research KB — 2026-04-17 Sweep Index

9 research cycles completed during 131M v1 training session. Each cycle: web search + 3 actionable findings + concrete recommendation.

## Index

| # | Cycle | File | Key finding |
|---|---|---|---|
| - | Session retro | [SESSION_RETRO.md](../SESSION_RETRO.md) | 10× deviation from plan, v2→v1 pivot validated |
| 1 | Hybrid SSM arch | [RESEARCH_KB_2026-04-17.md](RESEARCH_KB_2026-04-17.md) | WY chunkwise GDN (3-5× TPU speedup), Mamba-3 SISO drop-in |
| 2 | Post-training | [RESEARCH_KB_2026-04-17_posttraining.md](RESEARCH_KB_2026-04-17_posttraining.md) | SmolLM2 recipe (SFT smol-smoltalk + DPO ultrafeedback) |
| 3 | FineWeb-Edu data | [RESEARCH_KB_2026-04-17_data.md](RESEARCH_KB_2026-04-17_data.md) | 60% FineWeb-Edu + 40% DCLM + 4-stage anneal, 2.5T for 1B |
| 4 | muP HP transfer | [RESEARCH_KB_2026-04-17_mup.md](RESEARCH_KB_2026-04-17_mup.md) | Adopt u-muP before 1B → skip $5-20K LR sweep |
| 5 | Long context | [RESEARCH_KB_2026-04-17_longctx.md](RESEARCH_KB_2026-04-17_longctx.md) | Samba-pattern (SWA + SSM) → 256 to 32k in ~130 LOC |
| 6 | Distillation | [RESEARCH_KB_2026-04-17_distillation.md](RESEARCH_KB_2026-04-17_distillation.md) | On-policy from Llama-3.1-8B on Kaggle T4 (FREE) |
| 7 | Reasoning RL | [RESEARCH_KB_2026-04-17_reasoning.md](RESEARCH_KB_2026-04-17_reasoning.md) | JustRL + DAPO-Math-17k, ~60 T4h for +10-15 GSM8K pts |
| 8 | FP4/FP8 QAT | [RESEARCH_KB_2026-04-17_fp8.md](RESEARCH_KB_2026-04-17_fp8.md) | Skip on TRC — no FP8/FP4 MXU. Revisit with B200. |
| 9 | Synthetic data | [RESEARCH_KB_2026-04-17_synthetic.md](RESEARCH_KB_2026-04-17_synthetic.md) | Nemotron-CC-v2 15% + OpenR1-Math 5% on top of FineWeb-Edu |

## Recommended post-baseline sequence (2026 roadmap)

### Phase 1: After 131M v1 finishes (~16 hrs from now)
1. **Eval baseline** — lm-eval-harness on HellaSwag/ARC/PIQA/WinoGrande/LAMBADA (scripts/eval_harness.py)
2. **Publish on HuggingFace** — credibility for grants
3. **Apply to grants** — Lambda, NVIDIA Inception, AWS Activate, HF Community, AMD, Google Startup

### Phase 2: 400M-1B scale-up (if grants land)
1. **Port to u-muP** (cycle 4) — avoid $5-20K LR sweep
2. **Data mix** (cycle 3 + 9): 75% FineWeb-Edu + 15% Nemotron-CC-v2 + 5% OpenR1-Math + 5% code
3. **Architecture** (cycle 1): WY chunkwise GDN for v2 OR Mamba-3 SISO swap
4. **Train 1B** on GPU grant (Lambda H100) — Mamba-3 works on CUDA

### Phase 3: Post-training stack (cycle 2 + 6 + 7)
1. SFT on SmolTalk/Magpie-Ultra
2. On-policy distillation from Llama-3.1-8B
3. JustRL GRPO on DAPO-Math-17k for reasoning

### Phase 4: Long context (cycle 5)
1. RoPE theta 10k→500k + continued pretrain at 4k
2. Samba-pattern SWA at seq=32k
3. Optional YaRN to 128k

## Skipped directions (saturated or not applicable)
- FP4/FP8 pretraining on TPU (no hardware support)
- Net-new pure-SSM architectures (crowded)
- Diffusion LLMs at small scale
- General agentic training at 1B (frontier-bound)
- o3-style RL from scratch (rollout compute infeasible)

## Timeline (realistic)
- **Now**: 131M baseline training (16 hrs remaining)
- **Week 1**: eval + HF publish + grant applications
- **Week 2-4**: grants arrive, start 1B pretraining on H100
- **Month 2**: 1B pretrained, move to post-training
- **Month 3**: post-trained model, reasoning RL, long context
- **Month 4**: workshop paper submission (NeurIPS 2026 workshops)
