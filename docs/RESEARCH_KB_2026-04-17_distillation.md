# Distillation for ARIA 131M (Apr 2026 sweep)

## TOP 3 techniques

### 1. On-Policy Distillation (Gemma-2 recipe) — HIGHEST ROI
- Paper: Thinking Machines Lab 2025 + Gemma-2 arXiv 2408.00118
- Claim: Student generates rollouts, reverse-KL against teacher logits on student-visited states (avoids exposure bias)
- Evidence <1B: Gemma-2-2B trained this way, ">50× Chinchilla" compute effect
- LOC: ~150 (student.generate → teacher.forward → KL on logits)
- Risk: Needs open teacher (no API logit exposure)
- Teacher: Llama-3.1-8B/70B local. 50-200k prompts enough at 131M
- Adopted by: Qwen3, MiMo, GLM-5

### 2. Magpie Synthetic SFT (teacher-generated data)
- Paper: Magpie ICLR 2025, arXiv 2406.08464
- Claim: Prompt aligned LLM with just chat-template prefix → auto-generates diverse (query, answer) pairs
- Evidence: 300k Magpie samples matched Llama-3-8B-Instruct's 10M-sample alignment at SmolLM2 scale
- LOC: ~80 (or just use pre-built HuggingFaceTB/smoltalk)
- Risk: Caps at teacher text quality, no logit signal
- Teacher: Llama-3.1-8B-Instruct in 8-bit on Kaggle T4. ZERO COST if reusing smoltalk

### 3. Phi-4 Curated Synthetic + DPO (API-driven)
- Paper: Phi-4 Tech Report, arXiv 2412.08905
- Claim: Seed-curated multi-agent CoT + pivotal-token-search DPO beats raw distillation
- Evidence: Phi-4 surpasses teachers on STEM (14B, but principles transfer to <1B)
- LOC: ~400 (seed curator + multi-turn gen + judge + DPO pairs)
- Risk: Over-fits to teacher idiom; needs verifier for accuracy pillar
- Teacher: Claude 3.5 Sonnet API. ~$150-400 for 50M synthesis tokens

## RECOMMENDED TIERED STACK for ARIA 131M

**Cheapest path to +5-10 benchmark pts:**

1. **Tier 1 (FREE)**: SFT on Magpie-Ultra + SmolTalk — 0 cost, proven at 135M
2. **Tier 2 (free, more effort)**: On-policy distillation from Llama-3.1-8B-Instruct on Kaggle T4
   - This is where SmolLM2-class models get decisive jump
   - Highest ROI per dollar for 131M hybrid
3. **Tier 3 (~$200)**: Targeted 20k-sample Phi-4-style synthesis via Claude API for reasoning/code slice only
   - Not main pipeline — complement weakest areas

## Skip
- Black-box distillation (arXiv 2511.10643) — GAN-style, overkill at 131M

## Sources
- [On-Policy Distillation blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Gemma-2 Report](https://arxiv.org/html/2408.00118v1)
- [Magpie](https://arxiv.org/abs/2406.08464)
- [SmolLM2](https://arxiv.org/html/2502.02737v1)
- [Phi-4 Tech Report](https://arxiv.org/abs/2412.08905)
- [MiniLLM on-policy](https://arxiv.org/abs/2306.08543)
