# Reasoning RL recipes for ARIA 131M (Apr 2026 sweep)

## TOP 3 techniques

### 1. JustRL (ICLR 2026) — PRIMARY PICK
- Paper: arXiv 2512.16649 (Dec 2025) + ICLR 2026 blog
- Claim: Single-stage GRPO + fixed HPs + binary rule verifier beats multi-stage ProRL-V2/QuestA at 1.5B, 2× less compute (54.5% → 64.3% avg on 9 math)
- Evidence: 1.5B (DeepSeek-R1-Distill-Qwen-1.5B)
- LOC: ~200 on top of veRL; reward = exact-match (50 LOC)
- Risk: Entropy collapse at 131M if base too weak — mitigate with temp=1.0, group=16
- Compute: ~40 T4-hours
- Data: DAPO-Math-17k or MATH train (12.5k)

### 2. DAPO (ByteDance Mar 2025)
- Paper: arXiv 2503.14476
- 4 tricks: Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Reward Shaping
- Claim: Fixes long-CoT instability; 50 pts AIME at 32B with 50% fewer steps
- Evidence <1B: Token-level loss especially helps small models
- LOC: +150 over GRPO
- Risk: Dynamic sampling wastes ~30% rollouts → skip if data scarce
- Compute: ~50 T4-hours
- Used as reward backbone by JustRL

### 3. Liquid LFM-1B-Math Concise-Reasoner (SFT-heavy + short RLVR)
- Source: Liquid AI blog + LFM2.5-1.2B-Thinking (Jan 2026)
- Claim: CoT SFT unlocks reasoning; short-horizon RLVR + n-gram penalty cuts doom-loops 15.7→0.36%
- Evidence: 1.2B hybrid SSM-like (matches ARIA)
- LOC: ~300 (SFT + RLVR + repetition penalty)
- Risk: Requires curated CoT SFT data first
- Compute: SFT 20 T4h + RLVR 25 T4h = 45 T4-hours
- Data: OpenR1-Math-220k (SFT) + MATH/GSM8K train (RLVR, 15k)

## RECOMMENDED RECIPE for ARIA 131M

**Sequence:**
1. SFT 1-2 epochs on filtered-short subset of OpenR1-Math-220k (50k samples)
2. JustRL GRPO on DAPO-Math-17k — group=16, lr=1e-6, KL=0.001

**Skip**: PRMs (ThinkPRM arXiv 2504.16828 shows outcome verifiers match PRMs at <1.5B, 10× cheaper)

**Expected**: GSM8K +10-15 pts IF SFT base clears ~20%
- If below 20%: add TinyGSM/MetaMathQA augmentation first

**Total compute**: ~60 T4-hours (Kaggle 30h/week × 2 sessions) OR ~8 TPU-v3 hours on TRC

## Sources
- [JustRL blog](https://iclr-blogposts.github.io/2026/blog/2026/justrl/)
- [JustRL arXiv](https://arxiv.org/html/2512.16649v1)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [LFM-1B-Math](https://www.liquid.ai/research/lfm-1b-math-can-small-models-be-concise-reasoners)
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
- [ThinkPRM](https://arxiv.org/abs/2504.16828)
- [GRPO tricks](https://cameronrwolfe.substack.com/p/grpo-tricks)
- [Tulu 3 RLVR](https://allenai.org/blog/tulu-3-technical)
