# muP + HP transfer for ARIA 131M→1B scale-up (Apr 2026 sweep)

## TOP recommendation: u-muP (Unit-Scaled muP) — adopt before 1B

### 1. u-muP (RECOMMENDED)
- Paper: arXiv 2407.17465 (ICLR 2025 Spotlight, Graphcore/Blake et al.)
- Claim: muP + unit scaling → activations/weights/grads at scale 1, free FP8 training
- Evidence: Matches/beats muP at widths 256-2048, ~4× fewer sweep points needed
- LOC: ~200-400 on top of microsoft/mup package
- Risk: Must manually derive abc-scaling for SSM blocks (paper only covers transformers)
- ARIA hybrid: Partial — attention direct, Mamba-2 SSM needs manual derivation

### 2. Classical muP (Yang et al. 2022) — fallback
- arXiv 2203.03466, microsoft/mup package
- Claim: Sweep LR on 40M proxy → zero-shot transfer to 6.7B+
- Evidence: Cerebras-GPT 111M→13B, ~2× compute savings
- LOC: ~150 using `set_base_shapes`, `MuReadout`, `MuAdamW`
- ARIA hybrid: Community confirms works on Mamba, transfer LR + init std only

### 3. Distributed Shampoo (orthogonal) — skip for now
- arXiv 2309.06497
- Won MLCommons AlgoPerf external-tuning
- Untested on SSMs — defer
- Sophia/Lion gains evaporated in controlled comparisons (arXiv 2407.07972) — skip those

## Decision: adopt u-muP NOW before 1B

Reasons:
1. **Cost avoidance**: 1B LR sweep costs $5-20K of compute
2. **Compatibility**: post-hoc parametrization swap invalidates checkpoints anyway
3. **Future-proof**: u-muP's FP8 readiness for H100/B200

## Concrete plan
1. Port ARIA 131M to u-muP parametrization
2. Coord-check plots flat across widths 256/512/1024 (verification)
3. Sweep LR on 40M proxy (cheap)
4. Transfer to 1B
5. Keep AdamW optimizer (skip exotic)

## Implementation checklist
- [ ] Install `pip install mup` + optional u-muP utilities
- [ ] Refactor `aria/lsa.py` linear layers to `MuReadout` for output
- [ ] Define base model config (40M proxy) + target (1B)
- [ ] Set `mup.set_base_shapes(model, base_shapes, delta_shapes)`
- [ ] Replace `torch.optim.AdamW` with `mup.optim.MuAdamW`
- [ ] For SSM blocks: keep `dt_init` and `A_init` NON-mup scaled (width-independent)
- [ ] Coord-check: run 100 steps at each width, plot activation norms

## Sources
- [u-muP paper](https://arxiv.org/abs/2407.17465)
- [Tensor Programs V / muP](https://arxiv.org/abs/2203.03466)
- [microsoft/mup](https://github.com/microsoft/mup)
- [Cerebras muP guide](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)
- [EleutherAI muTransfer](https://blog.eleuther.ai/mutransfer/)
- [Cerebras-GPT muP used](https://arxiv.org/abs/2304.03208)
- [Deconstructing optimizers](https://arxiv.org/html/2407.07972v1)
