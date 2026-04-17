# Long-context extension for ARIA (seq=256 → 32k-128k) — Apr 2026 sweep

## TOP 3 techniques

### 1. RoPE theta scaling + YaRN/LongRoPE2
- Paper: LongRoPE2 arXiv 2502.20082 (Feb 2025, refreshed Mar 2026)
- Claim: Rescale RoPE base frequency + NTK-by-parts → 32k-128k from short-trained model in ~400 continued-pretraining steps
- Evidence <1B: SmolLM3, Phi-4 validated
- LOC: ~50 (rescale theta + 2-stage dataloader)
- Risk: Works only if RoPE is in MLA path (yes for ARIA)

### 2. Sliding Window Attention + periodic global tokens (Samba pattern)
- Paper: Samba ICLR 2025
- Claim: Restrict MLA to 2k window, SSM carries long-range state → 1M zero-shot (256× ratio)
- Evidence: Samba-421M/1.3B extrapolates 4k→1M on Proof-Pile
- LOC: ~80 (causal window mask + 1-2 global layers)
- Risk: Low — ARIA's 3:1 hybrid already fits Samba template

### 3. Two-stage context curriculum (256 → 4k → 32k)
- Paper: ProLong ACL 2025 + NVIDIA UltraLong arXiv 2504.06214
- Claim: Short base + continued-pretraining at increasing seq → beats from-scratch at 5% token budget
- LOC: ~30 (dataset bucketing + phase scheduler)
- Risk: Needs long docs (books, arXiv, code repos)

## CONCRETE PLAN for ARIA (seq=256 → 32k)

### Phase A (cheapest, first)
- Continued-pretrain ARIA-131M at seq=4096 for ~5B tokens
- RoPE theta: 10k → 500k
- No architecture change

### Phase B (add long-range capability)
- Add SWA mask (window=1024) to 1-in-4 MLA layers
- Keep 1 global-attn layer at depth 0
- Continue at seq=32768 for ~3B tokens
- Corpus: Books3, arXiv, GitHub (long docs)

### Phase C (optional, 128k)
- YaRN rescale to 128k, 400 steps

Total: ~130 LOC, ~8B tokens.
Preserves ARIA's shared-MLA + joint-softmax fusion.
Inference: SWA keeps KV cache flat → runs on GTX 1050 at 32k.

## Sources
- [Samba ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/84a7fc24ed52e8eff514c33e8ac76ea3-Paper-Conference.pdf)
- [LongRoPE2](https://arxiv.org/pdf/2502.20082)
- [ProLong ACL 2025](https://aclanthology.org/2025.acl-long.366.pdf)
- [NVIDIA UltraLong](https://arxiv.org/html/2504.06214v1)
- [Cerebras extend context 99% fewer tokens](https://www.cerebras.ai/blog/extending-llm-context-with-99-less-training-tokens)
- [RAttention](https://arxiv.org/html/2506.15545v1)
- [YaRN overview](https://www.emergentmind.com/topics/yarn-yet-another-rope-extension-method)
- [Mamba-3 ICLR 2026](https://openreview.net/pdf?id=HwCvaJOiCj)
