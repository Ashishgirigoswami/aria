# Top 3 Actionable ARIA Improvements (Apr 2026 research sweep)

Discovered during v1 131M training session on 2026-04-17. Apply AFTER baseline completes.

## 1. WY-representation chunkwise GDN — TOP PRIORITY
- **Paper**: Gated Delta Networks ICLR 2025, arXiv 2412.06464
- **Claim**: WY/UT factorization → batched GEMMs, eliminates per-token Python loop
- **Evidence**: 45 Kt/s throughput at 1.3B on H100. Adopted by OLMo-Hybrid, Qwen3.5, Qwen3-Next.
- **Follow-up**: TFLA (arXiv 2503.14376) — 2-4× speedup over softmax kernels
- **Effort**: ~80-120 LOC rewrite of `chunked_gated_delta_rule_torch` (aria/lsa_v2.py:182) + XLA wrapper (aria/lsa_xla.py:256)
- **Risk**: MEDIUM — math tricky but explicit in paper §3.2
- **Expected speedup**: 3-5× on TPU for v2 architecture
- **Blocks**: seq_len 1024/2048 experiments

## 2. Mamba-3 SISO recurrence — LOW RISK ADD
- **Paper**: Mamba-3 ICLR 2026, arXiv 2603.15569
- **Claim**: Trapezoidal discretization + complex state → better state tracking same params
- **Evidence**: -0.17 PPL at 180M, -0.13 PPL at 440M vs Mamba-2 on FineWeb-Edu
- **Effort**: 40 LOC, scaffold already exists at aria/mamba3.py and aria/mamba3_model.py
- **Risk**: LOW for SISO. SKIP MIMO (needs ≥1.5B to shine)
- **Expected**: workshop-paper-grade novelty claim restored

## 3. 3:1 Hybrid ratio — ALREADY HAVE
- **Source**: Qwen3-Next tech report, Qwen3.5 (Feb 2026)
- **Claim**: 3 SSM : 1 Attention beats 1:1 by ~0.3 PPL at matched tokens
- **Status**: ARIA already uses `interleave_ratio: 3` — verify shared-MLA trick still amortizes correctly with only 25% attention layers
- **Effort**: Ablation study post-baseline, no code change

## Skipped (not applicable)
- Log-Linear Attention (unknown TPU cost, no small-scale numbers)
- HALO distillation (needs pretrained teacher, we don't have one yet)
- Minitron-SSM pruning (post-training only)

## Concrete post-baseline TODO
1. Finish current 131M v1 run (baseline)
2. Implement WY chunkwise GDN — enables v2 at reasonable TPU speed
3. Swap in Mamba-3 SISO as a new variant
4. Compare three runs: v1 (done), v2-WY, v3-Mamba3 on identical data + tokens
5. Pick winner, scale to 1B on FineWeb-Edu

## Sources
- [Mamba-3](https://arxiv.org/abs/2603.15569)
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464)
- [TFLA](https://arxiv.org/abs/2503.14376)
- [Qwen3.5 analysis](https://huggingface.co/blog/mlabonne/qwen35)
- [Hybrid Linear Attention Done Right](https://arxiv.org/abs/2601.22156)
