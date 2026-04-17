# FP4/FP8 quantization-aware pretraining — Apr 2026 sweep

## Bottom line for ARIA on TRC: SKIP FP8/FP4 for now.

TPU v4/v6e has NO FP8 or FP4 MXU. Emulation costs more than BF16 native.
Stay BF16 + optimizer-state sharding = already optimal for our compute.

## TOP 3 techniques (for future H100/B200 access)

### 1. NVFP4 Pretraining (NVIDIA Sep 2025)
- Paper: arXiv 2509.25149
- Claim: 4-bit end-to-end at FP8 loss parity (MMLU-pro 62.58 vs 62.62 FP8), 2-3× throughput
- Evidence <1B: Quartet II reports 2.4× BF16 throughput at 1B scale
- Recipe:
  - 16-element micro-blocks + E4M3 block scale + FP32 tensor scale (two-level)
  - Random Hadamard Transform (16×16) on activations/gradients
  - Stochastic rounding on backward
  - 2D quantization (forward+backward consistent)
  - Keep LM head + embeddings + LayerNorm in BF16
- LOC: ~0 with Transformer Engine ≥2.13 `fp4_autocast`
- Risk: 0.1-0.3% MMLU gap; non-deterministic (stochastic rounding)
- HW: B200 required for native (H100 emulates, no speedup)

### 2. TransformerEngine FP8 Hybrid + DeepSeek-V3 scaling
- DeepSeek-V3 recipe (production)
- Claim: E4M3 forward / E5M2 backward + tile-wise scaling, loss Δ <0.25% vs BF16, 1.8× throughput
- Recipe:
  - 1×128 per-token activation tiles, 128×128 per-block weight scaling
  - CUDA-core high-precision accumulate
  - MXFP8BlockScaling (E8M0 scale, group-32)
- LOC: ~20 using `te.fp8_autocast`
- Risk: Loss spikes if LayerNorm/softmax/router go FP8 — keep BF16
- HW: H100/H200/B200 native

### 3. Q-GaLore — ONLY TPU/T4-compatible option
- Paper: arXiv 2407.08296
- INT4 projection + INT8 weights + stochastic rounding
- Evidence: LLaMA-7B pretraining on RTX 4060 Ti (16GB)
- LOC: ~200 drop-in optimizer + projection quantizer
- Risk: Low-rank approximation may hurt ARIA's SSM state dynamics — ablate on Mamba layers
- HW: Works on ANY hardware (storage quantization, not compute)

## Decision matrix for ARIA

| Compute | Recipe | Speedup | When |
|---|---|---|---|
| TRC TPU (current) | Stay BF16 | - | Now |
| Kaggle T4 (fp16 max) | Q-GaLore for memory | 0× compute, 8× memory | If OOM on 1B |
| Lambda H100 grant | TE FP8 hybrid | 1.8× | Phase 2 scale-up |
| B200 access | NVFP4 | 2.4× | Phase 3, ≥1B scale |

## Kaggle T4 specific
- NO FP4 hardware speedup possible
- MXFP4/MXFP6 storage useful for INFERENCE ONLY (7.5× memory savings vs FP32)
- For pretraining on T4: Q-GaLore is the only real option (INT storage quantization)

## Sources
- [NVFP4 arXiv 2509.25149](https://arxiv.org/abs/2509.25149)
- [NVIDIA NVFP4 blog](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
- [Bridging the Gap microscaling FP4 (2509.23202)](https://arxiv.org/html/2509.23202)
- [Transformer Engine 2.13 FP8/FP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [DeepSeek-V3 FP8 recipe](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/)
- [Q-GaLore](https://arxiv.org/html/2407.08296v1)
- [OCP Microscaling Formats](https://fprox.substack.com/p/ocp-mx-scaling-formats)
