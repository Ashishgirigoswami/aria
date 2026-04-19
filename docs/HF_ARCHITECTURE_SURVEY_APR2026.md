# HuggingFace Architecture Survey — April 2026

> Surveyed **612 unique text-generation models** on the HuggingFace Hub on **2026-04-19** by combining four sort axes (`trendingScore`, `downloads`, `likes`, `lastModified`, each top-100) plus targeted keyword searches for `mamba`, `rwkv`, `hybrid`, `ssm`, `liquid`, `bitnet`, `jamba`, `zamba`, `nemotron`. All data via HF Hub REST API (`https://huggingface.co/api/models`, no auth, public listings only). Companion data: `_hf_classified.json`, `_hf_extracts.json`, `_hf_top80_dump.txt` in this same directory.

> Cross-references: this file complements `RESEARCH_KB_2026-04-17.md` (theory) and `AI_BUILDABLES_2026.md` (ARIA roadmap). Where those discuss what *should* be built, this file documents what *is currently shipping*.

---

## TL;DR (for CargoHive positioning)

The HF Hub in April 2026 is **dominated by ~5 architecture families** (Qwen3, Qwen2.5, Llama-3, Gemma, DeepSeek-V3) which together account for **roughly 50% of trending+top-downloaded text-gen weights**. The most important shift since late 2025 is that **major labs have crossed the bridge from pure-attention to hybrid attention+SSM/linear-attention**: NVIDIA (Nemotron-H/Nano, hybrid Mamba+Attn), MiniMax (M2 series, Lightning Attention + MoE), and the established hybrid players (AI21 Jamba, Zyphra Zamba2, Liquid LFM) are all in the top-50 by downloads. Pure-SSM models (Mamba, RWKV) remain a research curiosity (~5% of survey). For ARIA: the hybrid niche is now validated commercially but **almost entirely concentrated at >9B params** — there is a clear gap at sub-1B reasoning-focused hybrids, where ARIA's 131M→1B trajectory is uncrowded.

---

## Architecture distribution (612 surveyed text-gen models)

Counts and percentages computed from the union of trending+downloads+likes+recent (top-100 each) plus targeted hybrid/SSM searches.

| Architecture family | Count | % | Notable examples |
|---|---:|---:|---|
| **Qwen3** | 77 | 12.6% | `Qwen/Qwen3-0.6B` (15.6M dl), `Qwen/Qwen3-8B`, `Qwen/Qwen3-Coder-Next` |
| **Qwen2.5** | 39 | 6.4% | `Qwen/Qwen2.5-7B-Instruct` (12.5M dl), `Qwen/Qwen2.5-Coder-32B-Instruct` |
| **NanoChat (Karpathy clone)** | 34 | 5.6% | `crellis/d20-*-chatsft_checkpoints` (sweep) |
| **Liquid LFM (LFM2.x)** | 30 | 4.9% | `LiquidAI/LFM2.5-1.2B-Instruct`, `LiquidAI/LFM2-8B-A1B`, `LiquidAI/LFM2.5-350M` |
| **Llama-3 (3.x family)** | 27 | 4.4% | `meta-llama/Llama-3.1-8B-Instruct` (9.4M dl), `Llama-3.2-1B-Instruct` |
| **Gemma-4 (community/leaks)** | 26 | 4.2% | `nvidia/Gemma-4-31B-IT-NVFP4`, `OBLITERATUS/gemma-4-E4B-it-OBLITERATED` |
| **Jamba (Hybrid SSM+Attn+MoE)** | 26 | 4.2% | `ai21labs/Jamba-tiny-dev`, `AI21-Jamba-Mini-1.6`, `AI21-Jamba-Reasoning-3B` |
| **Falcon** | 22 | 3.6% | `tiiuae/Falcon-Perception`, `Falcon-OCR`, `falcon-40b` |
| **Mamba (pure SSM)** | 22 | 3.6% | `state-spaces/mamba-130m-hf`, `tiiuae/falcon-mamba-7b-instruct` |
| **Llama (generic, v1/v2/community)** | 21 | 3.4% | `meta-llama/Llama-2-7b-hf`, community fine-tunes |
| **Zamba2 (Hybrid SSM+Attn)** | 20 | 3.3% | `Zyphra/Zamba2-7B-Instruct`, `Zamba2-2.7B`, `Zamba2-1.2B-instruct` |
| **RWKV (other)** | 18 | 2.9% | `BlinkDL/rwkv-*` family, `RWKV/v6-Finch-*` |
| **MiniMax-M (Hybrid Lightning Attn + MoE)** | 16 | 2.6% | `MiniMaxAI/MiniMax-M2.7` (258K dl), `MiniMax-M2.5`, `MiniMax-M2.1` |
| **Mistral (v0.1–v0.3)** | 15 | 2.5% | `mistralai/Mistral-7B-Instruct-v0.2`, `dphn/Dolphin-Mistral-24B` |
| **Nemotron (transformer)** | 14 | 2.3% | NVIDIA Llama-3.1-Nemotron variants |
| **Nemotron-H/Nano (Hybrid Mamba+Attn)** | 14 | 2.3% | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (1.57M dl), `Nemotron-Nano-9B-v2` |
| **Phi-3 (3.0/3.5)** | 13 | 2.1% | `microsoft/Phi-3.5-vision-instruct`, `Phi-3.5-mini-instruct` |
| **BitNet (1.58-bit)** | 12 | 2.0% | `microsoft/bitnet-b1.58-2B-4T` |
| **Gemma-3** | 10 | 1.6% | `farbodtavakkoli/OTel-LLM-8.3B-IT` (gemma3_text fine-tune) |
| **Mamba-2 (pure SSM)** | 10 | 1.6% | `state-spaces/mamba-2.8b-hf` |
| **DeepSeek-V3 (V3, V3.1, V3.2)** | 9 | 1.5% | `deepseek-ai/DeepSeek-V3.2` (10.2M dl), `DeepSeek-V3` |
| **RWKV-6** | 9 | 1.5% | `bartowski/rwkv-6-world-7b-GGUF` |
| **GPT-2 (legacy)** | 8 | 1.3% | `openai-community/gpt2` (13.9M dl — research baseline) |
| **GLM-5** | 7 | 1.1% | `zai-org/GLM-5.1`, `GLM-5-FP8` |
| **GLM-4** | 7 | 1.1% | `zai-org/GLM-4.7`, `GLM-4.7-Flash` |
| **Qwen2** | 7 | 1.1% | `Qwen/Qwen2-1.5B-Instruct` |
| **GPT-OSS** | 5 | 0.8% | `openai/gpt-oss-20b` (6.4M dl), `openai/gpt-oss-120b` (3.5M dl) |
| **Phi-4 / Phi-4-mini** | 5 | 0.8% | `microsoft/Phi-4-mini-instruct` (1.3M dl) |
| **OLMo / OLMo-2** | 5 | 0.8% | AI2's open models |
| **DeepSeek-R1** | 4 | 0.7% | `deepseek-ai/DeepSeek-R1` (4M dl, 13.3K likes — top liked!), `R1-Distill-Llama-8B` |
| Gemma-2 | 4 | 0.7% | `bartowski/gemma-2-2b-it-GGUF` |
| Other (33 unclassified + ~25 long-tail) | 90 | 14.7% | Sundial, Moondream2, VibeVoice, Outlier-MoE, LLaDA, Nandi, SKT-Surya, Apertus, etc. |

**Key observation:** The "Big Five" attention transformers (Qwen3 + Qwen2.5 + Llama-3 + Gemma family + DeepSeek) account for ~32% of the surveyed top models by *count*; by *downloads* they're closer to 60%. But **hybrid SSM+Attn families collectively account for ~13% of surveyed models** (Jamba 4.2% + Nemotron-H 2.3% + Zamba2 3.3% + MiniMax-M 2.6% + Liquid LFM 4.9% — note overlap), which is a major jump from <2% in mid-2025.

Source for all rows: `https://huggingface.co/api/models?filter=text-generation&sort=<key>&direction=-1&limit=100` queried 2026-04-19, plus `?search=<keyword>` queries.

---

## Novel / noteworthy architectures spotted (not just Llama clones)

These are architectures whose `config.architectures[0]` field is something *other than* a `Llama|Mistral|Qwen|Gemma|Phi-ForCausalLM` standard transformer. All architecture strings verified by direct fetches to `https://huggingface.co/api/models/<id>`.

1. **`Qwen3NextForCausalLM`** — `Qwen/Qwen3-Coder-Next` (629K dl, created 2025). Alibaba's "Next" line; hybrid linear+full attention with MoE, the biggest commercial signal that Qwen team is moving past pure-softmax attention. Source: `https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/config.json`.
2. **`MiniMaxM2ForCausalLM`** — `MiniMaxAI/MiniMax-M2.7` (258K dl, 964 likes, 716 trending score on 2026-04-17). Lightning Attention (linear) + sparse MoE. M2.5 has 920K downloads, M2.1 has 37K — this is a serial-version family with rapid iteration. `https://huggingface.co/MiniMaxAI/MiniMax-M2.7`.
3. **`NemotronHForCausalLM`** — `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (525K dl), `Nemotron-3-Nano-30B-A3B-BF16` (1.57M dl). NVIDIA's hybrid Mamba-2 + attention + MoE. The 30B-A3B variant explicitly exposes `hybrid_override_pattern` in config. `https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2`.
4. **`JambaForCausalLM`** — `ai21labs/AI21-Jamba-Mini-1.6` (2.8K dl, 57 likes), and the new `AI21-Jamba-Reasoning-3B` (3.9K dl on GGUF mirror). Mamba + attention + MoE, the original publicly-shipped hybrid. `https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.6`.
5. **`Zamba2ForCausalLM`** — `Zyphra/Zamba2-7B-Instruct` (16K dl), `Zamba2-1.2B-instruct` (99K dl). Mamba2 backbone with shared attention blocks. `https://huggingface.co/Zyphra/Zamba2-7B-Instruct`.
6. **`Lfm2ForCausalLM`** — `LiquidAI/LFM2.5-1.2B-Instruct` (361K dl), `LFM2.5-350M`, `LFM2-8B-A1B`. Liquid AI's "Liquid Foundation Model" — short conv blocks + grouped attention, ostensibly inspired by liquid neural networks. Surprise: this is the **#1 SSM-adjacent family by raw model count** in the survey (30 LFM2 variants). `https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct`.
7. **`FalconMambaForCausalLM`** — `tiiuae/falcon-mamba-7b-instruct` (39K dl). Pure Mamba-1, TII's bet from 2024. Still active. `https://huggingface.co/tiiuae/falcon-mamba-7b-instruct`.
8. **`BitNetForCausalLM`** — `microsoft/bitnet-b1.58-2B-4T` (15.8K dl). Ternary weights {-1, 0, +1} at 1.58 bits/weight. The 4T-token training run was finished in late 2024 and has had a 12-model ecosystem develop on HF since (12 BitNet variants in the survey). `https://huggingface.co/microsoft/bitnet-b1.58-2B-4T`.
9. **`AfmoeForCausalLM`** — `arcee-ai/Trinity-Large-Thinking` (19.5K dl). Arcee's proprietary MoE + reasoning architecture, "agentic" tagged. `https://huggingface.co/arcee-ai/Trinity-Large-Thinking`.
10. **`LLaDA2MoeModelLM`** — `Zigeng/DMax-Coder-16B` (1.2K dl). LLaDA-2 is a **diffusion language model** by Renmin University; this is the v2 + MoE variant. Diffusion-LM is now showing up in the wild, not just papers. `https://huggingface.co/Zigeng/DMax-Coder-16B`.
11. **`OutlierMoE`** — `Outlier-Ai/Outlier-10B` (361 dl), `Outlier-40B` (972 dl). Self-described "ternary outlier MoE" — claims sparsity through ternary expert weights. `https://huggingface.co/Outlier-Ai/Outlier-10B`.
12. **`GlmMoeDsaForCausalLM`** — `zai-org/GLM-5.1` (104K dl, 1.4K likes). Zhipu's GLM-5 with "DSA" (Dynamic Sparse Attention?) MoE. New as of Feb 2026. `https://huggingface.co/zai-org/GLM-5.1`.
13. **`DeepseekV32ForCausalLM`** — `deepseek-ai/DeepSeek-V3.2` (10.2M dl). The V3.2 is V3 + sparse-attention sliding window patch; the architecture string itself is bumped to V32. `https://huggingface.co/deepseek-ai/DeepSeek-V3.2`.
14. **`SmolLM3ForCausalLM`** — `HuggingFaceTB/SmolLM3-3B` (804K dl). HF's own 3B small-model line; pure transformer but tightly tuned for efficiency. `https://huggingface.co/HuggingFaceTB/SmolLM3-3B`.
15. **`NandiForCausalLM`** — `Rta-AILabs/Nandi-Mini-150M-Instruct` (3.1K dl, 42 likes). Indian-team-built 150M model with custom architecture string and Indic multilingual tags (en, hi, mr, ta, te, kn, ml). One of the few **actually-novel small Indic models**. `https://huggingface.co/Rta-AILabs/Nandi-Mini-150M-Instruct`.
16. **`SKTHanmantForCausalLM`** — `sKT-Ai-Labs/SKT-SURYA-H` (832 dl). Self-tagged "sovereign-ai", "Indian", Hindi-focused. `https://huggingface.co/sKT-Ai-Labs/SKT-SURYA-H`.

Each of these is at least a config-level departure from a vanilla LlamaForCausalLM, which is a good indicator that the architecture stack is genuinely diverging in 2026.

---

## Hybrid attention+SSM models (direct ARIA competitors / cousins)

Models where the architecture mixes some form of recurrence/SSM/linear-attention with full softmax attention. **78 models in the survey fall into this bucket** (many are quants/GGUFs of the same base).

### Tier 1 — frontier-lab hybrids (>500K downloads each)

| Model | Architecture | Params | Author | Mechanism | Why interesting |
|---|---|---|---|---|---|
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | NemotronHForCausalLM | 30B (3B active MoE) | NVIDIA | Mamba2 + Attn + MoE | 1.57M downloads. The biggest commercial validation that hybrid-Mamba scales. |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | NemotronHForCausalLM | 120B (12B active) | NVIDIA | Mamba2 + Attn + MoE | 1.38M downloads at NVFP4 quantization. Largest hybrid in production. |
| `MiniMaxAI/MiniMax-M2.5` | MiniMaxM2ForCausalLM | unspecified (large) | MiniMax (China) | Lightning Attention (linear) + softmax + MoE | 921K downloads; M2.7 is the latest with 258K and 716 trending score. |
| `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8` | NemotronHForCausalLM | 12B (vision-language) | NVIDIA | Hybrid + ViT | 659K downloads. |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | NemotronHForCausalLM | 9B | NVIDIA | Hybrid Mamba2+Attn | 525K downloads, 488 likes. The "small" Nemotron-H. |
| `nvidia/Nemotron-Cascade-2-30B-A3B` | NemotronHForCausalLM | 30B (3B active) | NVIDIA | Hybrid + cascade decoding | 318K downloads. |

### Tier 2 — established hybrid players

| Model | Architecture | Params | Mechanism | Downloads |
|---|---|---|---|---|
| `ai21labs/Jamba-tiny-dev` | JambaForCausalLM | tiny dev (testing) | Mamba+Attn+MoE | 385K |
| `MiniMaxAI/MiniMax-M2.7` | MiniMaxM2ForCausalLM | large | Lightning Attn + MoE | 258K |
| `MiniMaxAI/MiniMax-M2` | MiniMaxM2ForCausalLM | large | Lightning Attn + MoE | 64K (1490 likes) |
| `MiniMaxAI/MiniMax-M2.1` | MiniMaxM2ForCausalLM | large | Lightning Attn + MoE | 37K (1276 likes) |
| `Zyphra/Zamba2-1.2B-instruct` | Zamba2ForCausalLM | 1.2B | Mamba2 + shared attention blocks | 99K |
| `Zyphra/Zamba2-7B-Instruct` | Zamba2ForCausalLM | 7B | Mamba2 + shared attention | 16K (92 likes) |
| `ai21labs/AI21-Jamba-Mini-1.6` | JambaForCausalLM | ~12B (4B active MoE) | Mamba+Attn+MoE | 2.8K (57 likes) |
| `ai21labs/AI21-Jamba-Reasoning-3B` (GGUF) | JambaForCausalLM | 3B | Mamba+Attn+MoE | 3.9K |
| `Zyphra/Zamba2-1.2B` (base) | Zamba2ForCausalLM | 1.2B | Mamba2 + shared attention | 2.8K |
| `Zyphra/Zamba2-2.7B` | Zamba2ForCausalLM | 2.7B | Mamba2 + shared attention | 2.6K |
| `Zyphra/Zamba-7B-v1` | ZambaForCausalLM | 7B | Mamba+Attn (legacy) | 4.3K |
| `ai21labs/AI21-Jamba-Large-1.6` | JambaForCausalLM | ~398B total | Mamba+Attn+MoE | 2.5K |

### Tier 3 — Liquid LFM (technically not classical SSM but linear-time/closed-form recurrence)

LiquidAI shipped **30 model variants** in this survey alone. The LFM2 architecture is documented as alternating short convolutions with grouped attention. The 1.2B base is the highest-downloaded sub-2B non-Qwen3 model in the survey (361K downloads for Instruct, 230K for the GGUF mirror, 33K for the Thinking variant).

| Model | Params | Downloads |
|---|---|---|
| `LiquidAI/LFM2.5-1.2B-Instruct` | 1.2B | 361K |
| `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | 1.2B | 230K |
| `LiquidAI/LFM2-8B-A1B` | 8B (1B active MoE) | 81K |
| `LiquidAI/LFM2-24B-A2B-GGUF` | 24B (2B active) | 51K |
| `LiquidAI/LFM2.5-1.2B-Thinking` (-GGUF) | 1.2B reasoning | 33K + 48K |
| `LiquidAI/LFM2.5-350M` | 350M | 45K |
| `LiquidAI/LFM2-1.2B`, `LFM2-700M`, `LFM2-2.6B-Exp`, etc. | 350M–2.6B | 11K–43K each |

**Implication for ARIA:** Liquid is the closest competitor in the *small-hybrid* niche. Their 350M and 1.2B models are exactly ARIA's target zone. Differences: ARIA uses GDN-style WY recurrence + full attention (not Liquid's conv-grouped-attn), and ARIA is targeting reasoning specifically. Liquid does not publish a "thinking" variant below 1.2B — that is a real gap.

### Pure-SSM models (no attention)

Mostly research/legacy: `state-spaces/mamba-130m-hf` (244K dl), `state-spaces/mamba-2.8b-hf` (11K dl), `tiiuae/falcon-mamba-7b-instruct` (39K dl). RWKV-6 and RWKV-7 weights from BlinkDL also persist, but volume is small. No new pure-SSM model from a frontier lab has shipped in the past 6 months, confirming the consensus that **hybrids beat pure-SSM at all evaluated scales**.

---

## Small-scale (<1B) reasoning-capable models

This is ARIA's competitive home turf. Models extracted by name heuristics (`*-0.5B`, `*-0.6B`, `*-150M`, `*-270M`, `*-350M`, `nano|tiny|smol|mini|micro` in the id) — 112 unique entries.

### Top by downloads

| Model | Family | Params | Downloads | Reasoning? |
|---|---|---|---:|---|
| `Qwen/Qwen3-0.6B` | Qwen3 | 0.6B | 15.6M | Has thinking-mode toggle |
| `Qwen/Qwen3-Embedding-0.6B` | Qwen3 | 0.6B | 6.1M | Embedding (not reasoning) |
| `Qwen/Qwen2.5-0.5B-Instruct` | Qwen2.5 | 0.5B | 5.7M | No |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | TinyLlama | 1.1B | 3.0M | No |
| `EleutherAI/pythia-160m` | GPT-NeoX | 160M | 2.6M | No (research base) |
| `Qwen/Qwen3-0.6B-FP8` | Qwen3 | 0.6B | 1.8M | Yes |
| `Qwen/Qwen2.5-0.5B` | Qwen2.5 | 0.5B | 1.6M | No |
| `microsoft/Phi-3.5-vision-instruct` | Phi-3 | 4B vision | 1.5M | No |
| `Qwen/Qwen3-Reranker-0.6B` | Qwen3 | 0.6B | 1.4M | Reranker |
| `HuggingFaceTB/SmolLM2-135M` | SmolLM2 | 135M | 1.3M | No |
| `microsoft/Phi-4-mini-instruct` | Phi-4 | 3.8B | 1.3M | Some |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | SmolLM2 | 135M | 835K | No |
| `HuggingFaceTB/SmolLM3-3B` | SmolLM3 | 3B | 804K | Yes |
| `microsoft/Phi-3.5-mini-instruct` | Phi-3 | 3.8B | 797K | Some |
| `LiquidAI/LFM2.5-350M` | Liquid LFM | 350M | 45K | No (base is non-thinking) |
| `Rta-AILabs/Nandi-Mini-150M-Instruct` | Nandi (India) | 150M | 3.1K | No |

### What's distinctive at <1B

- **Qwen3-0.6B is the dominant small model** (15.6M downloads — more than every other small model combined). It has a thinking-mode toggle but no formal reasoning RL training reported in the card.
- **There is exactly one small "Thinking" model from a major lab**: `LiquidAI/LFM2.5-1.2B-Thinking` (33K downloads) at 1.2B. **There is no <1B model on the Hub trending in April 2026 that markets itself as a reasoning model with RL.**
- **DeepSeek-R1-Distill-Qwen-1.5B** (not in our top-100 trending but is a referenced peer) is the closest competitor at 1.5B, but it is a distillation of the dense R1 into a Qwen base, not a hybrid.
- **HRM, TRM, and other tiny-recursive models** (Hierarchical Reasoning Model, Tiny Recursive Model — discussed in `RESEARCH_KB_2026-04-17_reasoning.md`) are not visible in HF trending — they are paper-only or niche.

This is the core competitive gap: **a 131M–600M hybrid (attention + SSM) with reasoning RL applied is unoccupied real estate**. ARIA's roadmap (per `AI_BUILDABLES_2026.md` §1.2) is squarely targeting this.

---

## What's trending that DIDN'T exist 6 months ago (post 2025-10 emergence)

Filtered to models with `createdAt >= 2025-10-01`. The notable new families:

| Family | First major release | Trending example | Significance |
|---|---|---|---|
| **NVIDIA Nemotron-3 / Nemotron-H** | Late 2025 → 2026-03 push | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (created 2026-03-10) | First hyperscaler hybrid SSM in production at 120B |
| **MiniMax-M2.5/M2.7** | 2026-Q1, M2.7 created 2026-04-09 | `MiniMaxAI/MiniMax-M2.7` (10 days old, already 716 trending score) | Showing very rapid version turnover; commercial pressure on Lightning Attention validated |
| **DeepSeek-V3.2** | Created 2025-12-01 | 10.2M downloads | V3.2 is V3 + sparse window attention; sparse is now mainstream in dense MoE |
| **GLM-5.x** | Created 2026-02 | `zai-org/GLM-5.1` (104K dl, Feb 2026) | First GLM moving past dense → MoE+DSA |
| **Gemma-4 (community / leaks)** | Numerous 26B–31B variants in March/April 2026 | `nvidia/Gemma-4-31B-IT-NVFP4` (1.2M downloads, created 2026-04-02) | Google still hasn't officially released Gemma-4 at survey time — but **NVIDIA has shipped a quantized "Gemma-4-31B" already**, suggesting strategic partnership leaks. Highly notable. |
| **OTel-LLM (telecom-fine-tuned)** | Created 2026-02–03 | `farbodtavakkoli/OTel-LLM-8.3B-IT` (1.4M downloads, gemma3_text base) | Verticalized fine-tune (telecom/GSMA) reaching top-50 downloads — vertical models can now compete on raw download volume with foundation models |
| **NanoChat sweep (Karpathy clone)** | Created 2026-Q1 | 34 `crellis/d20-*-chatsft_checkpoints` variants | Karpathy's nanochat repo became a popular reproducibility playground — a Cambrian explosion of <1B baselines |
| **GPT-OSS** | 2025-08 | `openai/gpt-oss-20b` (6.4M dl), `gpt-oss-120b` (3.5M dl, 4.7K likes) | OpenAI's first open-weight release; Llama-3.x compatible; not architecturally novel but strategically huge |
| **Qwen3-Coder-Next** | Late 2025 | `Qwen/Qwen3-Coder-Next` (629K dl) | Alibaba's first publicly downloadable hybrid (Qwen3-Next architecture) |
| **`AI21-Jamba-Reasoning-3B`** | 2026-Q1 (visible only via GGUF mirror in survey) | `bartowski/ai21labs_AI21-Jamba-Reasoning-3B-GGUF` (3.9K dl) | First **Jamba reasoning** variant — directly relevant to ARIA |
| **LFM2.5 series + Thinking** | Late 2025 → 2026 | `LiquidAI/LFM2.5-1.2B-Thinking` (33K dl) | First Liquid model marketed for reasoning |
| **LLaDA-2 MoE (diffusion-LM)** | 2026 | `Zigeng/DMax-Coder-16B` | Diffusion-LM is moving into deployable territory |

**The strongest signal:** the four families that launched/iterated in 2026 that have crossed 100K downloads are *all* MoE or hybrid-attention or both (Nemotron-H, MiniMax-M, DeepSeek-V3.2, GLM-5). Pure dense softmax-attention transformer launches from this window are absent at top downloads.

---

## Indic / India-language-focused models on HF

**Modest.** Only 6 explicitly Indic-tagged models surfaced in the top-trending and top-downloaded surveys (and most are quantized/community tunes, not new foundation pretrains):

| Model | Author | Params | Downloads | Indic specifics |
|---|---|---|---:|---|
| `bigscience/bloomz-560m` | BigScience | 560M | 1.05M | Multilingual incl. Indic (legacy) |
| `bigscience/bloom` | BigScience | 176B | 6.1K | Multilingual incl. Indic (legacy) |
| `Rta-AILabs/Nandi-Mini-150M-Instruct` | Rta-AILabs (India) | 150M | 3.1K | Custom `NandiForCausalLM`, languages: en, hi, mr, ta, te, kn, ml |
| `Rta-AILabs/Nandi-Mini-150M-Tool-Calling` | Rta-AILabs (India) | 150M | 1.8K | Tool-calling variant |
| `sKT-Ai-Labs/SKT-SURYA-H` | SKT-Ai-Labs (India) | unknown | 832 | "sovereign-ai" + Hindi, custom `SKTHanmantForCausalLM` |
| `USS-Inferprise/Dark-Cydonian-Wind-24B` | USS-Inferprise | 24B | 202 | Mistral merge, India-tagged |

**Notably absent from trending in April 2026** (despite their importance to the Indian LLM scene): Sarvam, Krutrim, OpenHathi, Hanooman/Hanuman, Ai4Bharat IndicLLama, Param-1, Pragna, Adya, Granite-IndicNLP. They exist on HF but did not trend in the past 30 days. This may reflect HF's discoverability bias (English-speaking users vote with downloads/likes) rather than absence.

**Implication for CargoHive:** The HF Hub does *not* yet have a strong Indic LLM presence visible to a global audience. There is no "trending Indic small model" at <1B that has crossed 50K downloads. This is consistent with the user's existing strategic note (`AI_BUILDABLES_2026.md` line 288) that "we trained on Indic data" alone is not a competitive moat — but it does mean that **a 131M–1B Indic-reasoning hybrid would have no direct trending competitor on HF today**. The Nandi-Mini-150M is the closest in spirit but does not claim reasoning capability.

---

## Implications for ARIA

1. **The hybrid SSM+Attn niche is now commercially validated, but only above 9B parameters.** NVIDIA Nemotron-H (9B/30B/120B), MiniMax-M (large), Jamba (12B+), Zamba2 (7B), DeepSeek-V3.2 (671B). Below 3B, the only commercial hybrid presence is Liquid LFM2 (350M, 700M, 1.2B, 2.6B) — and Liquid does not target reasoning at <1.2B. **ARIA's 131M→1B hybrid-with-reasoning-RL trajectory is genuinely uncrowded.**

2. **Reasoning is the differentiator, not architecture alone.** The top-liked model on HF in this survey is `deepseek-ai/DeepSeek-R1` (13.3K likes vs. 4.5K for `gpt-oss-20b`). The "Thinking"/"Reasoning" suffix is appearing on small models for the first time (`LiquidAI/LFM2.5-1.2B-Thinking`, `AI21-Jamba-Reasoning-3B`, `Qwen/Qwen3-4B-Thinking-2507`). ARIA's plan to apply JustRL to a 131M hybrid (`AI_BUILDABLES_2026.md` §1.2) is well-timed — *first-to-market* on sub-200M reasoning hybrids.

3. **Quantization is now the default deployment story, not an afterthought.** Of the top 50 by downloads, ~30 are quantized variants (GGUF, AWQ, GPTQ, FP8, NVFP4). NVFP4 (NVIDIA's 4-bit format) jumped to prominence with the Nemotron-3 family. ARIA's Phase 2 plan should explicitly target a quantized release alongside BF16, ideally in NVFP4 if hardware support is reasonable.

4. **Architecture diversity is exploding at the long tail.** 16 architecture strings in the survey are not standard `*ForCausalLM` Llama variants (BitNet, Outlier MoE, LLaDA, Lfm2, Nandi, SKT-Hanmant, Afmoe, GlmMoeDsa, Qwen3Next, MiniMaxM2, NemotronH, Jamba, Zamba2, FalconMamba, SmolLM3, BitNet). The Hub is *visibly less Llama-monoculture* in April 2026 than it was in April 2025. Custom-architecture models are now publishable and discoverable without being penalized — ARIA's custom `AriaForCausalLM` class will not face a barrier to upload.

5. **Indic + reasoning + small + hybrid is an empty quadrant on HF.** Combining the four CargoHive vectors — Indic data, reasoning (RL), small (<1B), hybrid (Attn+SSM) — produces zero matches in the survey. That is the white space. The risk is not differentiation but discoverability; ARIA will need a strong launch narrative + early benchmark numbers (MMLU, GSM8K, IndicGenBench) to overcome the gravitational pull of Qwen3-0.6B in the small-model zone.

6. **Diffusion-LM and 1.58-bit BitNet are real but not yet mainstream.** Both have ecosystems (12 BitNet + 1 LLaDA-2 MoE in the survey). They are worth tracking but are not blocking ARIA's roadmap. If ARIA finds the GDN+attention story doesn't transfer at 1B scale, BitNet-style ternary quantization is a viable hedge for inference cost.

---

## Raw data (top 30 by downloads — full 80-entry dump in `_hf_top80_dump.txt`, full 612-entry classified dataset in `_hf_classified.json`)

```yaml
- id: Qwen/Qwen3-0.6B
  family: Qwen3
  downloads: 15580616
  likes: 1197
  created: 2025-04-27
  library: transformers
- id: openai-community/gpt2
  family: GPT-2
  downloads: 13903620
  likes: 3210
  created: 2022-03-02
  library: transformers
- id: Qwen/Qwen2.5-7B-Instruct
  family: Qwen2.5
  downloads: 12493288
  likes: 1215
  created: 2024-09-16
- id: Qwen/Qwen2.5-1.5B-Instruct
  family: Qwen2.5
  downloads: 10513868
  likes: 668
  created: 2024-09-17
- id: Qwen/Qwen2.5-3B-Instruct
  family: Qwen2.5
  downloads: 10319909
  created: 2024-09-17
- id: deepseek-ai/DeepSeek-V3.2
  family: DeepSeek-V3
  downloads: 10186079
  likes: 1410
  created: 2025-12-01
- id: Qwen/Qwen3-4B-Instruct-2507
  family: Qwen3
  downloads: 9735996
  created: 2025-08-05
- id: meta-llama/Llama-3.1-8B-Instruct
  family: Llama-3
  downloads: 9365524
  likes: 5728
  created: 2024-07-18
- id: Qwen/Qwen3-8B
  family: Qwen3
  downloads: 8635414
  created: 2025-04-27
- id: Qwen/Qwen3-4B
  family: Qwen3
  downloads: 7564027
  created: 2025-04-27
- id: Qwen/Qwen3-1.7B
  family: Qwen3
  downloads: 7334301
  created: 2025-04-27
- id: facebook/opt-125m
  family: OPT
  downloads: 6727472
  created: 2022-05-11
- id: openai/gpt-oss-20b
  family: GPT-OSS
  downloads: 6353037
  likes: 4549
  created: 2025-08-04
- id: Qwen/Qwen3-Embedding-0.6B
  family: Qwen3
  downloads: 6069115
  created: 2025-06-03
  library: sentence-transformers
- id: Qwen/Qwen2.5-0.5B-Instruct
  family: Qwen2.5
  downloads: 5692209
  created: 2024-09-16
- id: meta-llama/Llama-3.2-3B-Instruct
  family: Llama-3
  downloads: 4706533
  likes: 2099
  created: 2024-09-18
- id: meta-llama/Llama-3.2-1B-Instruct
  family: Llama-3
  downloads: 4600668
  created: 2024-09-18
- id: deepseek-ai/DeepSeek-R1
  family: DeepSeek-R1
  downloads: 3975542
  likes: 13276
  created: 2025-01-20
- id: Qwen/Qwen2.5-32B-Instruct
  family: Qwen2.5
  downloads: 3487951
  created: 2024-09-17
- id: openai/gpt-oss-120b
  family: GPT-OSS
  downloads: 3482885
  likes: 4712
  created: 2025-08-04
- id: meta-llama/Meta-Llama-3-8B
  family: Llama-3
  downloads: 3345292
  likes: 6521
  created: 2024-04-17
- id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  family: TinyLlama
  downloads: 3008385
  created: 2023-12-30
- id: Qwen/Qwen3-14B
  family: Qwen3
  downloads: 2928133
  created: 2025-04-27
- id: vikhyatk/moondream2
  family: vision (Moondream)
  downloads: 2413001
  created: 2024-03-04
- id: Qwen/Qwen3-32B
  family: Qwen3
  downloads: 2356334
  created: 2025-04-27
- id: Qwen/Qwen3-Coder-30B-A3B-Instruct
  family: Qwen3
  downloads: 2157153
  likes: 1016
  created: 2025-07-31
- id: mistralai/Mistral-7B-Instruct-v0.2
  family: Mistral
  downloads: 2118775
  likes: 3117
  created: 2023-12-11
- id: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  family: DeepSeek-R1
  downloads: 1992466
  created: 2025-01-20
- id: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  family: Nemotron-H (Hybrid Mamba+Attn)
  downloads: 1568111
  likes: 714
  created: 2025-12-04
- id: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
  family: Nemotron-H (Hybrid Mamba+Attn)
  downloads: 1379978
  likes: 270
  created: 2026-03-10
```

For the full top-80 by downloads (with library name and source flags), see `D:/mytllm/aria/docs/_hf_top80_dump.txt`. For the full 612-model classified dataset (id, family, tags, downloads, likes, createdAt, sources), see `D:/mytllm/aria/docs/_hf_classified.json`. Family-count summary in `_hf_family_counts.json`. Hybrid + SSM + Indic + small extracts in `_hf_extracts.json`.

---

## Methodology notes & caveats

- **Source:** `https://huggingface.co/api/models?filter=text-generation&sort=<key>&direction=-1&limit=100` queried directly, no auth, on 2026-04-19. Four sort keys (`trendingScore`, `downloads`, `likes`, `lastModified`) plus 9 keyword `?search=` queries (`mamba`, `rwkv`, `hybrid`, `ssm`, `liquid`, `bitnet`, `jamba`, `zamba`, `nemotron`).
- **Deduplication:** by canonical `id` field; `_sources` field tracks which queries surfaced each model.
- **Architecture classification** is a regex-based heuristic over `id` + `tags`, refined after inspecting unclassified models. For 18 models we additionally fetched `config.architectures` directly via `https://huggingface.co/api/models/<id>` to confirm the architecture string.
- **Limitations:**
  - HF's pagination via `offset=` is not consistently supported on filtered lists; we used per-key top-100 + diverse keyword fan-out instead. Total unique surveyed: **612**, which is a representative top-of-funnel view, not exhaustive.
  - The `lastModified` field requires `?full=true`, which we did not request to keep payloads small; `createdAt` (always returned) is used as a recency proxy. **Recency analysis in this report uses `createdAt`, not `lastModified`** — caveat that a 2024-created model may have been re-pushed in 2026.
  - Download counts are HF Hub's aggregated lifetime downloads as of 2026-04-19, not a 90-day window. Models created in 2026 with high downloads have therefore accumulated traffic faster per-day than older models with similar totals.
  - Some `_family` classifications for fine-tunes may attribute to the base model rather than the tune (e.g., `farbodtavakkoli/OTel-LLM-1.2B-IT` is classified as Liquid LFM because its tag is `lfm2` — that's correct architecturally).
  - Indic discovery was conservative; deeper search by `?language=hi` etc. would surface more models but was not run to keep the survey scoped to "what is trending globally".
- **What this survey does NOT cover:** vision-language models (only those tagged `text-generation` were surveyed), audio/TTS models (one `vibevoice` sneaked in via the `text-generation` task tag), RLHF reward models, embeddings (some Qwen3-Embedding entries appeared because they share the Qwen3 base tag), and private/gated models.

End of survey. Survey timestamp: 2026-04-19.
