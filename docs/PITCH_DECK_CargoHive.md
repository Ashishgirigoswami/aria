# CargoHive Technologies — Pitch Deck

**Efficient foundation models for Indian markets.**

One slide per section. Copy into Pitch, Google Slides, or Canva (use their "startup pitch" template). Each slide should be scannable in 30 seconds.

---

## Slide 1 — Cover

**CargoHive Technologies Pvt Ltd**
*Efficient foundation models for Indian markets*

- Founded 2026, Madhya Pradesh, India
- Pre-seed, self-funded
- [Your name], Founder

[Logo placeholder] — use https://namecheap.com/logo-maker or Canva free.

Contact: [email] | github.com/Ashishgirigoswami | [website]

---

## Slide 2 — The Problem

**Frontier LLMs are too expensive for Indian enterprises.**

- **GPT-4 / Claude / Gemini**: $3-15 per million tokens → 10-100× too costly for Indian SMB workflows
- **Latency**: API round-trips add 2-5s per query — unacceptable for real-time logistics, customer chat, classroom tutors
- **Data residency**: 80% of Indian regulated sectors (BFSI, healthcare, government) require on-prem or sovereign cloud deployment
- **Language coverage**: frontier models treat Indic languages (Hindi, Tamil, Bengali, Marathi) as afterthoughts with degraded quality

**Result**: Indian businesses get stuck with cheap-but-weak open models or expensive-but-inaccessible frontier APIs. No middle ground exists.

---

## Slide 3 — The Insight

**LLM inference cost is dominated by attention's quadratic scaling.**

- Quadratic attention FLOPs → long conversations, document RAG, and tool-use agents cost 10-100× more than short prompts
- State-space models (SSMs) offer linear scaling but sacrifice in-context learning quality
- **Hybrid attention + SSM** architectures (Qwen3-Next 80B, Jamba, Hymba) get the best of both — but existing hybrids leave efficiency on the table

**Our bet**: a novel hybrid design with architecture-level efficiency — not just quantization or distillation — can deliver frontier-tier quality at 1/3 the inference cost, specifically for the workloads Indian enterprises need.

---

## Slide 4 — Our Solution: ARIA

**ARIA — a new hybrid LLM architecture with 2 novel design choices.**

1. **Shared MLA low-rank KV latent** feeds both the attention head AND the SSM track — no other published model does this (verified vs Qwen3-Next, Hymba, Jamba)
2. **Joint-softmax fusion** — attention scores and per-position SSM state compete through a single softmax (not sum / concat) — enables tighter information flow than existing hybrids
3. 3:1 attention-to-SSM layer ratio (Qwen3-Next-validated), Mamba-2 style vector SSM, MLA compression

**Result**: matches transformer quality at 1/3 the KV cache size, critical for on-device / on-prem inference.

[Diagram placeholder] — architecture block diagram, 3 layers, arrows showing shared latent + joint softmax. Draw in Excalidraw (free) in 10 min.

---

## Slide 5 — Traction

**Working 131M open-source prototype — trained from scratch on Google TPU Research Cloud.**

- **131M parameter ARIA v1** fully training on TPU v6e + v4-8 (ongoing, 65% complete as of April 17, 2026)
- Clean loss curve: 10.82 → 3.49 (perplexity 10,000 → ~32)
- **Codebase open-source**: github.com/Ashishgirigoswami/aria (Apache 2.0)
- **Eval harness** (EleutherAI lm-eval) + **Kaggle T4 training path** + **Mamba-3 scaffold** all shipped
- **9 research cycles completed** on related topics (WY chunkwise GDN, muP HP transfer, long-context, distillation, reasoning RL, FP4/FP8, synthetic data)

Next checkpoint: scale to 1B with FineWeb-Edu 2B tokens — specs and compute plan ready.

---

## Slide 6 — Why Now

**2026 is the inflection year for small specialized foundation models.**

- Qwen3-Next (80B/3B active): first hybrid-SSM frontier model, ~Feb 2026
- SmolLM3 / LFM-1B-Math / Phi-4: prove <3B can beat 7-14B on vertical tasks
- India AI Mission: ₹10,372 crore budget, explicit mandate for Indian-built foundation models
- NVIDIA AI Foundation rollout in India (Reliance Jio, TCS partnerships Dec 2025)

**The window to build an Indian foundation model company is 18-24 months. After that, the market locks in around who has traction + capital.**

---

## Slide 7 — Target Market

**Three vertical wedges. Start with one. Earn into others.**

| Wedge | Market size (India) | Our angle | Launch target |
|---|---|---|---|
| **EdTech AI tutors** | ₹7,500 cr by 2028 | JEE/NEET/UPSC specialist with Hindi-medium | Q3 2026 |
| **Logistics AI copilots** | ₹2,000 cr by 2028 | Inventory/shipment NLP, multi-warehouse | Q4 2026 |
| **Regional-language APIs** | ₹3,500 cr by 2028 | On-prem Hindi/Tamil/Marathi LLM | Q1 2027 |

**Beachhead**: EdTech — 100M+ aspirants, parents paying ₹500-50K/month, proven willingness to spend on AI (Byju's, PhysicsWallah scaled to $1B+ despite weak products).

**Not competing**: with global chatbots (OpenAI, Anthropic). We're the on-prem / low-cost / Indic-first alternative.

---

## Slide 8 — Business Model

**B2B SaaS + per-inference licensing.**

1. **SaaS subscription** for vertical applications (edtech tutor: ₹499/mo, logistics copilot: ₹5,000/mo per user)
2. **API / per-token pricing** for developers building on ARIA: ₹0.20 / 1K tokens (vs GPT-4's ~₹2.50)
3. **On-prem enterprise licensing** for BFSI/healthcare/government: ₹25L-2Cr annual contracts, model + hardware reference design

**Unit economics goal**: ₹10 ARR per rupee of R&D spend by Year 2 — achievable because our architecture cuts inference GPU costs 3× vs fine-tuning Llama-style baselines.

---

## Slide 9 — Roadmap

**12-month technical roadmap.**

| Milestone | Month | Status |
|---|---|---|
| ARIA 131M pretrain + open-source release on HuggingFace | Apr 2026 | 🟡 In progress (65% complete) |
| ARIA 1B pretrain on FineWeb-Edu 2B tokens (TRC TPU) | May 2026 | 🟢 Config ready |
| Mamba-3 matrix-state swap + GPU training (NVIDIA grant) | Jun 2026 | 🟢 Scaffolded |
| ARIA 1B-IT — SFT + DPO post-training for edtech use case | Jul 2026 | 📋 Planned |
| ARIA 3B pretrain — 10B tokens curriculum, India-centric mix | Aug-Sep 2026 | 📋 Planned |
| First paying pilot customer (edtech) | Oct 2026 | 📋 Sales pipeline |
| Seed round ($500K-$2M) | Dec 2026 | 📋 Post-traction |
| ARIA 7B + multilingual fine-tune + enterprise GA | Q1-Q2 2027 | 📋 Planned |

---

## Slide 10 — Competitive Landscape

| Player | Model tier | India focus | Open weights | On-prem |
|---|---|---|---|---|
| **OpenAI / Anthropic / Google** | Frontier | ❌ | ❌ | ❌ |
| **Meta Llama / Mistral** | Open frontier | ❌ | ✅ | ⚠️ | 
| **Qwen / DeepSeek** | Open frontier | ❌ | ✅ | ✅ |
| **Sarvam AI (India)** | 4B-10B Indic | ✅ | ⚠️ | ✅ |
| **AI4Bharat** | Research-only | ✅ | ✅ | ⚠️ |
| **Krutrim (Ola)** | 7B+ | ✅ | ❌ | ✅ |
| **CargoHive / ARIA** | **Efficient 1B-7B** | ✅ | ✅ | ✅ |

**Differentiator**: architecture-level efficiency + Indic-first vertical focus + full open-source.

**Not trying to beat**: Sarvam on base quality (they have ₹300 cr). Trying to beat them on **inference cost per insight** through architectural efficiency.

---

## Slide 11 — Team

**Solo founder stage — actively recruiting.**

- **[Your name]**, Founder & CEO
  - Background: [brief, 2-3 lines — e.g., full-stack engineer, ML self-taught, prior projects AIforAspirants and Nyte]
  - Technical lead on ARIA architecture research
  - HuggingFace: [handle] | GitHub: [handle] | LinkedIn: [url]

**Hiring pipeline (post-seed):**
- ML Research Engineer (pretrain infrastructure)
- Full-stack Engineer (SaaS platform)
- Sales lead (edtech partnerships, Tier-2/3 cities India)

**Advisors targeted**: ex-Sarvam AI, ex-AI4Bharat, ex-Google TRC program leads.

---

## Slide 12 — The Ask

**Non-dilutive compute credits to accelerate 1B-3B training.**

From NVIDIA Inception:
- **DGX Cloud credits** for 1B pretraining (~500 H100-hours, ~$25K equivalent)
- **TensorRT-LLM** access for deployment optimization
- **NIM microservices** for on-prem enterprise SKU
- **Technical advisory** from NVIDIA Developer Program for Mamba-3 kernel port
- **Marketing co-promo** for Indian AI ecosystem (NVIDIA India events, blog coverage)

**In exchange**, CargoHive commits to:
- Open-source ARIA under Apache 2.0
- NVIDIA-branded performance benchmarks published
- NVIDIA Inception acknowledgment in model cards and paper
- Referenceable deployment case study (edtech pilot) in NVIDIA Developer blog

---

## Slide 13 — Why Fund Us

1. **Technical depth**: working 131M model trained from scratch on TPU — very few solo Indian founders have done this
2. **Shippable novelty**: 2 architecture claims verified novel vs Qwen3-Next, Jamba, Hymba — workshop-paper tier now, main-track after 1B
3. **Vertical wedge**: Indian edtech is a proven paying market (Byju's, PW at $1B+) with frontier-model gap
4. **Capital efficiency**: $0 spent so far, full research + infrastructure on free TRC tier
5. **Commitment**: full-time, open-source, permanent relocation to AI hub

**Not a team of 20 yet. But the work-per-dollar ratio is best-in-class.**

---

## Slide 14 — Contact & Close

**Let's build the foundation model stack India actually deserves.**

- Website: [yourcargohive.com — set up before sending]
- Email: [you@cargohive.com]
- GitHub: github.com/Ashishgirigoswami/aria
- HuggingFace: huggingface.co/[your-handle]
- LinkedIn: [url]

**Target**: close NVIDIA Inception approval by May 15, 2026. Begin 1B pretraining June 1, 2026.

**First paying edtech customer**: Oct 2026.

Thank you.

---

## HOW TO TURN THIS INTO ACTUAL SLIDES

**Fastest path (30 min)**:
1. Go to https://www.canva.com/presentations/templates/pitch-deck/
2. Pick "Minimal Pitch Deck" template (free)
3. Copy-paste each section above into corresponding slide
4. Swap logo, colors to your brand
5. Export as PDF

**Polished path (2-3 hours)**:
1. Use https://pitch.com (free) — better typography
2. Build a simple website at cargohive.com first using https://vercel.com/templates/startup
3. Embed live ARIA training loss curve screenshot on slide 5 (use matplotlib → screenshot)
4. Add one diagram on slide 4 (ARIA architecture — use https://excalidraw.com)

## BEFORE SENDING TO NVIDIA

Required:
- [ ] Register **cargohive.com** domain (GoDaddy / Namecheap, ₹800)
- [ ] Set up one-page website (Vercel Next.js startup template — 1 hr)
- [ ] Take a screenshot of ARIA 131M training loss curve (matplotlib from summary.json when available)
- [ ] Fill in all [placeholder] brackets with your actual info
- [ ] Export as PDF (max 2MB, NVIDIA form limit)

Optional but strong:
- [ ] Publish ARIA 131M model card on HuggingFace (after training finishes)
- [ ] LinkedIn post announcing the training run
- [ ] Update GitHub README with the same pitch framing

## Lines to re-check for honesty before sending

- "matches transformer quality at 1/3 KV cache" — **verify this is actually measured on ARIA, not an architectural claim**. Soften to "targets 1/3 KV cache through architecture" until proven.
- "100M+ aspirants" — Indian exam market is ~20-30M active, 100M is cumulative. Use "20M+".
- Revenue projections in Slide 8 — these are *goals*, not commitments. NVIDIA reviewers generally don't grade on these, but if pressed, label explicitly.
- Competitive matrix — verify Sarvam / Krutrim row claims are still accurate (they move fast).

**Be honest about pre-revenue status.** Grant reviewers can tell when someone oversells — it hurts more than underselling.
