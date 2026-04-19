# AI Buildables — April 2026 survey

> Researched for CargoHive / Ashish Giri Goswami. Cutoff: April 2026.
> Bias toward: solo/small-team feasible, revenue-generating OR credibility-building, 3-12 month horizon.
> Deduped against `D:/mytllm/aria/docs/RESEARCH_KB_*.md`, `D:/mytllm/research/kb/*.md`, `D:/mytllm/research/FRONTIER_RESEARCH_SWEEP_APR2026.md`, `D:/mytllm/research/ARIA_Frontier_v2.md`.
>
> Rule: every numeric claim has a source URL. "Unverified" appears where the primary source isn't confirmed.

---

## 1. Research directions still wide open

These are things a 1-2 person team with ARIA's existing code, free TPU hours, and a consumer GPU can credibly attack in 3-9 months. All are "not yet in the KB as explored lines."

### 1.1 In-Place Test-Time Training (IP-TTT) for small hybrid models

- **Idea.** Repurpose MLP projection matrices as fast weights updated in-place during inference — no architectural surgery, preserves pre-trained weights. The original ICLR 2026 result shows a 4B Transformer handling 128k context with in-place updates.
- **Why it matters.** ARIA already has a gated DeltaNet-style recurrence. Coupling IP-TTT on the attention-block MLPs while the recurrent branch handles long-range gives you test-time adaptation on top of a hybrid — nobody has published that combo.
- **Who is close.** Original IP-TTT paper (Stanford/NVIDIA collaborators), TTT-E2E (35x faster than full attention at 2M context). None have tested on a GDN-hybrid.
- **Effort.** 2-3 weeks of research code on top of existing ARIA 131M. Can be smoke-tested on a single GTX 1050 at short context.
- **Reference.** https://arxiv.org/html/2604.06169 (In-Place Test-Time Training), https://openreview.net/pdf?id=dTWfCLSoyl (ICLR 2026).
- **Workshop-paper potential: HIGH.** Novel combination, small-scale-reproducible, differentiates against Qwen3-Next.

### 1.2 JustRL applied to sub-200M hybrid SSM models

- **Idea.** JustRL (ICLR 2026 Blogpost Track) shows a *fixed-hyperparameter* GRPO recipe on binary verifier rewards reaches SOTA on 1.5B math benchmarks with 2x less compute than sophisticated recipes. No curriculum, no temperature scheduling, no mid-training resets.
- **Why it matters.** This is the first credibly-cheap RL-for-reasoning recipe. Applying it to a 131M hybrid is <100 T4-hours (Kaggle free tier) and would be the smallest JustRL reproduction ever published.
- **Who is close.** THUNLP's JustRL release; the state-of-RL-reasoning literature emphasizes "simple RL, good base model." Nobody has reproduced at <1B or on a hybrid.
- **Effort.** ~4 weeks once the 131M SFT checkpoint exists. Compute: ~60 Kaggle T4-hours.
- **Reference.** https://github.com/thunlp/JustRL , https://arxiv.org/pdf/2512.16649 .
- **Workshop-paper potential: HIGH.** Ties directly into the existing ARIA KB cycle 7.

### 1.3 Mamba-3 MIMO swap inside ARIA's LSA stack

- **Idea.** Mamba-3 (ICLR 2026) introduces a MIMO SSM formulation, complex-valued state update, and exponential-trapezoidal discretization. At 1.5B it's +1.8 pts over Gated DeltaNet with half the state size. ARIA's recurrent branch is currently a GDN-style WY chunk; swapping in Mamba-3's MIMO block is a drop-in.
- **Why it matters.** Direct empirical test of whether Mamba-3's 1.5B gains hold at 131M AND whether they stack on top of the ARIA joint-softmax + shared-MLA-latent mechanisms.
- **Who is close.** Published March 2026. No one has tested it at sub-500M scale yet, and no one has combined it with full-attention hybrids reported publicly.
- **Effort.** 2-3 weeks of porting + 1 full pretraining run. Fits the existing Phase-2 plan.
- **Reference.** https://arxiv.org/abs/2603.15569 , https://openreview.net/pdf?id=HwCvaJOiCj .
- **Workshop-paper potential: MEDIUM-HIGH** (your 131M ablation would be one of the earliest third-party Mamba-3 replications).

### 1.4 Nested Learning "Hope" block for continual pretraining

- **Idea.** Google's Nested Learning framework (Dec 2025) treats the model as a hierarchy of nested optimization problems. The "Hope" sequence module (self-modifying + continuum memory) was proposed for continual learning and long-context reasoning.
- **Why it matters.** Nobody has reproduced Hope at <1B; the framework claims it beats Mamba/Transformer on continual learning tasks. A small-scale reproduction on FineWeb-Edu would be a high-signal validation.
- **Risk.** Paper is philosophical as much as engineering; the practical recipe is not fully spelled out.
- **Effort.** 4-6 weeks research, uncertain payoff.
- **Reference.** https://arxiv.org/abs/2512.24695 , https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/ .
- **Workshop-paper potential: MEDIUM.** High reward, high variance.

### 1.5 TurboQuant for KV-cache compression on the attention-25% layers of ARIA

- **Idea.** Google's ICLR 2026 TurboQuant (PolarQuant rotation + Quantized Johnson-Lindenstrauss) reduces KV-cache memory overhead substantially for long-context serving.
- **Why it matters.** ARIA is 75% linear / 25% attention. The attention layers own the KV cache. Applying TurboQuant only to those layers is a free 2-4x context extension at serving time, and a cheap paper angle ("TurboQuant for Hybrids").
- **Effort.** 2 weeks of kernel porting + eval.
- **Reference.** https://www.sciencedaily.com/releases/2026/04/260405003952.htm (public writeup; ICLR 2026 paper full text unverified).
- **Workshop-paper potential: MEDIUM.** Systems workshop (MLSys 2026).

### 1.6 SYNTHETIC-1 + RLVR pipeline for a 131M-class reasoner

- **Idea.** SYNTHETIC-1 is an open 1.4M-task verified-reasoning corpus (math/code/science) from Prime Intellect — fully DeepSeek-R1-distilled. The "From Self-Evolving Synthetic Data to Verifiable-Reward RL" paper (arXiv 2601.22607) closes the loop: generate, verify, RL. Neither has been applied to sub-500M hybrids.
- **Why it matters.** You get a frontier-style SFT+RL pipeline with no teacher-model calls (dataset is pre-generated) and no reward model.
- **Effort.** ~3 weeks; can run on Kaggle T4 + free Colab.
- **Reference.** https://www.primeintellect.ai/blog/synthetic-1 , https://arxiv.org/abs/2601.22607 .
- **Workshop-paper potential: HIGH.** Combines two fresh threads into a reproducible recipe.

### 1.7 Indian-language reasoning benchmark + a small Indic-specialist model

- **Idea.** IndicMMLU-Pro covers 9 Indic languages with accuracy ranges 9.9%-44.8%; IndicDB is a text-to-SQL benchmark in 7 variants including Hinglish; IndQA and IndicVisionBench are newer. But there is NO open reasoning trace / GSM8K-equivalent for Indic languages with published verified solutions.
- **Why it matters.** Sarvam-M and Granite 4.0 beat baselines on general Indic benchmarks but the reasoning dimension is wide open. A single-person team can (a) translate GSM8K/MATH-500 into Hindi/Tamil/Bengali with a strong teacher, (b) verify numerically, (c) publish as a HF benchmark, (d) train ARIA on it.
- **Effort.** 3-4 weeks, ~$300-500 in Claude API or free-tier Gemini.
- **Reference.** https://arxiv.org/abs/2501.15747 (IndicMMLU-Pro), https://arxiv.org/html/2604.13686 (IndicDB), https://www.ibm.com/think/news/indqa-llm-benchmark (IndQA).
- **Workshop-paper potential: HIGH.** Benchmark papers compound into credibility.

### 1.8 BLT-style byte-patching on Indic text

- **Idea.** Meta's Byte Latent Transformer matches Llama 3 with up to 50% fewer inference FLOPs. Tokenization is the single biggest cost for Indic languages (Sarvam 1 demonstrated 80% fewer tokens on Tamil vs Llama 3.3). BLT's dynamic entropy-based byte patching is a natural fit; nobody has measured it on Devanagari/Dravidian scripts at scale.
- **Why it matters.** Direct leverage on the existing "Indic tokenization is broken" pain point. Could become the de facto Indic tokenizer-free architecture.
- **Effort.** 6-8 weeks. Requires re-architecting ARIA's embedding layer and re-pretraining.
- **Reference.** https://arxiv.org/abs/2412.09871 , https://github.com/facebookresearch/blt , https://medium.com/@manick2411/benchmarking-llm-tokenization-indic-languages-under-the-lens-6b86844e54e9 .
- **Workshop-paper potential: HIGH.** Crosses with product story (see §2.2).

### 1.9 Test-Time A* Search for small-model math (TTA*)

- **Idea.** Test-Time A* (TTA*, ICLR 2026) lets 1.5B models match 32B-70B on math benchmarks with 10x less memory by framing reasoning as goal-directed search over partial solutions.
- **Why it matters.** It's cheap TTS compute (no training) and directly applicable to ARIA 131M post-SFT. Great demo for a pitch: "our 131M matches a 1.5B on GSM8K at inference" is visceral.
- **Effort.** 2 weeks of integration + eval.
- **Reference.** https://openreview.net/forum?id=eJ1yDj6vtH .
- **Workshop-paper potential: MEDIUM** (recipe paper).

### 1.10 muP for hybrid GDN+attention stacks (closing the KB gap)

- **Idea.** The KB already flags muP as the highest-leverage saving (skip the LR sweep at 1B). But no published muP derivation exists specifically for GDN + full attention interleave with joint softmax. That is the open gap.
- **Effort.** Theoretical + small-scale sweep, 3-5 weeks.
- **Reference.** ARIA KB cycle 4 (`RESEARCH_KB_2026-04-17_mup.md`). Public baseline: https://magazine.sebastianraschka.com/p/state-of-llms-2025 .
- **Workshop-paper potential: HIGH.** Sits in a clear gap, and the output is directly useful to every subsequent ARIA training run.

---

## 2. Shippable products for the Indian market

Products that a solo founder can launch in under 6 months, with a plausible first-10-customer path. All market sizes labeled with source; if not verifiable, flagged.

### 2.1 Agentic GST filing + reconciliation for Tier-2/3 SMBs

- **Product.** Connect to Tally/Zoho Books + bank statements + invoice mailbox. An agent categorizes transactions, computes GST liability, reconciles ITC, files GSTR-1/3B, answers notices via natural language (Hindi + English).
- **Target.** 63M+ Indian MSMEs, of which the 6-10M with annual turnover Rs 40L-2Cr feel the GST pain most acutely. (MSME count: https://msme.gov.in/ , *unverified segmentation*.)
- **TAM signal.** One market report pegs automated-GST-invoicing at USD 10B with 20% CAGR ("unverified" — aggregator source: https://startup.ai/startup-idea/ai-gst-invoicing-sui2je ). Agentic-AI-fintech predicted 10x operational cost reduction for early movers — https://aidevdayindia.org/blogs/the-state-of-agentic-ai-in-india-2026/agentic-ai-fintech-applications-india.html .
- **Competitive landscape.** Invoiso, LEDGERS.Cloud, Tally-at-Cloud already have the "AI billing" label but are mostly chat-on-top-of-forms. No one has a true agent that *files and responds to notices* end-to-end.
- **MVP build.** 10-12 weeks: GSTN sandbox integration → Tally connector → a fine-tuned 7B with GST-notice SFT data → form-filling skill → human-in-the-loop approval UI. Charge Rs 999/month/GSTIN.
- **Source.** https://invoiso.ai/automated-gst-billing-software/ , https://ledgers.cloud/ .
- **Feasibility score: HIGH.**

### 2.2 Hindi / Hinglish voice agent for D2C return + COD confirmations

- **Product.** Outbound voice agent that calls COD buyers to confirm, handles returns, schedules deliveries, upsells. Hindi-first, code-switches with English, sub-350ms latency, Rs 1-2 per minute.
- **Target.** The fast-growing D2C brands in Tier-3 India that lose 25-40% of COD orders to refusal. Primary buyers: D2C operators on Shopify/Meesho/WooCommerce.
- **TAM signal.** Indian Voice-AI market $153M (2024) → $957M (2030) at 35.7% CAGR per market reports — https://www.ringg.ai/blogs/best-voice-ai-agents-for-indian-languages (aggregator, *unverified primary*).
- **Competitive landscape.** Bolna, Ringg, Haptik, Lumay, Tabbly, VoiceGenie, Squadstack, CarmaOne are all chasing this. Room exists for a *developer-first API priced like Twilio* rather than an enterprise contract. Also: none target D2C return workflows specifically; most are generic.
- **MVP build.** 8-10 weeks: Sarvam-TTS or Indic-Parler → a 7B instruction-tuned on code-switched telephony transcripts → Twilio/Exotel bridge → Shopify/Meesho app listing. Charge Rs 1.5/min + setup.
- **Source.** https://www.bolna.ai/ , https://www.haptik.ai/blog/voice-ai-agents-for-indian-languages .
- **Feasibility score: HIGH, but crowded.**

### 2.3 Exam-prep reasoning tutor for NEET/JEE/UPSC (leveraging the AIforAspirants context)

- **Product.** Subject-specific tutor that does step-by-step verified solutions, adaptive question generation, and tracks mastery. Core moat: *small Indic-reasoning model* fine-tuned on verified JEE/NEET/UPSC solutions, using the §1.7 Indic benchmark you'd publish.
- **Target.** 30M+ students prepping for these exams annually — https://www.imarcgroup.com/insight/top-factors-driving-growth-india-edtech-industry .
- **TAM signal.** India EdTech: $2.8B (2024) → $33.2B (2033), CAGR 28.7% — https://www.imarcgroup.com/insight/top-factors-driving-growth-india-edtech-industry . PhysicsWallah alone posted Rs 1,082Cr revenue in Q3 FY26 — https://www.mgiwebzone.com/indian-education-industry-marketing-statistics-report/ .
- **Competitive landscape.** PhysicsWallah, Byju's (distressed), Unacademy, Vedantu, Career Launcher. All are content + live-class brands. AI-native "unlimited tutor at Rs 299/month" is still a white space — nobody has the combination of (a) verified reasoning, (b) regional-language natively, (c) mobile-first, (d) consumer pricing.
- **MVP build.** 10-14 weeks: ARIA-131M SFT on NCERT + NEET previous year papers → TTA* at inference for stepped solutions → WhatsApp bot interface → UPI subscriptions (Rs 149-299/mo).
- **Source.** https://www.imarcgroup.com/insight/top-factors-driving-growth-india-edtech-industry , https://www.ncnonline.net/the-india-edtech-market-is-set-reach-new-heights-in-2026-with-ai-dominating-the-solutions/ .
- **Feasibility score: HIGH.** Best overlap with user's existing AIforAspirants context.

### 2.4 Vertical legal agent for Indian startup incorporation + compliance

- **Product.** Agent that does founder incorporation, ESOP drafting, MSME registration, ROC filings, LLP→PvtLtd conversion. Charges per transaction (Rs 3-15k), much cheaper than a CA/CS.
- **Target.** The 100k+ startups incorporated annually per DPIIT's Startup India registry (*unverified*).
- **Signal.** YC W26 batch has 6+ legal-tech verticals (Wayco, Arcline, General Legal, Legalos, Vector Legal) — clear signal that vertical legal agents work globally — https://www.extruct.ai/research/ycw26/ . Indian analogs: Vakilsearch and LegalKart exist but are heavy-workflow sites, not autonomous agents.
- **Moat.** India-specific regulation (MCA forms, GST law, RBI FEMA) changes enough that a foreign player can't just port. Hindi/regional-language support doubles defensibility.
- **MVP build.** 12-16 weeks: MCA v3 form automation + DIN/DSC flows + PDF extraction + a 7B legal-SFT model. Critical: partner with one or two licensed CS firms for human-in-loop on anything binding.
- **Source.** https://www.extruct.ai/research/ycw26/ .
- **Feasibility score: MEDIUM-HIGH.** Regulatory complexity is the moat but also the blocker for a solo operator — partnership needed.

### 2.5 B2B API: "Sarvam-class Indic reasoning at 1/10th price"

- **Product.** A hosted API serving a fine-tuned ARIA model specialized for Indic reasoning tasks (GST Q&A, education, legal, healthcare discharge summaries). Position as cheaper alternative to Sarvam and Krutrim at the entry tier.
- **Target.** Indian SaaS companies that need Indic-language AI features but can't justify Sarvam-105B pricing.
- **TAM signal.** Sarvam announced 30B and 105B-A9B models in Feb 2026; is raising $300-350M at $1.5B valuation — https://www.bloomberg.com/news/articles/2026-04-02/india-ai-startup-sarvam-raises-funds-at-1-5-billion-valuation , https://en.wikipedia.org/wiki/Sarvam_AI . Price umbrella exists.
- **Competitive landscape.** Sarvam, Krutrim, Ola-Krutrim Cloud, AI4Bharat (academic), Pragna (academic). All are either expensive or non-commercial. No commercial sub-$0.5/M-token Indic API.
- **MVP build.** Fine-tune a 1B ARIA on Indic reasoning → deploy on Runpod/Vast/Modal → simple OpenAI-compatible endpoint. Charge $0.10 / $0.30 per 1M input/output tokens.
- **Source.** https://www.businessworld.in/article/sarvam-ai-350-million-funding-bessemer-nvidia-amazon-2026-600618 , https://restofworld.org/2026/india-frugal-ai-sarvam-krutrim-sovereign/ .
- **Feasibility score: MEDIUM.** Depends on ARIA 1B actually landing a competitive perplexity — high upside if it does.

### 2.6 "AI compliance officer" for Indian fintech startups

- **Product.** Monitors app store flows, UPI transactions, RBI circulars, and generates compliance reports + automatic patch recommendations. Aimed at the ~500 Indian neobanks and PAs/PAs regulated by RBI.
- **Target.** 500-2000 registered fintech entities under RBI regulations (*unverified count*).
- **TAM signal.** Indian fintech raised ~$2.7B in 2025 (*unverified*). Agentic fintech applications trend — https://aidevdayindia.org/blogs/the-state-of-agentic-ai-in-india-2026/agentic-ai-fintech-applications-india.html .
- **Competitive landscape.** Compliance is dominated by consulting firms (Grant Thornton, PwC India). No SaaS/agent exists.
- **MVP build.** 12-16 weeks; sell to design partner fintechs at Rs 50k-2L/month.
- **Feasibility score: MEDIUM.** Regulatory risk is real; needs ex-fintech-compliance co-founder.

### 2.7 WhatsApp-native AI micro-SaaS for Hindi/regional creators

- **Product.** WhatsApp bot for Hindi YouTube creators: scriptwriting, B-roll suggestions, comment triage, thumbnail A/B testing, sponsor CRM. Pay Rs 299/mo via UPI autopay.
- **Target.** India creator economy: $15.03B in 2026 → $61.87B by 2033; regional language ad spend at 35% of total digital — https://www.coherentmarketinsights.com/industry-reports/india-creator-economy-market (*unverified primary*).
- **TAM signal.** 1M+ YouTube channels used the platform's AI creation tools daily in Dec 2025 — https://www.thewrap.com/industry-news/tech/youtube-2026-goals-ai-monetization-neal-mohan-letter/ . Hindi CPM Rs 80-250; top channels outsized earnings.
- **Competitive landscape.** Crayo, Opus Clip, InVideo are global. No Hindi-first WhatsApp-native tool.
- **MVP build.** 4-6 weeks; WhatsApp Cloud API + Sarvam/ARIA backend + UPI autopay via Razorpay.
- **Source.** https://upgrowth.in/youtube-cpm-india-guide-2026/ , https://www.truefan.ai/blogs/faceless-youtube-earnings-india-2026 .
- **Feasibility score: HIGH** for quick-revenue, LOW for defensibility.

### 2.8 Healthcare discharge-summary + prior-auth agent for Tier-2 hospitals

- **Product.** Reads handwritten OPD notes + lab PDFs, produces structured discharge summaries, handles insurance prior-authorization. Hindi + English, hospital-EMR integrations.
- **Target.** ~69,000 hospitals in India; ~7,000 private multi-specialty (*unverified*).
- **TAM signal.** Qure.ai raised $123M Series D and is IPO-bound in 2026 — https://aifundingtracker.com/top-ai-startups-india/ . Vertical healthcare AI is 25%+ of Indian AI funding.
- **Competitive landscape.** Qure.ai is imaging-only; MFine, Practo are chat/teleconsult. Discharge-summary workflow is unclaimed by AI-native players.
- **MVP build.** 16-20 weeks due to HIPAA-equivalent (DISHA) compliance and EMR integrations. Not solo-feasible without a doctor co-founder.
- **Feasibility score: LOW (solo).** Flagged as *wait* until team grows.

### 2.9 "Sovereign-data" trained LLM for Indian government + PSU tenders

- **Product.** A transparent, fully-Indian-hosted (AWS Mumbai or Yotta), Apache-2-weighted LLM, positioned for Section 65B IT Rules, DPDP Act, and CAG/CERT-In compliant deployment. Sold through GeM (Government e-Marketplace) tenders.
- **Signal.** IndiaAI Mission is funding 12 indigenous-foundational-model orgs (incl. Sarvam) with Rs 246Cr+ in compute/cash — https://obcrights.org/blog/central-government-scheme/india-ai-mission-objectives-startups-research-centers-government-strategy-explained/ . Government tenders for Indic LLMs have started appearing.
- **Feasibility score: MEDIUM-LOW as a direct product**, but the ARIA HF release positions CargoHive as *a candidate* for future cohorts of the IndiaAI Mission — use this angle in grant applications, not as primary product.
- **Source.** https://www.pib.gov.in/PressReleasePage.aspx?PRID=2231169&reg=3&lang=1 .

### 2.10 Open-source DIY RAG for Indian SMB knowledge bases (free tier → paid)

- **Product.** Self-hostable RAG stack that plugs into WhatsApp Business, Razorpay, Shopify Bharat, KiranaKart. Free for <100 docs; Rs 499/mo for unlimited. Built on ARIA-131M + bge-m3 embeddings + Qdrant.
- **Target.** 100k+ Indian SMBs already on WhatsApp Business.
- **Competitive landscape.** Gupshup and Haptik dominate enterprise; no self-serve-free-tier offering exists for the SMB long-tail.
- **MVP build.** 6-8 weeks.
- **Feasibility score: HIGH** as a wedge into §2.2 / §2.7.

---

## 3. High-leverage open-source projects

Each would establish credibility for NVIDIA Inception / VC pitch AND plant seeds for later commercial work.

### 3.1 `aria-reasoner-131m` — JustRL+RLVR-trained reasoning checkpoint

- **What it demonstrates.** Smallest credible RL-for-reasoning checkpoint (you'd hold a scale record, at least transiently).
- **Effort.** 3-4 weeks after SFT on SmolTalk/SYNTHETIC-1.
- **Hiring / funding hook.** Every grant reviewer understands GSM8K numbers.
- **Reference.** https://github.com/thunlp/JustRL .

### 3.2 `indic-reasoning-bench` — GSM8K-MATH verified in Hindi/Tamil/Bengali

- **What it demonstrates.** You can curate and publish benchmarks (a rare, high-signal skill). Becomes the de-facto Indic reasoning benchmark if timed right.
- **Effort.** 3-4 weeks, ~$300-500 in teacher-model API costs.
- **Hook.** Cited by Sarvam, AI4Bharat, Krutrim within 6 months → instant credibility.
- **Reference.** https://arxiv.org/abs/2501.15747 (IndicMMLU-Pro as template).

### 3.3 `tpu-wsd-trainer` — reference WSD + muP trainer for TPU-TRC users

- **What it demonstrates.** Frontier-recipe fluency; fills a real gap (most WSD code is GPU-only).
- **Effort.** 2 weeks to extract+clean from existing ARIA trainer.
- **Hook.** Google TPU Research Cloud team is known to notice and amplify high-quality TRC-era releases.
- **Reference.** https://magazine.sebastianraschka.com/p/state-of-llms-2025 (WSD context).

### 3.4 `blt-indic` — Byte Latent Transformer trained on Devanagari/Dravidian scripts

- **What it demonstrates.** Handles the Indic tokenization problem head-on; naturally extends Meta's BLT release.
- **Effort.** 6-8 weeks; significant pretraining needed.
- **Hook.** Obvious talking point for Sarvam/Krutrim and for India-focused VCs.
- **Reference.** https://github.com/facebookresearch/blt .

### 3.5 `mini-swe-agent-indic` — small-model coding agent scored on SWE-Bench Verified

- **What it demonstrates.** Frontier agent-scaffolding skill.
- **Effort.** 4-6 weeks; SWE-Bench harness already open.
- **Hook.** SWE-Bench Verified at 80.9% (Claude Opus 4.5) vs smaller open models shows the gap — a credible sub-10B, sub-40% entry would stand out. https://www.swebench.com/ .

### 3.6 `ip-ttt-hybrid` — IP-TTT reference implementation for hybrid SSM+Attention

- **What it demonstrates.** First IP-TTT port beyond dense transformers.
- **Effort.** 2-3 weeks.
- **Hook.** Direct paper lead for §1.1.

### 3.7 `farseer-tpu` — Farseer-scaling-law reference on TPU-v2-8

- **What it demonstrates.** Rigorous scaling-law fluency; Farseer claims 433% lower extrapolation error than Chinchilla (sweep context from ARIA Frontier KB).
- **Effort.** 3 weeks (sweeps + plots).
- **Hook.** Academically useful; gets citations.
- **Reference.** ARIA Frontier sweep (`D:/mytllm/research/FRONTIER_RESEARCH_SWEEP_APR2026.md`).

### 3.8 `vllm-indic-kernels` — FlashAttention kernels tuned for Devanagari-heavy vocabularies

- **What it demonstrates.** Systems-level contribution to a project (vLLM) with 5-figure stars and enterprise adoption.
- **Effort.** 6-10 weeks, requires CUDA. If user doesn't own a non-1050 GPU, push to phase 3 post-grant.
- **Hook.** Merge to vLLM main = instant resume line.
- **Reference.** https://github.com/vllm-project/vllm .

---

## 4. Technical moats small teams can actually build

Frontier labs have compute, data scale, and brand. A solo Indian founder cannot outspend them. What you can outmaneuver on:

**A. Indic-specific data.** Sarvam/Krutrim are funded to do this at high end but neither is focused on (i) small-model efficiency, (ii) verified reasoning traces, (iii) under-represented languages like Santhali, Bodo, Konkani. A curated 1-10B-token verified corpus in Indic languages is a real moat because it takes 6 months of human-in-loop work that VCs find hard to rush.

**B. Regulatory/distribution moats via India-specific rails.** UPI autopay, ONDC merchant onboarding, GSTN API access, MCA v3, RBI Account Aggregator — these are all rails that non-Indian founders struggle to integrate because they need an Indian entity. CargoHive has that. A product like §2.1 has a real distribution moat because the integration is expensive.

**C. Vertical + hybrid architecture (true innovation, not GPT-wrapper).** ARIA's shared-MLA-latent + joint-softmax over a GDN+attention hybrid is genuinely novel vs Qwen3-Next; if the 1B scale-up holds, it is a paper-worthy architecture. "We train our own models" is a credible moat in 2026 because it's hard and most competitors don't. But only *if* the architecture pays off empirically.

**D. Speed of iteration.** A 2-person team can ship weekly; Sarvam cannot. Pick a narrow vertical, ship a weekly changelog publicly, build an audience on Twitter/X + Hacker News. This is an *execution* moat.

**E. Open-source community.** Repos with 1k+ stars become customer acquisition engines. Plan releases 3.1-3.8 as a deliberate credibility ladder.

**F. Geographic distribution.** GeM, IndiaAI Mission, State-government partnerships (MeghalayaAI, TamilNaduAI) are real channels. Each one is hard for a US startup to win because of the procurement-in-local-entity requirement.

**What does NOT work as a moat:**
- Prompt engineering over OpenAI/Anthropic APIs (zero defensibility; gross margin collapses the moment the underlying model drops price).
- Fine-tuning 7B Llama on a public dataset (takes 2 weeks; anyone can copy).
- A chat UI (commodity).

---

## 5. Don't build (commoditized or dead ends)

- **General-purpose Hindi chatbot.** Sarvam, Krutrim, Google Gemini (via Gemma 4), and OpenAI all have strong enough Hindi/Tamil/Bengali — https://www.ibtimes.com.au/indias-top-10-ai-companies-2026-sarvam-ai-krutrim-lead-sovereign-push-amid-29b-funding-boom-1866020 . Consumer chatbot is a brand/distribution game; unwinnable solo.
- **ChatGPT for X where X is sales or customer-support.** 10+ YC-funded players in each vertical globally — https://www.extruct.ai/research/ycw26/ . Indian analog market is equally crowded (Haptik, Ringg, Verloop, SquadStack, Yellow.ai, Exotel).
- **Another coding copilot.** Cursor/Claude Code/GitHub Copilot/Zed/Windsurf own this; leaders sit at 80%+ on SWE-Bench Verified — https://www.swebench.com/ . Rebuilding is capital arbitrage and you don't have the capital.
- **Pure-SSM architecture experiments at <1B.** ARIA context already acknowledges this; Mamba-3 is state-of-the-art but the ceiling without hybridization is known to underperform. ARIA's hybrid is right; pure SSM is not a research direction with residual value.
- **General-purpose "India LLM."** Sarvam-105B, Krutrim, Granite 4.0 all serve this. A pure "we trained on Indic data" foundation model from a 2-person team will never compete on benchmarks. Instead, ARIA should be a *specialist* model positioned on an axis the giants are not focused on (reasoning efficiency, or long-context hybrid, or Indic-reasoning).
- **Foundation-model training from scratch without $10M+ / 10M+ training FLOPs.** ARIA 131M/1B are fine as research + niche product backbones, but do not rebrand them as "frontier." Positioning matters.
- **Generic RAG-as-a-service.** LangChain, LlamaIndex, Haystack own the mindshare; pure RAG hosted offering has low gross margin. Only worth building as §2.10 (as a wedge into a vertical SaaS).
- **Reward-model training for Indic RLHF.** RLVR (no reward model) is the dominant paradigm in 2026 (GRPO + verifiers); training custom reward models is mostly dead research unless you have a very specific verifier gap.
- **FP4/FP8 kernels without B200/H200 access.** ARIA KB cycle 8 already flagged this; TPU v2-v4 lacks FP8 MXUs; GTX 1050 has no Tensor Cores at all. Revisit after hardware upgrade.
- **Text-to-SQL-as-a-product.** Commoditized; GPT-5, Claude, and Gemini solve it acceptably; IndicDB benchmark shows the gap but the product is being eaten — https://arxiv.org/html/2604.13686 .
- **Generic multilingual translation API.** Google Translate + Sarvam + Azure own this; consumer-grade is free; enterprise API is a wafer-thin margin business.

---

## 6. Recommended sequencing for CargoHive

Given: ARIA 160M training on FineWeb-Edu (TRC), GTX 1050 local, NVIDIA Inception pitch imminent, solo founder, India-based. The next 12 months should compound credibility into revenue, in that order — because credibility unlocks grants/compute, and compute unlocks the ARIA story that unlocks revenue.

### Months 0-2 (now → June 2026): Finish ARIA baseline + publish + apply

1. **Finish ARIA 160M training on FineWeb-Edu** (already in flight — per `RESEARCH_KB_INDEX.md` Phase 1).
2. **Publish on HuggingFace Hub** — weights, eval JSON, training logs, two small blog posts (muP derivation + WSD schedule). Establishes you as a "trains-from-scratch" builder, not a wrapper.
3. **Submit NVIDIA Inception application.** With ARIA on HF, the application is strong. Benefits: DLI training, cloud partner credits up to $100k AWS + $150k Nebius, Inception Capital Connect intro list. Source: https://www.thundercompute.com/blog/nvidia-inception-program-guide .
4. **Apply simultaneously to:** Lambda Labs research credits, AWS Activate ($100k), Google Cloud Startup ($200k over 2 years), HuggingFace Community Compute, AMD Instinct Cloud credits. These are parallel — none block the others.
5. **Draft the Indic-reasoning benchmark spec** (§3.2) so compute, when it arrives, has a destination.

### Months 2-4 (June → August 2026): Ship the first revenue wedge + first research paper

Pick ONE of the following two tracks based on which grants actually close:

- **Track A — revenue-first (if Inception + AWS credits land but compute is still bounded):** Ship §2.7 (WhatsApp creator micro-SaaS) OR §2.10 (DIY RAG SMB tier). Both are 4-8-week builds, low compute, UPI-monetizable. Target Rs 1-2L MRR by month 4.
- **Track B — credibility-first (if a compute-heavy grant like Lambda H100 closes):** Execute §1.2 + §1.3 (JustRL RL on a Mamba-3-swapped ARIA 131M). Submit to a NeurIPS 2026 workshop (deadline typically early September). This becomes the centerpiece of the VC pitch.

Recommended: **Track A in parallel with §1.1 (IP-TTT) on the local GTX 1050**, because §1.1 is low-compute research that doesn't require grants.

### Months 4-7 (August → November 2026): Second paper + first enterprise pilot

1. **If Track A worked:** Use the MRR + a user base to graduate into §2.3 (exam-prep tutor using Indic-reasoning-bench). This is the first product that uses ARIA meaningfully.
2. **Second research paper:** ship §1.7 (Indic-reasoning benchmark + small specialist) — workshop or direct ACL Rolling Review.
3. **First enterprise pilot:** cold-email 20 D2C brands, 20 SMB accounting firms, 20 exam-prep academies. Close 1-2 for Rs 25-50k/month.

### Months 7-12 (November 2026 → April 2027): Scale the wedge that worked

- Decide revenue vs research weighting based on what stuck.
- If exam-prep tutor has 500+ paying users: raise pre-seed at Rs 8-15Cr valuation for scaling compute + one-hire co-founder.
- If ARIA research got strong citation signal: pitch research-lab-in-India grants (OpenAI, Anthropic, Mila have these), and consider going non-profit-research + consulting model.

### Anti-pattern warning

Avoid spreading across all 10 product ideas in §2. The user has shipped ARIA — that's a rare moat. The second product must *use* ARIA, not be a stand-alone CRUD app. Prioritize products (§2.3, §2.5, §2.1) where the model itself is the product, over apps (§2.7, §2.10) where ARIA is optional.

### NVIDIA Inception pitch angles

- Concrete ARIA HF numbers (once published).
- ARIA's architectural novelty (shared-MLA + joint-softmax) — papers cite-able.
- India-specific distribution: UPI autopay, ONDC, WhatsApp Business.
- Ask for: DGX Cloud credits, access to B200 for Mamba-3-MIMO experiments, Inception Capital Connect intros to India-focused VCs (Accel, Peak XV, Lightspeed India).

---

## Sources

- https://llm-stats.com/ai-news
- https://llm-stats.com/llm-updates
- https://www.sciencedaily.com/releases/2026/04/260405003952.htm
- https://magazine.sebastianraschka.com/p/state-of-llms-2025
- https://labs.adaline.ai/p/the-ai-research-landscape-in-2026
- https://www.clarifai.com/blog/llms-and-ai-trends
- https://www.ycombinator.com/companies?batch=Winter+2026
- https://www.extruct.ai/research/ycw26/
- https://forgeup.in/news/startup-news/ai-startups-india-2026-sovereign-llms-funding/
- https://aifundingtracker.com/top-ai-startups-india/
- https://www.ibtimes.com.au/indias-top-10-ai-companies-2026-sarvam-ai-krutrim-lead-sovereign-push-amid-29b-funding-boom-1866020
- https://restofworld.org/2026/india-frugal-ai-sarvam-krutrim-sovereign/
- https://www.businessworld.in/article/sarvam-ai-350-million-funding-bessemer-nvidia-amazon-2026-600618
- https://www.bloomberg.com/news/articles/2026-04-02/india-ai-startup-sarvam-raises-funds-at-1-5-billion-valuation
- https://en.wikipedia.org/wiki/Sarvam_AI
- https://www.pib.gov.in/PressReleasePage.aspx?PRID=2231169&reg=3&lang=1
- https://obcrights.org/blog/central-government-scheme/india-ai-mission-objectives-startups-research-centers-government-strategy-explained/
- https://arxiv.org/abs/2501.15747
- https://arxiv.org/html/2404.16816v1
- https://arxiv.org/html/2501.13912v1
- https://arxiv.org/html/2604.13686
- https://www.ibm.com/think/news/indqa-llm-benchmark
- https://arxiv.org/html/2511.04727v1
- https://medium.com/@manick2411/benchmarking-llm-tokenization-indic-languages-under-the-lens-6b86844e54e9
- https://arxiv.org/html/2603.17915
- https://blog.belsterns.com/post/sarvam-m-the-powerhouse-indic-ai-model-built-for-efficiency
- https://awesomeagents.ai/leaderboards/agentic-ai-benchmarks-leaderboard/
- https://arxiv.org/abs/2311.12983
- https://hal.cs.princeton.edu/gaia
- https://simmering.dev/blog/agent-benchmarks/
- https://galileo.ai/blog/agent-evaluation-framework-metrics-rubrics-benchmarks
- https://kili-technology.com/blog/ai-benchmarks-guide-the-top-evaluations-in-2026-and-why-theyre-not-enough
- https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/
- https://testtimescaling.github.io/
- https://openreview.net/forum?id=eJ1yDj6vtH
- https://arxiv.org/abs/2512.02008
- https://openreview.net/forum?id=S3GhJooWIC
- https://openreview.net/forum?id=BMJ3pyYxu2
- https://arxiv.org/abs/2503.24235
- https://arxiv.org/abs/2603.15569
- https://openreview.net/pdf?id=HwCvaJOiCj
- https://arxiv.org/abs/2312.00752
- https://arxiv.org/abs/2604.14191
- https://arxiv.org/html/2507.12442v2
- https://github.com/thunlp/JustRL
- https://iclr-blogposts.github.io/2026/blog/2026/justrl/
- https://arxiv.org/pdf/2512.16649
- https://arxiv.org/html/2512.16649v1
- https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training
- https://openreview.net/forum?id=BOsuSLbD8L
- https://arxiv.org/abs/2512.24695
- https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
- https://abehrouz.github.io/files/NL.pdf
- https://arxiv.org/abs/2506.03320
- https://arxiv.org/html/2604.06169
- https://openreview.net/pdf?id=dTWfCLSoyl
- https://venturebeat.com/infrastructure/new-test-time-training-method-lets-ai-keep-learning-without-exploding
- https://introl.com/blog/ttt-e2e-test-time-training-long-context-inference-breakthrough-2026
- https://arxiv.org/abs/2412.09871
- https://github.com/facebookresearch/blt
- https://graphcore-research.github.io/byte-latent-transformer/
- https://venturebeat.com/ai/metas-new-blt-architecture-replaces-tokens-to-make-llms-more-efficient-and-versatile
- https://www.primeintellect.ai/blog/synthetic-1
- https://arxiv.org/abs/2601.22607
- https://arxiv.org/abs/2604.02621
- https://huggingface.co/papers/2604.03128
- https://github.com/opendilab/awesome-RLVR
- https://startup.ai/startup-idea/ai-gst-invoicing-sui2je
- https://invoiso.ai/automated-gst-billing-software/
- https://rajeshrnair.com/blog/ai-technology/ai-business/ai-accounting-gst-tools-indian-businesses
- https://smestreet.in/technology/smbs-in-india-focus-on-ai-spend-control-and-compliance-11704157
- https://www.appzen.com/blog/how-ai-simplifies-indias-einvoice-processing
- https://aidevdayindia.org/blogs/the-state-of-agentic-ai-in-india-2026/agentic-ai-fintech-applications-india.html
- https://ledgers.cloud/
- https://www.tallyatcloud.com/article/smart-ai-accounting-billing-software-in-india-intelligent-gst-solutions-automated-ledger-management-inventory-control-business-growth-tools-in-2026/1335/0/1
- https://www.businesstoday.in/technology/story/not-ai-models-ai-applications-are-quietly-driving-indias-revenue-streams-report-524654-2026-04-08
- https://blogs.nvidia.com/blog/state-of-ai-report-2026/
- https://medium.com/@pratik-rupareliya/top-20-indian-ai-companies-building-real-infrastructure-in-2026-a97473ffa287
- https://www.turing.com/resources/vertical-ai-agents
- https://www.aspirion.com/the-year-ai-transformed-revenue-cycle-2025-insights-and-2026-predictions/
- https://dev.to/dograh/where-voice-ai-agents-are-actually-getting-used-in-2026-49oe
- https://nexos.ai/blog/vertical-ai-agents/
- https://www.grandviewresearch.com/horizon/outlook/ai-agents-market/india
- https://awesmai.com/agencies/ai-agent-development-companies-in-india
- https://www.ringg.ai/blogs/best-voice-ai-agents-for-indian-languages
- https://www.haptik.ai/blog/voice-ai-agents-for-indian-languages
- https://www.lumay.ai/blogs/best-multilingual-voice-ai-tamil-hindi-telugu
- https://www.tabbly.io/blogs/best-ai-voice-agent-india
- https://www.zartek.in/best-voice-ai-agents-indian-languages/
- https://www.bolna.ai/
- https://www.carmaone.in/blog/best-voice-ai-agents-indian-languages-2026
- https://www.huskyvoice.ai/best-hindi-voice-ai
- https://www.squadstack.ai/voicebot/voice-ai-agent-for-indian-languages
- https://voicegenie.ai/en/voice-ai-agent-in-hindi
- https://www.truefan.ai/blogs/faceless-youtube-earnings-india-2026
- https://www.youtubeservices.in/youtube-ai-content-monetization-2026-guide/
- https://upgrowth.in/youtube-cpm-india-guide-2026/
- https://www.coherentmarketinsights.com/industry-reports/india-creator-economy-market
- https://www.podcastvideos.com/articles/youtube-2026-updates-ai-live-streaming-monetization/
- https://almcorp.com/blog/youtube-2026-ai-tools-in-app-shopping-roadmap/
- https://www.thewrap.com/industry-news/tech/youtube-2026-goals-ai-monetization-neal-mohan-letter/
- https://www.dqindia.com/interview/ai-agents-in-sales-what-indian-organisations-should-expect-in-2026-11174170
- https://www.crazehq.com/in/blog/ai-recruitment-trends-india
- https://www.ptechpartners.com/2026/02/04/top-10-ai-agents-transforming-sales-and-customer-engagement-in-2026/
- https://www.salesmate.io/blog/future-of-ai-agents/
- https://goconsensus.com/blog/17-best-ai-sales-agents-in-2026
- https://news.outsourceaccelerator.com/indias-hiring-2026-by-ai/
- https://www.horizontaltalent.com/blog/2025/12/17/2026-talent-acquisition-predictions-lean-into-an-AI-human-partnership
- https://www.imarcgroup.com/insight/top-factors-driving-growth-india-edtech-industry
- https://www.raysolute.com/indian-edtech-analysis-2026.html
- https://www.ncnonline.net/the-india-edtech-market-is-set-reach-new-heights-in-2026-with-ai-dominating-the-solutions/
- https://www.kenresearch.com/industry-reports/india-edtech-market
- https://www.mgiwebzone.com/indian-education-industry-marketing-statistics-report/
- https://razorpay.com/blog/how-global-edtech-companies-can-unlock-indias-30b-market/
- https://www.skydo.com/blog/edtech-market-india
- https://www.entrepreneurindia.com/blog/en/article/contribution-of-edtech-to-national-education-goals-an-analysis-of-top-companies.59388
- https://indiamarketentry.com/edtech-india-opportunity/
- https://www.insightsonindia.com/2025/12/25/artificial-intelligence-in-education/
- https://www.startupgrantsindia.com/nvidia-inception-program
- https://blogs.nvidia.com/blog/india-ai-mission-infrastructure-models/
- https://www.thundercompute.com/blog/nvidia-inception-program-guide
- https://grantedai.com/grants/nvidia-inception-program-ai-startups-cloud-credits-hardware-nvidia-b5c6d7e8
- https://techcrunch.com/2026/02/19/nvidia-deepens-early-stage-push-into-indias-ai-startup-ecosystem/
- https://blogs.nvidia.com/blog/india-inception-ai-startups/
- https://www.nvidia.com/en-us/startups/
- https://www.startupindia.gov.in/content/sih/en/ams-application/accelerator-program.html?applicationId=665575ace4b0608c0b18b273
- https://cxotoday.com/media-coverage/ai-grants-india-collaborates-with-nvidia-inception-to-empower-indian-entrepreneurs-in-forming-new-startups/
- https://huggingface.co/open-llm-leaderboard
- https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- https://huggingface.co/spaces/OpenEvals/every-leaderboards
- https://www.datacamp.com/blog/top-small-language-models
- https://analyticsindiamag.com/ai-news-updates/hugging-face-releases-smollm-a-series-of-small-language-models-beats-qwen2-and-phi-1-5/
- https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard
- https://huggingface.co/spaces/lmarena-ai/arena-leaderboard
- https://llm-stats.com/
- https://www.programming-helper.com/tech/vllm-2026-open-source-llm-inference-engine
- https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/
- https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/
- https://github.com/vllm-project/vllm
- https://www.gmicloud.ai/en/blog/which-ai-inference-platform-is-fastest-for-open-source-models-2026-engineering-guide
- https://openreview.net/forum?id=59OJOgKLzN
- https://blog.starmorph.com/blog/local-llm-inference-tools-guide
- https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm
- https://www.swebench.com/
- https://labs.scale.com/leaderboard/swe_bench_pro_public
- https://epoch.ai/benchmarks/swe-bench-verified
- https://openai.com/index/introducing-swe-bench-verified/
- https://swe-bench-live.github.io/
- https://benchlm.ai/coding
- https://www.swebench.com/verified.html
- https://dev.to/rahulxsingh/swe-bench-scores-and-leaderboard-explained-2026-54of
- https://www.codeant.ai/blogs/swe-bench-scores
- https://github.com/murataslan1/ai-agent-benchmark
- https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/
- https://github.com/NVIDIA-NeMo/Curator
- https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier
- https://www.emergentmind.com/topics/fineweb-edu-dataset
- https://developer.nvidia.com/blog/building-nemotron-cc-a-high-quality-trillion-token-dataset-for-llm-pretraining-from-common-crawl-using-nvidia-nemo-curator
- https://arxiv.org/html/2508.14444v2
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- https://arxiv.org/abs/2604.08970
- https://arxiv.org/pdf/2512.18470
- https://arxiv.org/html/2510.09721v3
- https://openreview.net/forum?id=41xrZ3uGuI
- https://arxiv.org/pdf/2601.11077
- https://hai.stanford.edu/ai-index/2026-ai-index-report/technical-performance
- https://benchlm.ai/blog/posts/best-open-source-llm
- https://aiproductivity.ai/blog/open-source-ai-models-comparison-2026/

*Compiled April 18 2026. All arXiv/OpenReview IDs listed were surfaced via WebSearch — cross-verification against official venues recommended before citing in a paper. Market-size numbers from aggregator sources are flagged "unverified."*
