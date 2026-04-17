# Phase 2: 1B model plan (post-131M baseline)

## Two parallel paths

### Path A (TPU, ready to deploy when 131M finishes)
**Architecture**: ARIA v1 1B — scales up current config.
- 1.1B params, d_model=2048, n_layers=24, seq_len=1024
- Vector SSM (Mamba-2 style), xla_scan-compatible
- Novelty: 2/3 claims (shared MLA + joint-softmax). Lost: matrix-state Mamba-3.
- Config: `configs/aria_v1_1b_fwe.yaml`

**Compute**:
- Need: v4-8 spot (128GB HBM) OR v5e-8 (64GB)
- Memory math: 1.1B × 16 bytes (params+grads+optim bf16) = 17.6GB state + ~10GB activations
- v4-8 fits easily. v5e-8 tight. v6e-4 impossible.

**Timing**:
- 60K steps × ~20-30s/step (2x longer scan vs 131M) = 20-50 hrs per run
- Plus compile time ~30-60 min first start
- Spot preemption: mitigated by ckpt_every=200 + auto-resume

**Data**:
- FineWeb-Edu sample-10BT subset (streaming via our retry-enabled loader)
- ~2B tokens (under-Chinchilla but realistic for TRC free tier)

### Path B (GPU, when grants arrive)
**Architecture**: ARIA Mamba-3 1B — full novelty.
- Same 1.1B scaffold but Mamba-3 MIMO (matrix state) SSM
- state-spaces/mamba CUDA kernels (works out-of-box on H100/A100)
- Novelty: 3/3 claims restored (shared MLA + joint-softmax + matrix state)
- Config: `configs/aria_mamba3_1b_fwe.yaml` (TODO — create when deploying)

**Compute**:
- Need: 8× H100 for 2-3 days OR 4× MI300X for similar
- Grants: Lambda ($2-10K), NVIDIA Inception (DGX), AMD AAC, HF Community

**Timing**: 3-5 days on 8×H100 for 20-40B tokens

## Post-131M deploy sequence (TPU Path A)

1. **Detect 131M training done** (final.pt present)
2. **Download 131M checkpoints to D:/mytllm/aria/runs/** (current poll config)
3. **Report 131M results** — loss curve + best val ppl
4. **Request new spot TPU**:
   ```
   gcloud compute tpus queued-resources create aria-v4-spot-1b \
     --node-id=aria-n-v4-1b --project=aria-trc \
     --zone=us-central2-b --accelerator-type=v4-8 \
     --runtime-version=tpu-ubuntu2204-base --spot
   ```
5. **When ACTIVE**: setup similar to current (torch+torch_xla[tpu]==2.9.0, libtpu, clone repo, tokenize FineWeb-Edu)
6. **Launch**: `python scripts/train_xla.py --config configs/aria_v1_1b_fwe.yaml` with `XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache`
7. **Poll every 30 min** same as before

## Risks to flag before starting 1B

1. **FineWeb-Edu streaming on TPU VM** — 2B tokens ~800GB raw text. Our retry-loader resumes partial downloads, but first-time tokenization takes hours. Pre-cache on a high-RAM VM first.
2. **v4-8 spot availability** — we've seen 1-12hr queue waits. Plan for latency.
3. **Preemption during compile** — compile takes 30-60 min on first start. If preempted at step 1 we restart from scratch. First ckpt at step 200 (~2-3 hrs on 1B) is critical.
4. **XLA graph size** — 1B model at seq=1024 is larger than 131M at seq=256. May hit compile ceiling on v4-8. Fallback: reduce seq_len to 512 or layers to 18.
5. **Evaluation not possible on TPU alone** — lm-eval-harness wants CUDA. After training, download ckpt + run eval on Kaggle T4.

## Skipped optimizations (defer)

- **u-muP port** (cycle 4): save $5-20K sweep cost at 1B — but adds ~1 week of careful coord-check work. Skip for this first 1B run; do on second run.
- **WY chunkwise GDN** (cycle 1): needs v2 architecture. Skip; stay v1 vector SSM on TPU.
- **Long-context extension** (cycle 5): seq=1024 at pretrain, extend to 32k post-hoc if needed.
- **u-muP + WY + Mamba-3 all together** (all novelty + efficient training): Phase 3, needs GPU grant + multi-week effort.

## Deliverable after Phase 2

- 1.1B open-weight model on HuggingFace Hub
- Eval results on HellaSwag/ARC/PIQA/WinoGrande/LAMBADA/MMLU
- Target: beat Pythia-1B on ≥3 benchmarks (credibility floor for grants/fundraising)
