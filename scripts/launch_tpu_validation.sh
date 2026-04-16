#!/usr/bin/env bash
# =========================================================================
# Phase 2a: Launch 3 TPU v4-8 VMs in parallel for hypothesis validation
# =========================================================================
#
# Creates 3 TPU VMs in us-central2-b (on-demand v4 quota), clones the repo,
# installs deps, tokenizes the dataset, and starts training — one config
# per VM. All three run in parallel and finish in ~45 min.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Project set: gcloud config set project aria-trc
#   - TPU API enabled: gcloud services enable tpu.googleapis.com
#
# Usage:
#   bash scripts/launch_tpu_validation.sh
#
# Monitor:
#   gcloud alpha compute tpus tpu-vm ssh aria-v1-inter --zone=us-central2-b --command="tail -20 ~/aria/train.log"
#   gcloud alpha compute tpus tpu-vm ssh aria-v2-pure  --zone=us-central2-b --command="tail -20 ~/aria/train.log"
#   gcloud alpha compute tpus tpu-vm ssh aria-v2-inter --zone=us-central2-b --command="tail -20 ~/aria/train.log"
#
# Download results:
#   for name in aria-v1-inter aria-v2-pure aria-v2-inter; do
#     gcloud alpha compute tpus tpu-vm scp ${name}:~/aria/checkpoints/*/summary.json ./${name}_summary.json --zone=us-central2-b
#     gcloud alpha compute tpus tpu-vm scp ${name}:~/aria/checkpoints/*/recall_v2.json ./${name}_recall.json --zone=us-central2-b 2>/dev/null
#   done
#
# Cleanup (IMPORTANT — don't leave VMs running, they consume quota):
#   for name in aria-v1-inter aria-v2-pure aria-v2-inter; do
#     gcloud alpha compute tpus tpu-vm delete ${name} --zone=us-central2-b --quiet
#   done
# =========================================================================

set -euo pipefail

PROJECT="aria-trc"
ZONE="us-central2-b"
TPU_TYPE="v4-8"
TPU_VERSION="tpu-ubuntu2204-base"
REPO="https://github.com/Ashishgirigoswami/aria.git"

# VM name → config file mapping
declare -A CONFIGS
CONFIGS[aria-v1-inter]="configs/val_v1_interleave_30m.yaml"
CONFIGS[aria-v2-pure]="configs/val_v2_pure_30m.yaml"
CONFIGS[aria-v2-inter]="configs/val_v2_interleave_30m.yaml"

# The setup + training script that runs on each VM
REMOTE_SCRIPT='
#!/bin/bash
set -euxo pipefail

CONFIG="$1"

# Install deps
pip install -q torch torch_xla[tpu] \
  -f https://storage.googleapis.com/libtpu-releases/index.html
pip install -q tiktoken pyyaml tqdm numpy datasets

# Clone repo
cd ~
rm -rf aria
git clone '"$REPO"'
cd aria

# Prepare data (tokenize wikitext-103 once)
python scripts/prepare_data.py --dataset wikitext-103 --max-tokens 50000000

# Train
python scripts/train_xla.py --config "$CONFIG" 2>&1 | tee train.log

# Run recall eval after training
python -c "
import torch, os, sys
sys.path.insert(0, \".\")
from aria.eval_recall import run_from_checkpoint, PasskeyConfig
from aria.lsa import LSALanguageModel
from aria.lsa_v2 import LSAv2LanguageModel
from aria.baseline import BaselineLanguageModel
import yaml, glob

# Find the checkpoint
ckpt = glob.glob(\"checkpoints/*/final.pt\")[0]
cfg_path = \"$CONFIG\"
cfg = yaml.safe_load(open(cfg_path))
model_name = cfg[\"model\"][\"name\"]
mcfg = dict(cfg[\"model\"])
mcfg.pop(\"name\")
mcfg[\"vocab_size\"] = 50257

reg = {\"lsa\": LSALanguageModel, \"lsa_v2\": LSAv2LanguageModel, \"baseline\": BaselineLanguageModel}
factory = lambda: reg[model_name](**mcfg)

recall_cfg = PasskeyConfig(
    context_lengths=(128, 256, 512),
    depths=(0.1, 0.5, 0.9),
    passkey_len=8, n_trials=16,
    filler_range=(0, 30000),
    passkey_range=(0, 30000),
    run_control=True,
)

import torch_xla
device = torch_xla.device()
report = run_from_checkpoint(ckpt, factory, 50257,
    out_path=ckpt.replace(\"final.pt\", \"recall_v2.json\"), cfg=recall_cfg)
s = report[\"summary\"]
print(f\"Recall: retr={s[\"mean_retrieval_loss\"]:.3f} ctrl={s[\"mean_control_loss\"]:.3f} delta={s[\"mean_retrieval_delta\"]:+.4f}\")
"

echo "=== DONE: $CONFIG ==="
'

echo "Creating 3 TPU v4-8 VMs in ${ZONE}..."

# Step 1: Create all 3 VMs in parallel
for VM_NAME in "${!CONFIGS[@]}"; do
  echo "  Creating ${VM_NAME}..."
  gcloud alpha compute tpus tpu-vm create "${VM_NAME}" \
    --zone="${ZONE}" \
    --accelerator-type="${TPU_TYPE}" \
    --version="${TPU_VERSION}" \
    --project="${PROJECT}" &
done
wait
echo "All VMs created."

# Step 2: Launch training on each VM in parallel
for VM_NAME in "${!CONFIGS[@]}"; do
  CONFIG="${CONFIGS[$VM_NAME]}"
  echo "  Launching training on ${VM_NAME} with ${CONFIG}..."
  gcloud alpha compute tpus tpu-vm ssh "${VM_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --command="bash -c '$(echo "$REMOTE_SCRIPT")' -- ${CONFIG}" &
done

echo ""
echo "All 3 training jobs launched in background."
echo ""
echo "Monitor with:"
echo "  gcloud alpha compute tpus tpu-vm ssh aria-v2-inter --zone=${ZONE} --command='tail -30 ~/aria/train.log'"
echo ""
echo "When all finish, download results:"
echo "  for name in aria-v1-inter aria-v2-pure aria-v2-inter; do"
echo "    gcloud alpha compute tpus tpu-vm scp \${name}:~/aria/checkpoints/*/summary.json ./\${name}_summary.json --zone=${ZONE}"
echo "  done"
echo ""
echo "CLEANUP (do this or you burn quota):"
echo "  for name in aria-v1-inter aria-v2-pure aria-v2-inter; do"
echo "    gcloud alpha compute tpus tpu-vm delete \${name} --zone=${ZONE} --quiet"
echo "  done"
