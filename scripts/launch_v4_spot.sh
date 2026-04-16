#!/usr/bin/env bash
# Launch ARIA v2 150M training on spot v4-8 in us-central2-b.
# Uses seq_len=256 config for faster XLA compile (halves HLO graph).
# Enables XLA persistent cache so restarts after preemption skip compile.
set -euxo pipefail

PROJECT="aria-trc"
ZONE="us-central2-b"
QUEUE="aria-v4-spot-us"
VM_NAME="aria-n-v4-spot"
CONFIG="${CONFIG:-configs/aria_v2_150m_t256.yaml}"
REPO="https://github.com/Ashishgirigoswami/aria.git"

# Poll queue state
until [ "$(gcloud compute tpus queued-resources describe "${QUEUE}" \
          --zone="${ZONE}" --project="${PROJECT}" \
          --format='get(state.state)' 2>/dev/null)" = "ACTIVE" ]; do
  echo "waiting for ${QUEUE} to become ACTIVE..."
  sleep 60
done

# Remote setup + training
REMOTE_SCRIPT='
#!/bin/bash
set -euxo pipefail
CONFIG="$1"

# Persistent XLA compile cache — survives preemption, reuses compiled graphs.
export XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache
mkdir -p $XLA_PERSISTENT_CACHE_PATH

# Install deps (idempotent)
if ! python -c "import torch_xla" 2>/dev/null; then
  pip install -q torch torch_xla[tpu] \
    -f https://storage.googleapis.com/libtpu-releases/index.html
  pip install -q tiktoken pyyaml tqdm numpy datasets
fi

cd ~
if [ ! -d aria ]; then
  git clone '"$REPO"'
fi
cd aria
git pull --rebase || true

# Tokenize (cached — no-op on re-run)
python scripts/prepare_data.py --dataset wikitext-103 --max-tokens 50000000

# Train with auto-resume from latest.pt
XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache \
  nohup python scripts/train_xla.py --config "$CONFIG" > train.log 2>&1 &
echo "training PID=$! config=$CONFIG"
'

yes y | gcloud compute tpus tpu-vm ssh "${VM_NAME}" \
  --zone="${ZONE}" --project="${PROJECT}" \
  --command="bash -c '$(echo "$REMOTE_SCRIPT")' -- ${CONFIG}"

echo "Launched. Monitor:"
echo "  yes y | gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone=${ZONE} --command='tail -30 ~/aria/train.log'"
