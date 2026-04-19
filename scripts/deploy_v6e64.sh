#!/bin/bash
# Deploy + launch 160M training on a v6e-64 pod when it goes ACTIVE.
# Usage:   bash scripts/deploy_v6e64.sh aria-v6e-64-us us-east1-d
#          bash scripts/deploy_v6e64.sh aria-v6e-32-eu europe-west4-a
#
# Does end-to-end:
#   1. Tarball repo locally (excl. data/ckpts/.git)
#   2. scp to every worker
#   3. pip install torch + torch_xla 2.9 + deps on every worker
#   4. Synchronized launch: train_xla_spmd.py on all workers (one per host)

set -euo pipefail

POD="${1:-aria-v6e-64-us}"
ZONE="${2:-us-east1-d}"
CONFIG="${3:-configs/aria_v1_160m_v6e64.yaml}"
# Derive run_name from config file — reads logging.run_name via grep
RUN_NAME=$(grep -E "^  run_name:" "$CONFIG" | awk '{print $2}')
if [ -z "$RUN_NAME" ]; then
  echo "Could not extract run_name from $CONFIG"; exit 2
fi

export PATH="/c/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin:$PATH"

# Verify pod is ACTIVE
STATE=$(gcloud compute tpus queued-resources describe "$POD" --zone="$ZONE" \
         --format="value(state.state)" 2>/dev/null)
if [ "$STATE" != "ACTIVE" ]; then
  echo "Pod $POD in $ZONE is '$STATE', not ACTIVE. Bailing."
  exit 2
fi

echo "=== 1/4 packaging repo ==="
tar --exclude='data' --exclude='checkpoints' --exclude='runs' --exclude='.git' \
    --exclude='decks' --exclude='ckpts-pulled' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='figures/*.png' \
    -czf /tmp/aria-deploy.tgz aria scripts configs requirements.txt
ls -lh /tmp/aria-deploy.tgz

echo "=== 2/4 scp tarball to all workers ==="
yes y | gcloud compute tpus tpu-vm scp /tmp/aria-deploy.tgz \
  "${POD}:/tmp/aria-deploy.tgz" --zone="$ZONE" --worker=all 2>&1 | tail -5

echo "=== 3/4 install deps on all workers (~5 min) ==="
cat > /tmp/setup_v6e.sh <<'SSH'
set -e
mkdir -p ~/aria && tar xzf /tmp/aria-deploy.tgz -C ~/aria
cd ~/aria
pip install -q -r requirements.txt 2>&1 | tail -2
pip install -q torch==2.9.0 "torch_xla[tpu]==2.9.0" \
  -f https://storage.googleapis.com/libtpu-releases/index.html 2>&1 | tail -2
pip uninstall -y fla-core triton 2>/dev/null || true
python3 -c "import torch_xla; print('torch_xla', torch_xla.__version__)"
SSH

yes y | gcloud compute tpus tpu-vm scp /tmp/setup_v6e.sh "${POD}:/tmp/setup_v6e.sh" \
  --zone="$ZONE" --worker=all 2>&1 | tail -3
yes y | gcloud compute tpus tpu-vm ssh "$POD" --zone="$ZONE" --worker=all \
  --command='bash /tmp/setup_v6e.sh' 2>&1 | tail -10

echo "=== 4/4 synchronized train launch on all workers ==="
cat > /tmp/launch_train.sh <<SSH
cd ~/aria
pkill -9 -f train_xla_spmd 2>/dev/null || true
sleep 2
mkdir -p checkpoints/${RUN_NAME} logs
setsid nohup env PJRT_DEVICE=TPU XLA_USE_SPMD=1 \
  python3 scripts/train_xla_spmd.py --config ${CONFIG} \
  > logs/train.log 2>&1 < /dev/null &
echo "\$(hostname) launched pid=\$!"
SSH

yes y | gcloud compute tpus tpu-vm scp /tmp/launch_train.sh \
  "${POD}:/tmp/launch_train.sh" --zone="$ZONE" --worker=all 2>&1 | tail -3
yes y | gcloud compute tpus tpu-vm ssh "$POD" --zone="$ZONE" --worker=all \
  --command='bash /tmp/launch_train.sh' 2>&1 | tail -10

echo ""
echo "=== all workers launched. Poll rank-0 progress with: ==="
echo "  gcloud compute tpus tpu-vm ssh $POD --zone=$ZONE --worker=0 \\"
echo "    --command='tail -20 ~/aria/logs/train.log'"
