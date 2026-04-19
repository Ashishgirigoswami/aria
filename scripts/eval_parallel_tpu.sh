#!/bin/bash
# Parallel eval of ARIA 160M across v4-32-spot's 4 workers.
# Each worker runs a different task subset on 1 TPU chip via pad_to_max.
# Fan-out: 6 benchmarks → 4 workers, ~1.5-2h total wallclock (vs ~5h serial).
#
# Usage (from laptop):
#   bash scripts/eval_parallel_tpu.sh <config> <ckpt-name>
# Defaults:
#   config = configs/aria_v1_160m_multihost.yaml
#   ckpt   = final.pt  (falls back to latest.pt if final not written)

set -euo pipefail

CFG="${1:-configs/aria_v1_160m_multihost.yaml}"
CKPT="${2:-final.pt}"
RUN_NAME="aria_v1_160m_multihost"
BUCKET="gs://aria-trc-ckpts/${RUN_NAME}"
ZONE="us-central2-b"
POD="aria-v4-32-spot"

# Task partition — balanced by measured eval duration from the 131M run.
declare -A TASKS=(
  [0]="hellaswag"                       # ~90 min, longest single task
  [1]="arc_easy,arc_challenge,piqa"     # ~60 min combined
  [2]="winogrande,openbookqa"           # ~40 min combined
  [3]="lambada_openai"                  # ~70 min
)

# Per-worker eval launcher written to the VM, then invoked via ssh.
cat > /tmp/aria_eval_worker.sh <<'WSH'
#!/bin/bash
set -e
TASKS="$1"
CFG="$2"
CKPT="$3"
BUCKET="$4"
RUN_NAME="$5"
cd ~/aria
mkdir -p checkpoints/"$RUN_NAME"
# Pull the eval checkpoint from GCS if not already local.
if [ ! -f "checkpoints/$RUN_NAME/$CKPT" ]; then
  gsutil -q cp "$BUCKET/$CKPT" "checkpoints/$RUN_NAME/$CKPT" || {
    echo "[$(hostname)] FATAL: $CKPT not in GCS"; exit 2
  }
fi
pip install -q lm-eval==0.4.11 2>&1 | tail -1

# One task group per worker. pad_to_max=True for single-XLA-compile on TPU.
OUT="logs/eval_${TASKS//,/_}.json"
mkdir -p logs
setsid nohup python3 scripts/eval_harness.py \
  --ckpt "checkpoints/$RUN_NAME/$CKPT" \
  --config "$CFG" \
  --tasks "$TASKS" \
  --device xla \
  --max-length 256 \
  --pad-to-max \
  --output "$OUT" \
  > "logs/eval_${TASKS//,/_}.log" 2>&1 < /dev/null &
echo "[$(hostname)] eval pid=$! tasks=$TASKS out=$OUT"
WSH

export PATH="/c/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin:$PATH"

# Ship + launch on each worker in parallel.
for W in 0 1 2 3; do
  (
    gcloud compute tpus tpu-vm scp /tmp/aria_eval_worker.sh \
      "${POD}:/tmp/aria_eval_worker.sh" --zone="$ZONE" --worker="$W" 2>&1 | tail -1
    yes y | gcloud compute tpus tpu-vm ssh "$POD" --zone="$ZONE" --worker="$W" \
      --command="bash /tmp/aria_eval_worker.sh '${TASKS[$W]}' '$CFG' '$CKPT' '$BUCKET' '$RUN_NAME'" \
      2>&1 | tail -2
  ) &
done
wait
echo "=== all 4 workers launched ==="
echo "Each worker writes logs/eval_<tasks>.json; pull with:"
echo "  gcloud compute tpus tpu-vm scp ${POD}:~/aria/logs/eval_*.json ./ --zone=$ZONE --worker=all"
