# Multi-host TPU deploy — `scripts/train_xla_ddp.py`

## Single-host smoke test (v4-8, v6e-8, v5e-8)

One host, `xmp.spawn` forks one process per chip. Run on the TPU VM:

```bash
cd ~/aria
python scripts/train_xla_ddp.py --config configs/aria_v1_160m_fwe_500m.yaml
```

World size = local chip count (4 on v4-8, 8 on v6e-8). Effective batch =
`batch_size * grad_accum_steps * world_size`.

## Multi-host pod (v4-64, v6e-64, v5e-64)

Every worker must start the same command in parallel. Use
`--worker=all --command=...` so gcloud fans the command out.

```bash
QUEUE=aria-v4-64-spot   # or aria-v6e-64-us, etc.
ZONE=us-central2-b      # match the queued_resource zone

# Describe to get the node name once it's ACTIVE:
gcloud compute tpus queued-resources describe "$QUEUE" --zone="$ZONE"

# Ship the repo to every worker:
gcloud compute tpus tpu-vm scp --recurse ~/aria "${QUEUE}:~/aria" \
  --zone="$ZONE" --worker=all

# Install deps on every worker in parallel (one-time):
gcloud compute tpus tpu-vm ssh "$QUEUE" --zone="$ZONE" --worker=all \
  --command "cd ~/aria && pip install -q -r requirements.txt"

# Launch training on every worker in parallel:
gcloud compute tpus tpu-vm ssh "$QUEUE" --zone="$ZONE" --worker=all \
  --command "cd ~/aria && python scripts/train_xla_ddp.py \
             --config configs/aria_v1_1b_multihost.yaml" \
  2>&1 | tee ~/aria_1b_multihost.log
```

`xmp.spawn` inside the script will launch `nprocs = local chip count` on each
host; `xm.xrt_world_size()` returns the global chip count (32 on v4-64).

## Data on multi-host

The BPE `.bin` files must already exist under `./data/bpe_tokens/<dataset>/` on
**every** worker (each host reads its own disk). Two options:

1. **Pre-tokenize once, rsync:**
   ```bash
   gcloud compute tpus tpu-vm scp --recurse ./data/bpe_tokens \
     "${QUEUE}:~/aria/data/bpe_tokens" --zone="$ZONE" --worker=all
   ```

2. **Let worker-0 tokenize, then broadcast:** first run the script on
   `--worker=0` only; after it finishes BPE caching, rsync `data/bpe_tokens` to
   the rest, then launch on `--worker=all`.

## Checkpoints

`xm.save` writes from rank-0 only. Checkpoints land on worker-0's local disk at
`./checkpoints/<run_name>/`. Pull them off with:

```bash
gcloud compute tpus tpu-vm scp --recurse \
  "${QUEUE}:~/aria/checkpoints/aria_v1_1b_multihost" ./ckpts-pulled \
  --zone="$ZONE" --worker=0
```

## Resume after spot preemption

The script auto-detects `latest.pt` in `checkpoint_dir` on startup. After a
preemption, re-run the same launch command — every rank loads the same
checkpoint and training continues at the saved step. Make sure `latest.pt`
exists on every worker if the checkpoint was broadcast (rank-0 only is fine
since only rank-0 reads during resume would desync replicas — the current
script loads on *every* rank, so you must broadcast `latest.pt` to all
workers before resuming).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ProcessGroup init timeout` | One worker slow to reach `xmp.spawn` | Retry; ensure NTP sync |
| Loss diverges on one rank only | Rank-local seed collision | Confirm `_set_seed(seed, rank)` runs |
| OOM on chip | Per-chip batch too big | Lower `batch_size`, raise `grad_accum_steps` proportionally |
| Hangs at first `xm.optimizer_step` | All-reduce topology mis-set | Check `TPU_WORKER_ID`/`TPU_WORKER_HOSTNAMES` env vars |
