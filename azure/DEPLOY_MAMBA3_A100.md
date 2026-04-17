# Deploy ARIA LSA+Mamba-3 131M on Azure A100

Parallel training to TPU v1 baseline for a fair 3/3-novelty ablation.

## Azure VM spec
- **Standard_NC24ads_A100_v4** (1× A100 80GB, $3.67/hr pay-as-you-go)
- Expected runtime: **~12-18 hrs** for 8000 steps
- Estimated cost: **$45-70** (well within $1000 credit)

## Step 1 — Create VM (5 min, on your laptop)

```bash
# Install Azure CLI first if needed:
# https://learn.microsoft.com/cli/azure/install-azure-cli

az login
az group create --name aria-rg --location eastus2

# Single A100 80GB VM
az vm create \
  --resource-group aria-rg \
  --name aria-a100 \
  --image microsoft-dsvm:ubuntu-hpc:2204:latest \
  --size Standard_NC24ads_A100_v4 \
  --admin-username azureuser \
  --ssh-key-values ~/.ssh/id_rsa.pub \
  --public-ip-sku Standard \
  --storage-sku Premium_LRS \
  --os-disk-size-gb 256

# Get public IP
IP=$(az vm show -d -g aria-rg -n aria-a100 --query publicIps -o tsv)
echo "VM IP: $IP"
```

## Step 2 — Setup on VM (10 min)

```bash
ssh azureuser@$IP

# Verify A100 visible
nvidia-smi

# Clone repo
cd ~
git clone --depth=1 https://github.com/Ashishgirigoswami/aria.git
cd aria

# Install deps (Ubuntu HPC DSVM has CUDA + PyTorch pre-installed)
pip install --user tiktoken pyyaml tqdm datasets transformers
pip install --user mamba-ssm causal-conv1d --no-build-isolation

# Verify Mamba-3 imports
python -c "from mamba_ssm.modules.mamba3 import Mamba3; print('Mamba-3 OK')"
python -c "from aria.lsa_mamba3 import LSAMamba3LanguageModel; print('ARIA+Mamba-3 OK')"

# Tokenize data (once, ~2-3 min for 50M wikitext-103 tokens)
python scripts/prepare_data.py --dataset wikitext-103 --max-tokens 50000000
```

## Step 3 — Launch training (15 min to first step)

```bash
# Inside ~/aria on the VM:
nohup python scripts/train_lsa_mamba3_cuda.py \
    --config configs/aria_mamba3_131m_azure.yaml \
    > train.log 2>&1 &

# Verify running
tail -f train.log
# Expected first lines:
#   Vocab: 50257 | train: 49,999,XXX | val: 246,XXX
#   Model: LSA+Mamba-3 ~131M on cuda:0 (torch.bfloat16)
```

## Step 4 — Monitor from laptop

```bash
# Every hour or so:
ssh azureuser@$IP "grep -oE 'step [0-9]+/8000 loss=[0-9.]+' ~/aria/train.log | tail -3; ls ~/aria/checkpoints/aria_mamba3_131m_azure/"
```

## Step 5 — Download results when done

```bash
# Once final.pt appears:
mkdir -p D:/mytllm/aria/runs/aria_mamba3_131m_azure
scp azureuser@$IP:~/aria/checkpoints/aria_mamba3_131m_azure/best.pt D:/mytllm/aria/runs/aria_mamba3_131m_azure/
scp azureuser@$IP:~/aria/checkpoints/aria_mamba3_131m_azure/summary.json D:/mytllm/aria/runs/aria_mamba3_131m_azure/
scp azureuser@$IP:~/aria/train.log D:/mytllm/aria/runs/aria_mamba3_131m_azure/
```

## Step 6 — **DESTROY VM IMMEDIATELY AFTER** (critical — Azure bills by the minute)

```bash
az vm delete -g aria-rg -n aria-a100 --yes --no-wait
az disk delete -g aria-rg -n aria-a100_OsDisk_1_xxx --yes --no-wait  # check exact name with `az disk list -g aria-rg`
# Keep the resource group (no cost when empty)
```

## Expected output (if Mamba-3 is working)

Based on v1 TPU trajectory (loss 8.37 → ~3.0 at step 8000):
- Mamba-3 should track v1 closely in first 1000 steps
- If Mamba-3 diverges >5% worse by step 4000 → something wrong
- If Mamba-3 beats v1 by >0.2 ppl at final → **novelty confirmed, scale to 1B**
- If within 0.2 ppl → tie; prefer v1 for simplicity (TPU-friendly)
- If Mamba-3 loses by >0.5 ppl → matrix state isn't helping at 131M; reconsider architecture

## Common issues

### causal-conv1d build fails
```bash
# Often fixed by forcing torch ABI match:
pip install --user causal-conv1d --no-build-isolation --force-reinstall
```

### mamba-ssm import fails with "undefined symbol"
```bash
# torch and mamba-ssm version mismatch. Check:
pip show torch mamba-ssm
# If needed, pin:
pip install --user torch==2.4.0 mamba-ssm==2.2.2 --force-reinstall
```

### CUDA OOM at batch=4
- A100 80GB should handle batch=4 easily for 131M
- If OOM on A100 40GB variant: reduce `batch_size` to 2, increase `grad_accum_steps` to 32

## What to compare after both finish

| Metric | v1 (TPU) | LSA+Mamba-3 (Azure) |
|---|---|---|
| Final train loss | ? | ? |
| Best val loss | ? | ? |
| Val perplexity | ? | ? |
| HellaSwag acc | ? | ? |
| ARC-E acc | ? | ? |
| PIQA acc | ? | ? |
| LAMBADA acc | ? | ? |

If LSA+Mamba-3 wins on ≥3 benchmarks → scale Mamba-3 to 1B with NVIDIA/Lambda grant.
