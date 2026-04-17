# Kaggle T4x2 Mamba-3 Training

Run ARIA's Mamba-3 baseline on free Kaggle T4x2. CUDA-native, no torch_xla.
Cross-compare perplexity / benchmarks with the TPU v1 run.

## Setup

1. Kaggle → New Notebook → **Accelerator = GPU T4 x2**.
2. Settings → Internet ON (required for pip + HuggingFace datasets).

## Notebook cells (copy-paste)

### Cell 1: Deps (~3 min)
```python
!pip install -q torch==2.4.0 transformers==4.45.0 datasets tiktoken tqdm pyyaml
!pip install -q mamba-ssm==2.2.2 causal-conv1d>=1.2.0 triton==3.0.0
!pip install -q git+https://github.com/state-spaces/mamba.git
!python -c "from mamba_ssm.modules.mamba3 import Mamba3; print('Mamba-3 OK')"
!python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

### Cell 2: Clone ARIA
```python
import os
os.chdir('/kaggle/working')
!rm -rf aria
!git clone --depth=1 https://github.com/Ashishgirigoswami/aria.git
%cd /kaggle/working/aria
!python scripts/prepare_data.py --dataset wikitext-103 --max-tokens 50000000
```

### Cell 3: Train (8000 steps, resumes automatically)
```python
!python scripts/train_mamba3_kaggle.py --config configs/mamba3_150m_kaggle.yaml
```

### Cell 4: Eval after training
```python
!pip install -q lm-eval>=0.4
!python scripts/eval_harness.py \
    --ckpt /kaggle/working/checkpoints/mamba3_150m_kaggle/best.pt \
    --config configs/mamba3_150m_kaggle.yaml \
    --tasks hellaswag,arc_easy,piqa,winogrande,lambada_openai \
    --output /kaggle/working/mamba3_eval.json
!cat /kaggle/working/mamba3_eval.json
```

## Session management (Kaggle 9-hour limit)

- `ckpt_every: 100` in config → save every ~1-2 min of work
- When session ends: **File → Save Version → Quick Save**. Output is persisted.
- Next session: **Add Data** → your previous notebook output. Copy `latest.pt` into `/kaggle/working/checkpoints/mamba3_150m_kaggle/` before running Cell 3.
- Cell 3 auto-resumes from `latest.pt`.

## Expected timing

| Phase | Time |
|---|---|
| Cell 1 (install) | 3-5 min |
| Cell 2 (clone + tokenize) | 2-3 min |
| First compile (Triton kernels) | 5-15 min |
| Steady-state throughput | ~2000-4000 tokens/sec → ~15-30 hrs total |
| Val eval every 400 steps | 1-2 min |
| Full 8000 steps | ~2-4 Kaggle sessions |

## Notes

- T4 Turing does NOT support bf16 — config uses **fp16 + GradScaler**.
- DataParallel across 2 T4s is simpler than DDP for Kaggle notebooks.
- If `nn.RMSNorm` fails with AttributeError, upgrade torch to 2.4+.
- If mamba-ssm import fails at first run, restart kernel after install.
- Mamba-3 MIMO rank=4, chunk_size=16 is the official default from state-spaces/mamba.

## Comparison target

- ARIA v1 (TPU, 131M, vector SSM): in progress as of 2026-04-17, expected ~22-34 hrs
- Mamba-3 Kaggle (T4x2, ~131M, matrix state): target val perplexity **within 0.3 ppl** of v1
- If Mamba-3 beats v1 by >0.5 ppl: pivot ARIA v2 architecture to use Mamba-3 as the SSM slot.
