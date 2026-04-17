"""Pure Mamba-3 language model — for Kaggle T4x2 CUDA validation.

Uses state-spaces/mamba's official Mamba3 CUDA kernels (Triton + TileLang)
instead of our torch_xla-targeted scaffold in aria.mamba3. Stack of Mamba-3
blocks, RMSNorm, tied embeddings. Roughly 131M params at d_model=768,
n_layers=12 — matches ARIA v1 for direct comparison.

Once validated on Kaggle, ARIA's shared-MLA latent + joint-softmax fusion
can be layered on top. This module is the "does Mamba-3 train at all" check.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba3 import Mamba3 as _Mamba3
    HAS_MAMBA3 = True
except ImportError:
    _Mamba3 = None
    HAS_MAMBA3 = False


class _MambaBlock(nn.Module):
    """Pre-norm residual wrapper around a Mamba-3 layer."""

    def __init__(self, d_model: int, d_state: int, headdim: int,
                 mimo_rank: int, chunk_size: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mamba = _Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=True,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class Mamba3LanguageModel(nn.Module):
    """Decoder-only LM using stacked Mamba-3 blocks. No attention."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        d_state: int = 128,
        headdim: int = 64,
        mimo_rank: int = 4,
        chunk_size: int = 16,
        max_seq_len: int = 256,
        tie_weights: bool = True,
        **_unused,
    ) -> None:
        super().__init__()
        if not HAS_MAMBA3:
            raise RuntimeError(
                "mamba-ssm not installed. Run: "
                "pip install git+https://github.com/state-spaces/mamba.git"
            )
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            _MambaBlock(d_model, d_state, headdim, mimo_rank, chunk_size)
            for _ in range(n_layers)
        )
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss


__all__ = ["Mamba3LanguageModel"]
