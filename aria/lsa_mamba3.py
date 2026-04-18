"""LSA + Mamba-3 variant — preserves ARIA's 3/3 novelty at 131M.

This module defines a language model that combines:
- LSA attention blocks (shared MLA latent + joint-softmax fusion over
  per-position SSM readout)
- Mamba-3 SSM blocks as the recurrent track (replacing v1's vector SSM)
- 3:1 attention:Mamba-3 interleave ratio (Qwen3-Next validated)

For fair ablation vs `aria.lsa.LSALanguageModel` at 131M:
- Same d_model, n_layers, d_head, d_ff, d_kv_latent, max_seq_len
- Same 3:1 interleave
- Same optimizer + LR + data (via matching config)

The Mamba-3 block is the SSM slot's replacement. LSA's "state attention"
(joint-softmax over local K/V + per-position state K/V) now consumes
Mamba-3's per-timestep readout instead of the vector-SSM state.

Runs on CUDA (requires mamba-ssm + causal-conv1d installed). Not intended
for TPU — Mamba-3 CUDA kernels have no XLA port as of April 2026.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lsa import (
    LSAAttention,
    _FullAttentionBlock,
    apply_rope,
    precompute_rope,
)
from .nn_utils import RMSNorm, SwiGLU

try:
    from mamba_ssm.modules.mamba3 import Mamba3 as _Mamba3
    HAS_MAMBA3 = True
except ImportError:
    _Mamba3 = None
    HAS_MAMBA3 = False


class LSAMamba3Block(nn.Module):
    """ARIA v3 block: LSA attention + Mamba-3 SSM + joint softmax via residual.

    Architecture:
      x  -> RMSNorm -> LSAAttention (with Mamba-3 in the state path) -> + -> x
            RMSNorm -> SwiGLU MLP -> +

    The LSAAttention internally uses shared-MLA compressed latent and a
    joint softmax over local K/V and state K/V. Here the "state" that
    feeds the joint softmax is derived from the Mamba-3 per-timestep
    readout rather than from a plain vector SSM scan.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 d_kv_latent: int, d_state: int, dropout: float = 0.0,
                 mamba3_headdim: int = 64, mamba3_chunk_size: int = 16,
                 is_mimo: bool = False, mimo_rank: int = 4,
                 qk_norm: bool = False):
        super().__init__()
        if not HAS_MAMBA3:
            raise RuntimeError(
                "mamba-ssm not installed. Install with "
                "`pip install mamba-ssm causal-conv1d` on a CUDA box."
            )
        self.norm1 = RMSNorm(d_model)
        self.attn = _LSAMamba3Attention(
            d_model, n_heads, d_kv_latent, d_state,
            dropout=dropout,
            mamba3_headdim=mamba3_headdim,
            mamba3_chunk_size=mamba3_chunk_size,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            qk_norm=qk_norm,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class _LSAMamba3Attention(nn.Module):
    """LSA attention with shared MLA latent + Mamba-3 SSM state.

    Mirrors aria.lsa.LSAAttention's structure but the SSM scan step is
    a Mamba-3 module taking the MLA latent as input. The Mamba-3 readout
    is projected to per-position K/V pairs that compete with local K/V
    through a single joint softmax (the ARIA fusion).
    """

    def __init__(self, d_model: int, n_heads: int, d_kv_latent: int,
                 d_state: int, dropout: float = 0.0,
                 mamba3_headdim: int = 64, mamba3_chunk_size: int = 16,
                 is_mimo: bool = False, mimo_rank: int = 4,
                 qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_kv_latent = d_kv_latent
        self.d_state = d_state
        self.qk_norm = qk_norm

        # Q projection
        self.w_q = nn.Linear(d_model, d_model, bias=False)

        # Shared MLA low-rank KV (the ARIA novelty)
        self.w_kv_down = nn.Linear(d_model, d_kv_latent, bias=False)
        self.w_k_up = nn.Linear(d_kv_latent, d_model, bias=False)
        self.w_v_up = nn.Linear(d_kv_latent, d_model, bias=False)

        # Mamba-3 SSM takes the MLA latent (shared input) and produces
        # a per-timestep readout of shape (B, T, d_kv_latent).
        self.mamba3 = _Mamba3(
            d_model=d_kv_latent,
            d_state=d_state,
            headdim=mamba3_headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=mamba3_chunk_size,
        )

        # Project Mamba-3 readout to per-position K, V "virtual tokens"
        self.w_state_k = nn.Linear(d_kv_latent, d_model, bias=False)
        self.w_state_v = nn.Linear(d_kv_latent, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.w_q(x).view(B, T, H, Dh)
        c_kv = self.w_kv_down(x)                                  # (B, T, d_kv_latent)
        k_local = self.w_k_up(c_kv).view(B, T, H, Dh)
        v_local = self.w_v_up(c_kv).view(B, T, H, Dh)
        q, k_local = apply_rope(q, k_local, rope_cos, rope_sin)

        # Mamba-3 consumes the shared latent and returns a per-timestep readout.
        states = self.mamba3(c_kv)                                # (B, T, d_kv_latent)

        k_state = self.w_state_k(states).view(B, T, H, Dh)
        v_state = self.w_state_v(states).view(B, T, H, Dh)

        if self.qk_norm:
            q = self.q_norm(q)
            k_local = self.k_norm(k_local)

        q = q.transpose(1, 2)
        k_local = k_local.transpose(1, 2)
        v_local = v_local.transpose(1, 2)
        k_state = k_state.transpose(1, 2)
        v_state = v_state.transpose(1, 2)

        scale = 1.0 / math.sqrt(Dh)
        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) * scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores_local = scores_local.masked_fill(
            ~causal_mask.view(1, 1, T, T), float("-inf")
        )
        scores_state = (q * k_state).sum(dim=-1, keepdim=True) * scale

        scores = torch.cat([scores_local, scores_state], dim=-1)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn_local = attn[..., :T]
        attn_state = attn[..., T:]

        out_local = torch.matmul(attn_local, v_local)
        out_state = attn_state * v_state
        out = out_local + out_state
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(out)


class LSAMamba3LanguageModel(nn.Module):
    """Decoder-only LM with LSA+Mamba-3 hybrid blocks and 3:1 interleave."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        d_kv_latent: int,
        d_state: int,
        max_seq_len: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        tie_weights: bool = True,
        interleave_ratio: int | None = 3,
        qk_norm: bool = False,
        mamba3_headdim: int = 64,
        mamba3_chunk_size: int = 16,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        **_unused,
    ) -> None:
        super().__init__()
        assert d_model == n_heads * d_head
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.interleave_ratio = interleave_ratio

        self.token_emb = nn.Embedding(vocab_size, d_model)

        blocks: list[nn.Module] = []
        for i in range(n_layers):
            is_full = (
                interleave_ratio is not None
                and interleave_ratio > 0
                and (i + 1) % (interleave_ratio + 1) == 0
            )
            if is_full:
                blocks.append(_FullAttentionBlock(d_model, n_heads, d_ff, dropout))
            else:
                blocks.append(
                    LSAMamba3Block(
                        d_model, n_heads, d_ff, d_kv_latent, d_state, dropout,
                        mamba3_headdim=mamba3_headdim,
                        mamba3_chunk_size=mamba3_chunk_size,
                        is_mimo=is_mimo,
                        mimo_rank=mimo_rank,
                        qk_norm=qk_norm,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        rope_cos, rope_sin = precompute_rope(d_head, max_seq_len, base=rope_base)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.token_emb(idx)
        rope_cos = self.rope_cos[:T]
        rope_sin = self.rope_sin[:T]
        for blk in self.blocks:
            x = blk(x, rope_cos, rope_sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss


__all__ = [
    "LSAMamba3Block",
    "LSAMamba3LanguageModel",
]
