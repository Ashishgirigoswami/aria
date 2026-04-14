"""Matched baseline transformer for fair comparison with LSA.

Identical everything except the attention mechanism:
- Same d_model, n_layers, n_heads, d_head, d_ff
- Same RMSNorm + SwiGLU + RoPE
- Same weight init, same training config
- Uses standard causal multi-head attention (full KV cache)

Any perplexity gap between this and LSA at matched parameter count is attributable
to the LSA attention mechanism, which is exactly what we want to measure.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import RMSNorm, SwiGLU, precompute_rope, apply_rope


class CausalAttention(nn.Module):
    """Standard causal multi-head self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.w_q(x).view(B, T, H, Dh)
        k = self.w_k(x).view(B, T, H, Dh)
        v = self.w_v(x).view(B, T, H, Dh)

        q, k = apply_rope(q, k, rope_cos, rope_sin)

        q = q.transpose(1, 2)   # (B, H, T, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch fused scaled-dot-product attention when available.
        # It handles causal masking efficiently.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(out)


class TransformerBlock(nn.Module):
    """Pre-LN block: RMSNorm -> causal attn -> residual -> RMSNorm -> SwiGLU -> residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class BaselineLanguageModel(nn.Module):
    """Matched baseline decoder-only transformer."""

    def __init__(self, *, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_head: int, d_ff: int,
                 max_seq_len: int, dropout: float = 0.0,
                 rope_base: float = 10000.0, tie_weights: bool = True,
                 # Accept (and ignore) LSA-only kwargs so the same config schema loads:
                 d_kv_latent: int | None = None, d_state: int | None = None):
        super().__init__()
        assert d_model == n_heads * d_head

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        cos, sin = precompute_rope(d_head, max_seq_len, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        assert T <= self.max_seq_len

        rope_cos = self.rope_cos[:T]
        rope_sin = self.rope_sin[:T]

        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ids = input_ids[:, -self.max_seq_len:]
            logits, _ = self(ids)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
