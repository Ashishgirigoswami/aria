"""Layered State Attention (LSA) — ARIA's first architectural innovation.

Fuses two memory-efficient attention ideas into one operation:

1. **Low-rank KV compression** (DeepSeek-V2 MLA, arXiv:2405.04434):
   x -> c_kv (low-dim latent) -> K, V via up-projection.
   Reduces KV cache from O(N * d_model) to O(N * d_kv_latent).

2. **Selective SSM state** (Mamba-2, arXiv:2405.21060):
   A content-dependent recurrent state that compresses all past tokens
   into a fixed-size summary. State transitions A(x), B(x) are input-dependent.

Per-token attention then computes over BOTH:
- Recent exact KV (sliding window W — here we use full causal window)
- A "virtual" K/V reconstructed from the current SSM state

Effective long-range memory cost: O(W + d_state) instead of O(N).

In the Phase 0 prototype we use sequential (Python-loop) SSM scan, which is slow
but correct. A parallel scan implementation will replace it in Phase 1.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import RMSNorm, SwiGLU, precompute_rope, apply_rope
from .baseline import CausalAttention


@torch.jit.script
def ssm_scan_jit(A: torch.Tensor, Bg: torch.Tensor,
                 state_input: torch.Tensor) -> torch.Tensor:
    """Sequential causal SSM scan, JIT-compiled.

    Mathematically identical to the Python-loop version but compiled to
    a fused kernel, eliminating the per-step Python dispatch overhead
    that dominated wall time on the sequential version.

    Args:
        A: (B, T, d_state) per-token decay gate in [0, 1]
        Bg: (B, T, d_state) per-token write gate in [-1, 1]
        state_input: (B, T, d_state) what to write

    Returns:
        states: (B, T, d_state) causal per-position states
    """
    B, T, D = A.shape
    states = torch.empty(B, T, D, device=A.device, dtype=A.dtype)
    s = torch.zeros(B, D, device=A.device, dtype=A.dtype)
    for t in range(T):
        s = A[:, t] * s + Bg[:, t] * state_input[:, t]
        states[:, t] = s
    return states


class LSAAttention(nn.Module):
    """Layered State Attention block.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_kv_latent: Low-rank KV compression dim (<< d_model).
        d_state: SSM state dim.
        dropout: Attention dropout.
        window_size: If set, restrict local attention to a sliding window of
            W recent tokens (in addition to the causal mask). When None the
            layer uses full causal attention over up-projected K/V, matching
            Phase 0 behavior. Setting W makes the per-layer KV cache bounded
            to O(W * d_kv_latent + d_state) regardless of sequence length.
    """

    def __init__(self, d_model: int, n_heads: int,
                 d_kv_latent: int, d_state: int, dropout: float = 0.0,
                 window_size: int | None = None):
        super().__init__()
        self.window_size = window_size
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_kv_latent = d_kv_latent
        self.d_state = d_state

        # Q projection (standard)
        self.w_q = nn.Linear(d_model, d_model, bias=False)

        # Low-rank KV: x -> c_kv -> (K, V)
        self.w_kv_down = nn.Linear(d_model, d_kv_latent, bias=False)
        self.w_k_up = nn.Linear(d_kv_latent, d_model, bias=False)
        self.w_v_up = nn.Linear(d_kv_latent, d_model, bias=False)

        # Selective SSM gates (input-dependent)
        self.w_A = nn.Linear(d_model, d_state)  # decay gate (with bias)
        self.w_B = nn.Linear(d_model, d_state)  # write gate

        # What goes into the state (derived from compressed KV latent)
        self.w_in_state = nn.Linear(d_kv_latent, d_state, bias=False)

        # Project SSM state back into K, V "virtual tokens"
        self.w_state_k = nn.Linear(d_state, d_model, bias=False)
        self.w_state_v = nn.Linear(d_state, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Initialize SSM gates so the state is neutral at init:
        # A ~ 1 (preserve state), B ~ 0 (minimal new writes)
        nn.init.constant_(self.w_A.bias, 4.0)   # sigmoid(4) ~= 0.98
        nn.init.zeros_(self.w_B.bias)

    def ssm_scan(self, A: torch.Tensor, Bg: torch.Tensor,
                 state_input: torch.Tensor) -> torch.Tensor:
        """Sequential causal SSM scan.

        Delegates to the module-level ``ssm_scan_jit`` for the JIT-compiled
        fast path. Kept as a method for clarity and easy replacement if we
        later swap in a true parallel scan.
        """
        return ssm_scan_jit(A, Bg, state_input)

    def forward(self, x: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            rope_cos, rope_sin: (T, d_head/2) — precomputed RoPE tables (already sliced)

        Returns:
            out: (B, T, d_model)
        """
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # --- Q projection ---
        q = self.w_q(x).view(B, T, H, Dh)

        # --- Low-rank KV compression ---
        c_kv = self.w_kv_down(x)  # (B, T, d_kv_latent)

        # Reconstruct "local" K, V from compressed latent
        k_local = self.w_k_up(c_kv).view(B, T, H, Dh)
        v_local = self.w_v_up(c_kv).view(B, T, H, Dh)

        # Apply RoPE to Q and local K (state K stays positionless)
        q, k_local = apply_rope(q, k_local, rope_cos, rope_sin)

        # --- Selective SSM state update ---
        A = torch.sigmoid(self.w_A(x))     # (B, T, d_state), in [0, 1]
        Bg = torch.tanh(self.w_B(x))       # (B, T, d_state), in [-1, 1]
        state_input = self.w_in_state(c_kv)  # (B, T, d_state)

        states = self.ssm_scan(A, Bg, state_input)  # (B, T, d_state)

        # Project state to K, V virtual tokens
        k_state = self.w_state_k(states).view(B, T, H, Dh)
        v_state = self.w_state_v(states).view(B, T, H, Dh)

        # --- Attention over [local causal KV] + [per-position state KV] ---
        # Reshape to (B, H, T, Dh)
        q = q.transpose(1, 2)
        k_local = k_local.transpose(1, 2)
        v_local = v_local.transpose(1, 2)
        k_state = k_state.transpose(1, 2)
        v_state = v_state.transpose(1, 2)

        scale = 1.0 / math.sqrt(Dh)

        # Local causal attention scores: (B, H, T, T)
        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) * scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        if self.window_size is not None:
            # Sliding window: position i may only attend to j in (i - W, i].
            # Combined with the causal mask below this gives j in [max(0, i-W+1), i].
            window_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=-(self.window_size - 1),
            )
            causal_mask = causal_mask & window_mask
        scores_local = scores_local.masked_fill(~causal_mask.view(1, 1, T, T), float('-inf'))

        # State attention scores: each position attends to its OWN state.
        # state[i] already summarizes tokens 0..i causally, so this is causal by construction.
        # (B, H, T, 1)
        scores_state = (q * k_state).sum(dim=-1, keepdim=True) * scale

        # Concatenate and softmax jointly so local vs state compete for attention mass
        scores = torch.cat([scores_local, scores_state], dim=-1)  # (B, H, T, T+1)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_local = attn[..., :T]                # (B, H, T, T)
        attn_state = attn[..., T:]                # (B, H, T, 1)

        out_local = torch.matmul(attn_local, v_local)   # (B, H, T, Dh)
        out_state = attn_state * v_state                # (B, H, T, Dh) — broadcast over last dim

        out = out_local + out_state                     # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.w_o(out)


class LSABlock(nn.Module):
    """Pre-LN block: RMSNorm -> LSA -> residual -> RMSNorm -> SwiGLU -> residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 d_kv_latent: int, d_state: int, dropout: float = 0.0,
                 window_size: int | None = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = LSAAttention(d_model, n_heads, d_kv_latent, d_state, dropout,
                                 window_size=window_size)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class _FullAttentionBlock(nn.Module):
    """Vanilla transformer block used in interleaved slots of LSA v1."""

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


class LSALanguageModel(nn.Module):
    """Decoder-only language model using LSA blocks."""

    def __init__(self, *, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_head: int, d_ff: int,
                 d_kv_latent: int, d_state: int,
                 max_seq_len: int, dropout: float = 0.0,
                 rope_base: float = 10000.0, tie_weights: bool = True,
                 window_size: int | None = None,
                 interleave_ratio: int | None = None):
        super().__init__()
        assert d_model == n_heads * d_head, \
            f"d_model ({d_model}) must equal n_heads ({n_heads}) * d_head ({d_head})"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.window_size = window_size
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
                    LSABlock(d_model, n_heads, d_ff, d_kv_latent, d_state, dropout,
                             window_size=window_size)
                )
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        # Precompute RoPE tables for d_head, cached as non-persistent buffers
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
        """
        Args:
            input_ids: (B, T) long tensor
            targets: (B, T) long tensor or None. If provided, returns cross-entropy loss.

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar tensor or None
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        rope_cos = self.rope_cos[:T]
        rope_sin = self.rope_sin[:T]

        x = self.token_emb(input_ids)              # (B, T, D)
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)                   # (B, T, V)

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
        """Simple greedy / top-k sampling for qualitative eval."""
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
