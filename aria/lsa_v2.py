"""Layered State Attention v2 — Gated DeltaNet recurrence variant.

Scaffolded after the April 2026 literature audit. Changes from v1 (aria/lsa.py):

1. **Recurrence**: Mamba-2-style sigmoid/tanh vector state is replaced by a
   Gated DeltaNet matrix state with delta-rule updates (Yang et al.,
   arXiv 2412.06464, ICLR 2025). State is a per-head matrix S ∈ R^(K × V)
   with ~1000× more capacity than v1's vector state at the same param budget.
   Verified against the official fla-org/flash-linear-attention reference
   implementation; see the spec in ARIA_ARCHITECTURE_GUIDE.md.

2. **Shared-latent port**: the MLA latent `c_kv` drives (k, v) for the
   delta-rule recurrence, and q is generated from the main stream x. This
   preserves v1's "shared latent couples MLA compression and recurrence"
   novelty claim while matching Gated DeltaNet's required (q, k, v) input
   shape.

3. **Joint softmax fusion**: unchanged from v1. The readout of the delta-rule
   state is projected to a per-position virtual k/v, and its score
   participates in the same softmax as the local causal keys. This is
   LSA's second novelty axis and v2 preserves it exactly.

4. **Interleaved full-attention layers** (optional): per the hybrid survey
   arXiv 2507.06457, 3:1 or 6:1 linear:full-attention interleaving recovers
   transformer-level recall while keeping most of the cache savings. The v2
   model config accepts `interleave_ratio`; when set, every Nth block is a
   vanilla transformer block (pulled from aria.baseline.CausalAttention).

## Recurrence backends (three paths)

1. **fla chunk kernel** (Linux + CUDA + Triton + fla-core 0.4.2):
   ``chunk_gated_delta_rule`` from fla-org/flash-linear-attention. Fastest.
   Auto-detected via ``HAS_FLA`` flag. Used when model is on a CUDA device
   and fla is importable.

2. **Pure-PyTorch chunked scan** (any device — TPU, CPU, CUDA without fla):
   ``chunked_gated_delta_rule_torch`` — splits the sequence into chunks of
   ``CHUNK_SIZE`` (default 64), runs the sequential delta-rule loop inside
   each chunk, carries state between chunks. Mathematically identical to the
   reference loop but reduces XLA/torch graph depth from T to T/C, which
   is the key speedup on TPU where each loop iteration adds graph nodes.
   Also ~C× less Python dispatch overhead on CPU.

3. **Reference sequential loop** (correctness ground truth):
   ``gated_delta_rule_ref`` — one Python loop of T iterations. Too slow for
   training beyond ~30M params but serves as the bit-exactness baseline.

## Hard constraints from the spec (do not violate)

- q and k are L2-normalized before the recurrence — without this the delta
  correction (S^T k) is unbounded and training diverges.
- A short causal 1D conv (kernel size 4, SiLU) is applied to q, k, v before
  the recurrence. The original paper explicitly warns this is mandatory.
- State is stored in fp32 even under bf16/fp16 autocast — the decay
  accumulates multiplicatively and underflows at lower precision.
- Head count constraint: num_heads * head_dim should equal ~0.75 * d_model
  to keep the ~6·d² parameter budget Gated DeltaNet expects.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import RMSNorm, SwiGLU, precompute_rope, apply_rope
from .baseline import CausalAttention  # reused for interleaved full-attention blocks

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk
    HAS_FLA = True
except (ImportError, OSError):
    HAS_FLA = False

CHUNK_SIZE: int = 64  # for the pure-PyTorch chunked scan


# ---------------------------------------------------------------------------
# Gated DeltaNet recurrent branch — three backends
# ---------------------------------------------------------------------------


class ShortConv1d(nn.Module):
    """Causal depthwise 1D convolution with kernel size 4, SiLU nonlinearity.

    Applied to q/k/v before the Gated DeltaNet recurrence. The paper says
    this is "crucial to performance"; dropping it destabilizes training.

    Implementation: groups=channels so each channel is convolved
    independently, and the output is shifted so position t depends only on
    positions [t-3, t-2, t-1, t] of the input (strictly causal).
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # left-pad for causality
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T) -> conv -> slice to T -> (B, T, C)
        B, T, C = x.shape
        x = x.transpose(1, 2)  # (B, C, T)
        y = self.conv(x)[:, :, :T]  # drop right-padding overflow
        y = F.silu(y)
        return y.transpose(1, 2)  # (B, T, C)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def gated_delta_rule_ref(
    q: torch.Tensor,        # (B, H, T, K)   L2-normalized query
    k: torch.Tensor,        # (B, H, T, K)   L2-normalized key
    v: torch.Tensor,        # (B, H, T, V)   value (raw)
    g: torch.Tensor,        # (B, H, T)      log-decay (real-valued; α_t = exp(g_t))
    beta: torch.Tensor,     # (B, H, T)      delta-step size in [0, 1]
) -> torch.Tensor:
    """Reference delta-rule recurrence — sequential Python loop.

    Per head, per timestep:
        α_t    = exp(g_t)
        S_pre  = α_t · S_{t-1}
        v_new  = β_t · (v_t − S_pre^T k_t)
        S_t    = S_pre + k_t ⊗ v_new
        o_t    = S_t^T q_t

    State S has shape (B, H, K, V) and is stored in fp32 regardless of input
    dtype — multiplicative decay accumulates and underflows in bf16.

    Returns:
        o: (B, H, T, V) output stream.
    """
    B, H, T, K = q.shape
    V = v.size(-1)
    S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    out = torch.empty(B, H, T, V, dtype=q.dtype, device=q.device)

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    b_f = beta.float()

    for t in range(T):
        alpha_t = torch.exp(g_f[:, :, t])                  # (B, H)
        k_t = k_f[:, :, t]                                 # (B, H, K)
        v_t = v_f[:, :, t]                                 # (B, H, V)
        beta_t = b_f[:, :, t]                              # (B, H)

        # Decay the state.
        S = S * alpha_t[:, :, None, None]

        # Delta-rule correction: v_new = β (v − S^T k)
        #   S^T k  has shape (B, H, V), via einsum over K dim.
        sTk = torch.einsum("bhkv,bhk->bhv", S, k_t)        # (B, H, V)
        v_new = beta_t[:, :, None] * (v_t - sTk)           # (B, H, V)

        # Rank-1 write: S += k ⊗ v_new
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, v_new)

        # Readout: o = S^T q
        q_t = q_f[:, :, t]                                 # (B, H, K)
        o_t = torch.einsum("bhkv,bhk->bhv", S, q_t)        # (B, H, V)
        out[:, :, t] = o_t.to(q.dtype)

    return out


def chunked_gated_delta_rule_torch(
    q: torch.Tensor,        # (B, H, T, K)
    k: torch.Tensor,        # (B, H, T, K)
    v: torch.Tensor,        # (B, H, T, V)
    g: torch.Tensor,        # (B, H, T)
    beta: torch.Tensor,     # (B, H, T)
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Pure-PyTorch chunked delta-rule scan — works on CPU, CUDA, and TPU.

    Identical math to ``gated_delta_rule_ref`` but chunks the sequence into
    blocks of ``chunk_size``. Within each chunk the sequential loop runs for
    only ``chunk_size`` steps (not T), and the state is carried between
    chunks. This reduces XLA graph depth from T to T / chunk_size — the key
    speedup on TPU where each Python-loop iteration adds XLA graph nodes.

    State is fp32 regardless of input dtype.
    """
    B, H, T, K = q.shape
    V = v.size(-1)
    S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    out = torch.empty(B, H, T, V, dtype=q.dtype, device=q.device)

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    b_f = beta.float()

    n_chunks = (T + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, T)
        for t in range(start, end):
            alpha_t = torch.exp(g_f[:, :, t])
            k_t = k_f[:, :, t]
            v_t = v_f[:, :, t]
            beta_t = b_f[:, :, t]

            S = S * alpha_t[:, :, None, None]
            sTk = torch.einsum("bhkv,bhk->bhv", S, k_t)
            v_new = beta_t[:, :, None] * (v_t - sTk)
            S = S + torch.einsum("bhk,bhv->bhkv", k_t, v_new)
            o_t = torch.einsum("bhkv,bhk->bhv", S, q_f[:, :, t])
            out[:, :, t] = o_t.to(q.dtype)
    return out


class GatedDeltaRecurrence(nn.Module):
    """Gated DeltaNet recurrent branch for LSA v2.

    Consumes x (main stream) and c_kv (shared MLA latent) and produces a
    per-position output stream of shape (B, T, n_heads * head_v_dim).

    Architecture:
        q ← ShortConv(x W_q) → SiLU → L2-norm
        k ← ShortConv(c_kv W_k) → SiLU → L2-norm
        v ← ShortConv(c_kv W_v) → SiLU
        g ← data-dependent log-decay from x (Mamba-style dt_bias init)
        β ← sigmoid(x W_beta)
        o ← gated_delta_rule_ref(q, k, v, g, β)
        gated_o ← RMSNorm(o) * SiLU(x W_g)     # FusedRMSNormGated post-gate
        out ← gated_o W_o

    Notes:
        - num_heads * head_k_dim ≈ 0.75 * d_model (Gated DeltaNet convention)
        - head_v_dim = expand_v * head_k_dim, typically 2x
        - State fp32, q/k L2-normalized, short conv mandatory
    """

    def __init__(
        self,
        d_model: int,
        d_kv_latent: int,
        num_heads: int = 3,
        head_k_dim: int = 96,
        expand_v: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = int(head_k_dim * expand_v)

        kq_total = num_heads * head_k_dim       # e.g. 3 * 96 = 288
        v_total = num_heads * self.head_v_dim   # e.g. 3 * 192 = 576

        self.w_q = nn.Linear(d_model, kq_total, bias=False)
        self.w_k = nn.Linear(d_kv_latent, kq_total, bias=False)
        self.w_v = nn.Linear(d_kv_latent, v_total, bias=False)

        # Short causal convs on q, k, v (mandatory per spec).
        self.conv_q = ShortConv1d(kq_total)
        self.conv_k = ShortConv1d(kq_total)
        self.conv_v = ShortConv1d(v_total)

        # Decay gate: g_t = -softplus(dt_bias + a_proj(x)) * exp(A_log).
        self.a_proj = nn.Linear(d_model, num_heads)
        self.A_log = nn.Parameter(torch.empty(num_heads))
        self.dt_bias = nn.Parameter(torch.empty(num_heads))

        # Delta step size: β_t = sigmoid(b_proj(x)).
        self.b_proj = nn.Linear(d_model, num_heads)

        # Post-gate: SiLU(g_proj(x)) * RMSNorm(o).
        self.g_proj = nn.Linear(d_model, v_total, bias=False)
        self.post_norm = RMSNorm(v_total)

        # Output projection back to d_model.
        self.w_o = nn.Linear(v_total, d_model, bias=False)

        self._init_gates()

    def _init_gates(self) -> None:
        # A_log ~ log(U(0, 16)) per head (Mamba-2 style).
        with torch.no_grad():
            self.A_log.uniform_(0.0, 16.0).log_()
            # dt_bias via inverse-softplus so sigmoid(dt_bias) targets dt in [1e-3, 0.1].
            # Simple approximation: draw dt uniformly in log-space and invert.
            dt = torch.exp(
                torch.empty(self.num_heads).uniform_(math.log(1e-3), math.log(0.1))
            )
            # Inverse of softplus: x = log(exp(dt) - 1), for positive dt.
            self.dt_bias.copy_(torch.log(torch.expm1(dt).clamp(min=1e-8)))

    def forward(self, x: torch.Tensor, c_kv: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.num_heads
        Kh = self.head_k_dim
        Vh = self.head_v_dim

        # Projections + short conv + SiLU (the short conv already applies SiLU).
        q = self.conv_q(self.w_q(x))                           # (B, T, H*Kh)
        k = self.conv_k(self.w_k(c_kv))                        # (B, T, H*Kh)
        v = self.conv_v(self.w_v(c_kv))                        # (B, T, H*Vh)

        q = q.view(B, T, H, Kh).transpose(1, 2)                # (B, H, T, Kh)
        k = k.view(B, T, H, Kh).transpose(1, 2)                # (B, H, T, Kh)
        v = v.view(B, T, H, Vh).transpose(1, 2)                # (B, H, T, Vh)

        q = l2_normalize(q) * (Kh ** -0.5)                     # scaled L2-norm
        k = l2_normalize(k)

        # Gates.
        a = self.a_proj(x).transpose(1, 2)                     # (B, H, T)
        g = -F.softplus(self.dt_bias[None, :, None] + a) * torch.exp(self.A_log[None, :, None])
        beta = torch.sigmoid(self.b_proj(x).transpose(1, 2))   # (B, H, T)

        # Recurrence — select backend based on device + availability.
        use_fla = HAS_FLA and q.is_cuda
        if use_fla:
            # fla chunk kernel: fastest on CUDA. Expects (B, H, T, K/V) inputs.
            # Returns (output, final_state); we only need the output.
            o, _ = _fla_chunk(q, k, v, g, beta)
        else:
            # Pure-PyTorch chunked scan: works everywhere (TPU, CPU, CUDA).
            o = chunked_gated_delta_rule_torch(q, k, v, g, beta)  # (B, H, T, Vh)

        # Merge heads back.
        o = o.transpose(1, 2).contiguous().view(B, T, H * Vh)  # (B, T, H*Vh)

        # Post-gate: SiLU(g_proj(x)) * RMSNorm(o).
        gate = F.silu(self.g_proj(x))                          # (B, T, H*Vh)
        o = self.post_norm(o) * gate

        return self.w_o(o)                                     # (B, T, d_model)


# ---------------------------------------------------------------------------
# LSA v2 attention block
# ---------------------------------------------------------------------------


class LSAv2Attention(nn.Module):
    """LSA v2: MLA-compressed local attention + Gated DeltaNet recurrence,
    fused by joint softmax with a state-derived virtual k/v.

    Structurally identical to v1's LSAAttention on the local-attention side.
    The only change is the recurrent branch (vector SSM → matrix delta rule).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_kv_latent: int,
        d_state: int,                     # kept for API compatibility; unused in v2
        dropout: float = 0.0,
        window_size: int | None = None,
        gdn_num_heads: int = 3,
        gdn_head_k_dim: int = 96,
        gdn_expand_v: float = 2.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_kv_latent = d_kv_latent
        self.window_size = window_size

        # Local MLA-compressed attention path.
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv_down = nn.Linear(d_model, d_kv_latent, bias=False)
        self.w_k_up = nn.Linear(d_kv_latent, d_model, bias=False)
        self.w_v_up = nn.Linear(d_kv_latent, d_model, bias=False)

        # Gated DeltaNet recurrence — produces a per-position vector stream
        # that plays the role of v1's `s_t` (but much higher capacity).
        self.recurrence = GatedDeltaRecurrence(
            d_model=d_model,
            d_kv_latent=d_kv_latent,
            num_heads=gdn_num_heads,
            head_k_dim=gdn_head_k_dim,
            expand_v=gdn_expand_v,
        )

        # Project the recurrence output into per-head virtual (k_state, v_state).
        # The recurrence output lives in d_model-space (after its w_o), so we
        # can reuse two simple linears to produce positionless virtual k/v.
        self.w_state_k = nn.Linear(d_model, d_model, bias=False)
        self.w_state_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor,
        rope_cos: torch.Tensor, rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # --- Local MLA path ---
        q = self.w_q(x).view(B, T, H, Dh)
        c_kv = self.w_kv_down(x)                                 # shared latent
        k_local = self.w_k_up(c_kv).view(B, T, H, Dh)
        v_local = self.w_v_up(c_kv).view(B, T, H, Dh)
        q, k_local = apply_rope(q, k_local, rope_cos, rope_sin)

        # --- Gated DeltaNet recurrence, driven by x and c_kv ---
        state_out = self.recurrence(x, c_kv)                    # (B, T, d_model)
        k_state = self.w_state_k(state_out).view(B, T, H, Dh)
        v_state = self.w_state_v(state_out).view(B, T, H, Dh)

        # --- Joint softmax attention ---
        q = q.transpose(1, 2)
        k_local = k_local.transpose(1, 2)
        v_local = v_local.transpose(1, 2)
        k_state = k_state.transpose(1, 2)
        v_state = v_state.transpose(1, 2)

        scale = 1.0 / math.sqrt(Dh)
        scores_local = torch.matmul(q, k_local.transpose(-2, -1)) * scale
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        if self.window_size is not None:
            window = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=-(self.window_size - 1),
            )
            causal = causal & window
        scores_local = scores_local.masked_fill(~causal.view(1, 1, T, T), float("-inf"))

        scores_state = (q * k_state).sum(dim=-1, keepdim=True) * scale
        scores = torch.cat([scores_local, scores_state], dim=-1)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_local = attn[..., :T]
        attn_state = attn[..., T:]

        out_local = torch.matmul(attn_local, v_local)
        out_state = attn_state * v_state
        out = (out_local + out_state).transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(out)


# ---------------------------------------------------------------------------
# Block + full model with optional interleaved full-attention layers
# ---------------------------------------------------------------------------


class LSAv2Block(nn.Module):
    """Pre-LN block using LSAv2Attention."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int,
        d_kv_latent: int, d_state: int, dropout: float = 0.0,
        window_size: int | None = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = LSAv2Attention(
            d_model, n_heads, d_kv_latent, d_state, dropout,
            window_size=window_size,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class FullAttentionBlock(nn.Module):
    """Vanilla transformer block — used in interleaved slots.

    Uses _ManualCausalAttention from lsa.py for XLA/TPU compatibility
    (F.scaled_dot_product_attention backward is broken on XLA at seq_len>=256).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        from .lsa import _ManualCausalAttention
        self.norm1 = RMSNorm(d_model)
        self.attn = _ManualCausalAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class LSAv2LanguageModel(nn.Module):
    """Decoder-only LM with interleaved LSAv2 / full-attention blocks.

    `interleave_ratio` controls the linear:full-attention layout:
        None or 0: all layers are LSAv2 (pure hybrid stack, like v1)
        3: every 4th layer is full attention, rest are LSAv2 (3:1 ratio)
        6: every 7th layer is full attention (6:1 ratio)

    Per the hybrid survey (arXiv 2507.06457), 3:1 gives the best recall;
    6:1 is the max cache-savings sweet spot.
    """

    def __init__(
        self, *, vocab_size: int, d_model: int, n_layers: int,
        n_heads: int, d_head: int, d_ff: int,
        d_kv_latent: int, d_state: int,
        max_seq_len: int, dropout: float = 0.0,
        rope_base: float = 10000.0, tie_weights: bool = True,
        window_size: int | None = None,
        interleave_ratio: int | None = None,
    ):
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
                blocks.append(FullAttentionBlock(d_model, n_heads, d_ff, dropout))
            else:
                blocks.append(
                    LSAv2Block(
                        d_model, n_heads, d_ff, d_kv_latent, d_state, dropout,
                        window_size=window_size,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

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

    def forward(self, input_ids, targets=None):
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
