"""Run standard LM evaluation harness on an ARIA checkpoint.

Wraps EleutherAI's lm-evaluation-harness so results are directly comparable
to Pythia, OLMo, Mamba, Qwen, and any other model published on the HF leaderboard.

Reported benchmarks (0-shot unless noted):
  - HellaSwag        — commonsense reasoning
  - ARC-Easy         — grade-school science
  - ARC-Challenge    — harder science
  - PIQA             — physical reasoning
  - WinoGrande       — coreference / commonsense
  - OpenBookQA       — science recall
  - LAMBADA (OpenAI) — last-word prediction (acc + ppl)
  - BoolQ            — yes/no QA
  - WikiText-103 ppl — standard pretraining metric (5-shot optional)

Expected 150M range (random baseline shown in parens):
  HellaSwag:   ~30-35% (25%)
  ARC-E:       ~40-50% (25%)
  ARC-C:       ~22-26% (25%)
  PIQA:        ~60-65% (50%)
  WinoGrande:  ~50-52% (50%)
  LAMBADA acc: ~25-40% (0%)

Usage:
    pip install lm-eval>=0.4
    python scripts/eval_harness.py \\
        --ckpt checkpoints/aria_v2_150m_t256/best.pt \\
        --config configs/aria_v2_150m_t256.yaml \\
        --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai \\
        --output eval_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Patch SSM scan for XLA before importing the model — same trick the trainer uses.
try:
    import torch_xla
    HAS_XLA = True
    from aria import lsa_xla  # noqa: F401  (side-effect: monkey-patches SSM scan)
except ImportError:
    HAS_XLA = False

from aria.baseline import BaselineLanguageModel
from aria.lsa import LSALanguageModel
from aria.lsa_v2 import LSAv2LanguageModel

# Mamba variants require CUDA (mamba-ssm kernels). Load lazily so TPU/CPU eval
# of LSA v1 still works without mamba-ssm installed.
try:
    from aria.mamba3_model import Mamba3LanguageModel
except Exception:
    Mamba3LanguageModel = None
try:
    from aria.lsa_mamba3 import LSAMamba3LanguageModel
except Exception:
    LSAMamba3LanguageModel = None


MODEL_REGISTRY = {
    "lsa": LSALanguageModel,
    "lsa_v2": LSAv2LanguageModel,
    "baseline": BaselineLanguageModel,
}
if Mamba3LanguageModel is not None:
    MODEL_REGISTRY["mamba3"] = Mamba3LanguageModel
if LSAMamba3LanguageModel is not None:
    MODEL_REGISTRY["lsa_mamba3"] = LSAMamba3LanguageModel


def load_model(ckpt_path: str, config_path: str, device) -> torch.nn.Module:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model_cfg["vocab_size"] = model_cfg.get("vocab_size", 50257)
    model = MODEL_REGISTRY[model_name](**model_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    # Re-tie weights post-.to() — XLA can break tied params.
    if hasattr(model, "lm_head") and hasattr(model, "token_emb"):
        if model.lm_head.weight is not model.token_emb.weight:
            model.lm_head.weight = model.token_emb.weight
    return model


class ARIALMWrapper:
    """Minimal lm-eval-harness LM wrapper for the ARIA family.

    Implements only the two methods lm-eval actually calls for the standard
    multiple-choice benchmarks: loglikelihood and loglikelihood_rolling.
    Greedy generation is not required for the reported tasks.
    """

    def __init__(self, model: torch.nn.Module, tokenizer_name: str = "gpt2",
                 max_length: int = 2048, device: str = "cuda",
                 batch_size: int = 1, pad_to_max: bool = False):
        import tiktoken
        self.model = model
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.max_length = max_length
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.eot_token_id = self.tokenizer.eot_token
        # On XLA devices we MUST pad to a fixed length to avoid recompile per shape.
        is_xla = HAS_XLA and str(device).startswith("xla")
        self.pad_to_max = pad_to_max or is_xla
        if self.pad_to_max:
            print(f"  pad_to_max enabled (input length fixed at {max_length}) — "
                  "single XLA compile, post-process on CPU")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def tok_encode(self, s: str) -> list[int]:
        return self.tokenizer.encode_ordinary(s)

    def tok_decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def _score_tokens(self, input_ids: torch.Tensor,
                      target_ids: torch.Tensor) -> torch.Tensor:
        """Return per-sequence log P(target | context) summed over target tokens."""
        logits, _ = self.model(input_ids)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        # Gather log P at target positions
        gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return gathered

    def _pad_ids(self, ids: list[int]) -> list[int]:
        """Truncate-or-pad to exactly max_length tokens. Pad with EOT."""
        ids = ids[-self.max_length:]
        return ids + [self.eot_token_id] * (self.max_length - len(ids))

    @torch.no_grad()
    def loglikelihood(self, requests: list[tuple[str, str]]
                      ) -> list[tuple[float, bool]]:
        """requests: list of (context, continuation). Return list of (logp, is_greedy).

        When ``self.pad_to_max`` is True, requests are batched ``self.batch_size``
        at a time into a single ``(B, max_length-1)`` forward pass — dramatically
        faster on XLA, since the shape is fixed and the compiled graph is reused.
        """
        results: list[tuple[float, bool]] = []
        is_xla = HAS_XLA and str(self.device).startswith("xla")
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover - optional dep
            tqdm = None

        # Fast path: batched forward for the pad_to_max case.
        if self.pad_to_max and self.batch_size > 1:
            pbar = tqdm(total=len(requests), desc="loglikelihood", leave=False) if tqdm else None
            for start in range(0, len(requests), self.batch_size):
                batch = requests[start : start + self.batch_size]
                padded_batch: list[list[int]] = []
                meta: list[tuple[int, int, list[int]]] = []  # (n, n_cont, cont_ids)
                for ctx, cont in batch:
                    ctx_ids = self.tok_encode(ctx) if ctx else [self.eot_token_id]
                    cont_ids = self.tok_encode(cont)
                    all_ids = (ctx_ids + cont_ids)[-self.max_length:]
                    n = len(all_ids)
                    n_cont = min(len(cont_ids), n - 1)
                    padded_batch.append(self._pad_ids(all_ids)[:-1])
                    meta.append((n, n_cont, all_ids[n - n_cont : n]))

                input_ids = torch.tensor(padded_batch, device=self.device)
                logits, _ = self.model(input_ids)
                if is_xla:
                    torch_xla.sync()
                logits_cpu = logits.float().cpu()
                for i, (n, n_cont, cont_target_ids) in enumerate(meta):
                    row = logits_cpu[i, n - 1 - n_cont : n - 1]
                    cont_log_probs = torch.log_softmax(row, dim=-1)
                    cont_targets = torch.tensor(cont_target_ids, dtype=torch.long)
                    gathered = cont_log_probs.gather(
                        -1, cont_targets.unsqueeze(-1)
                    ).squeeze(-1)
                    total_logp = float(gathered.sum())
                    is_greedy = bool((cont_log_probs.argmax(-1) == cont_targets).all())
                    results.append((total_logp, is_greedy))
                if pbar is not None:
                    pbar.update(len(batch))
            if pbar is not None:
                pbar.close()
            return results

        # Original single-item path (unbatched, or non-pad_to_max).
        iterator = tqdm(requests, desc="loglikelihood", leave=False) if tqdm else requests

        for ctx, cont in iterator:
            ctx_ids = self.tok_encode(ctx) if ctx else [self.eot_token_id]
            cont_ids = self.tok_encode(cont)
            all_ids = (ctx_ids + cont_ids)[-self.max_length:]
            n = len(all_ids)
            n_cont = min(len(cont_ids), n - 1)

            if self.pad_to_max:
                # Always-fixed shape (1, max_length-1) → single XLA compile
                padded = self._pad_ids(all_ids)
                input_ids = torch.tensor([padded[:-1]], device=self.device)
                logits, _ = self.model(input_ids)
                if is_xla:
                    torch_xla.sync()
                # Move to CPU, then slice on CPU (no recompile)
                logits_cpu = logits.float().cpu()
                # Real positions in padded are 0..n-2 in the input/target.
                # Continuation occupies last n_cont positions → indices [n-1-n_cont:n-1] of target
                cont_log_probs = torch.log_softmax(
                    logits_cpu[0, n - 1 - n_cont : n - 1], dim=-1
                )
                cont_targets = torch.tensor(all_ids[n - n_cont : n], dtype=torch.long)
            else:
                input_ids = torch.tensor([all_ids[:-1]], device=self.device)
                target_ids = torch.tensor([all_ids[1:]], device=self.device)
                logits, _ = self.model(input_ids)
                cont_log_probs = torch.log_softmax(
                    logits[0, -n_cont:].float(), dim=-1
                ).cpu()
                cont_targets = target_ids[0, -n_cont:].cpu()

            gathered = cont_log_probs.gather(
                -1, cont_targets.unsqueeze(-1)
            ).squeeze(-1)
            total_logp = float(gathered.sum())
            is_greedy = bool((cont_log_probs.argmax(-1) == cont_targets).all())
            results.append((total_logp, is_greedy))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        """Rolling log-likelihood over a full string (for LAMBADA, wikitext ppl).

        For texts longer than ``max_length``, evaluates via a sliding window
        with stride = ``max_length // 2``: each position from 0 to len(ids)-1
        is scored at most once, in a context that includes at least the
        previous ``max_length // 2`` tokens. This matches the lm-eval-harness
        reference semantics for rolling perplexity (see
        EleutherAI/lm-evaluation-harness#huggingface_model.py).

        Short texts (``len(ids) <= max_length``) are scored in a single
        forward pass on all valid positions (tokens 1..n-1 predicted from
        tokens 0..n-2).
        """
        results: list[float] = []
        is_xla = HAS_XLA and str(self.device).startswith("xla")
        try:
            from tqdm import tqdm
            iterator = tqdm(requests, desc="loglikelihood_rolling", leave=False)
        except ImportError:
            iterator = requests

        max_len = self.max_length
        stride = max(1, max_len // 2)

        for text in iterator:
            ids = self.tok_encode(text)
            if len(ids) < 2:
                results.append(0.0)
                continue
            n_full = len(ids)

            total_logp = 0.0
            # Sliding window. For each window [start:start+max_len], we
            # score targets from position `scored_up_to` (one past the last
            # target already counted in the previous window) through
            # window_end - 1, using context `ids[start : start+max_len-1]`.
            start = 0
            scored_up_to = 1  # targets are ids[1:], start scoring at pos 1

            while True:
                window_end = min(start + max_len, n_full)
                # Positions in this window we still need to score.
                score_from = max(scored_up_to, start + 1)
                if score_from >= window_end:
                    break
                window_ids = ids[start:window_end]
                n_win = len(window_ids)

                if self.pad_to_max:
                    padded = self._pad_ids(window_ids)
                    input_ids = torch.tensor([padded[:-1]], device=self.device)
                else:
                    input_ids = torch.tensor([window_ids[:-1]], device=self.device)
                logits, _ = self.model(input_ids)
                if is_xla:
                    torch_xla.sync()
                logits_cpu = logits.float().cpu()

                # Map absolute positions to window-local target indices.
                tgt_start_local = score_from - start  # >= 1
                tgt_end_local = n_win                  # exclusive
                # input positions predicting these targets:
                ctx_start = tgt_start_local - 1
                ctx_end = tgt_end_local - 1
                window_log_probs = torch.log_softmax(
                    logits_cpu[0, ctx_start:ctx_end], dim=-1
                )
                window_targets = torch.tensor(
                    ids[score_from:start + tgt_end_local], dtype=torch.long
                )
                gathered = window_log_probs.gather(
                    -1, window_targets.unsqueeze(-1)
                ).squeeze(-1)
                total_logp += float(gathered.sum())

                scored_up_to = start + tgt_end_local
                if window_end >= n_full:
                    break
                start += stride

            results.append(total_logp)
        return results

    @torch.no_grad()
    def _loglikelihood_rolling_legacy(self, requests: list[str]) -> list[float]:
        """Legacy truncating path — kept only so existing saved LAMBADA/ppl
        JSONs can be regenerated bit-exact if needed. New code should call
        ``loglikelihood_rolling`` above."""
        results: list[float] = []
        is_xla = HAS_XLA and str(self.device).startswith("xla")
        try:
            from tqdm import tqdm
            iterator = tqdm(requests, desc="loglikelihood_rolling_legacy", leave=False)
        except ImportError:
            iterator = requests

        for text in iterator:
            ids = self.tok_encode(text)
            if len(ids) < 2:
                results.append(0.0)
                continue
            ids = ids[:self.max_length]
            n = len(ids)

            if self.pad_to_max:
                padded = self._pad_ids(ids)
                input_ids = torch.tensor([padded[:-1]], device=self.device)
                logits, _ = self.model(input_ids)
                if is_xla:
                    torch_xla.sync()
                logits_cpu = logits.float().cpu()
                # Real positions are 0..n-2; targets are ids[1:n]
                real_log_probs = torch.log_softmax(
                    logits_cpu[0, : n - 1], dim=-1
                )
                real_targets = torch.tensor(ids[1:n], dtype=torch.long)
                gathered = real_log_probs.gather(
                    -1, real_targets.unsqueeze(-1)
                ).squeeze(-1)
                results.append(float(gathered.sum()))
            else:
                input_ids = torch.tensor([ids[:-1]], device=self.device)
                target_ids = torch.tensor([ids[1:]], device=self.device)
                logits, _ = self.model(input_ids)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                gathered = log_probs.gather(
                    -1, target_ids.unsqueeze(-1)
                ).squeeze(-1)
                results.append(float(gathered.sum().cpu()))
        return results


def run_lm_eval_harness(wrapper: ARIALMWrapper, tasks: list[str],
                        num_fewshot: int = 0, limit: int | None = None
                        ) -> dict[str, Any]:
    """Invoke lm-eval-harness directly via its Python API."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.api.model import LM
    except ImportError as e:
        raise RuntimeError(
            "lm-eval not installed. Run: pip install lm-eval>=0.4"
        ) from e

    class _LMAdapter(LM):
        def __init__(self, inner: ARIALMWrapper):
            super().__init__()
            self.inner = inner

        def loglikelihood(self, requests):  # type: ignore[override]
            # requests: list of Instance with args = (context, continuation)
            pairs = [req.args for req in requests]
            return self.inner.loglikelihood(pairs)

        def loglikelihood_rolling(self, requests):  # type: ignore[override]
            texts = [req.args[0] for req in requests]
            return self.inner.loglikelihood_rolling(texts)

        def generate_until(self, requests):  # type: ignore[override]
            raise NotImplementedError(
                "ARIA eval wrapper does not support generation tasks; "
                "use only multiple-choice/loglikelihood tasks."
            )

    adapter = _LMAdapter(wrapper)
    results = simple_evaluate(
        model=adapter,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,piqa,"
                        "winogrande,openbookqa,lambada_openai")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit docs per task (for smoke testing)")
    parser.add_argument("--device", default="auto",
                        help="auto | cuda | cpu | xla")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Default 256 = ARIA training context. Required for XLA.")
    parser.add_argument("--pad-to-max", action="store_true",
                        help="Pad all inputs to max_length (auto-on for XLA)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch N loglikelihood requests per forward pass "
                        "(only effective with --pad-to-max). 16 is a good start on v4.")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    if args.device == "auto":
        if HAS_XLA:
            device = torch_xla.device()
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif args.device == "xla":
        if not HAS_XLA:
            raise RuntimeError("--device xla requested but torch_xla not installed")
        device = torch_xla.device()
    else:
        device = torch.device(args.device)

    print(f"Loading model from {args.ckpt} on {device}...")
    model = load_model(args.ckpt, args.config, device)
    if HAS_XLA and str(device).startswith("xla"):
        torch_xla.sync()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    wrapper = ARIALMWrapper(model, max_length=args.max_length, device=device,
                            pad_to_max=args.pad_to_max, batch_size=args.batch_size)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"Running tasks: {tasks} (num_fewshot={args.num_fewshot})")

    results = run_lm_eval_harness(wrapper, tasks,
                                  num_fewshot=args.num_fewshot,
                                  limit=args.limit)

    # Compact summary: one line per task with headline metric
    summary = {
        "n_params": n_params,
        "checkpoint": args.ckpt,
        "num_fewshot": args.num_fewshot,
        "tasks": {},
    }
    for task, res in results.get("results", {}).items():
        headline = {}
        for key, val in res.items():
            if isinstance(val, (int, float)) and not key.endswith("_stderr"):
                headline[key] = val
        summary["tasks"][task] = headline

    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"\n=== Summary ({n_params/1e6:.1f}M params) ===")
    for task, metrics in summary["tasks"].items():
        main_metric = next(iter(metrics.items()), ("-", 0))
        print(f"  {task:25s} {main_metric[0]:20s} {main_metric[1]:.4f}")
    print(f"\nFull results: {args.output}")


if __name__ == "__main__":
    main()
