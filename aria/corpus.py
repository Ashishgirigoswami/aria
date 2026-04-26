"""Large-scale corpus preprocessing + mmap shard sampling for ARIA.

This module keeps the small Phase-0 dataset helpers in ``aria.data`` intact
while adding a production path for 50B-100B token pretraining runs:

* Streaming HF dataset ingestion
* Exact cross-source deduplication backed by sqlite
* Buffered global document shuffle across sources
* GPT-NeoX / GPT-2 tokenization into mmap-friendly ``.bin`` shards
* TPU-friendly fixed-shape random window sampling from shard manifests
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import BPETokenizer


def _nested_get(row: dict[str, Any], path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def normalize_document(text: str) -> str:
    """Normalize text before deduplication."""
    return " ".join(text.strip().split()).lower()


def infer_text(row: dict[str, Any], text_fields: list[str] | None = None) -> str:
    """Extract a text payload from a streamed dataset row."""
    if text_fields:
        pieces = []
        for field in text_fields:
            value = _nested_get(row, field)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
        return "\n\n".join(pieces)

    for key in ("text", "content", "body", "document"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    title = row.get("title")
    body = row.get("body")
    answer = row.get("answer")
    parts = [
        value.strip()
        for value in (title, body, answer)
        if isinstance(value, str) and value.strip()
    ]
    return "\n\n".join(parts)


class SqliteDeduper:
    """Exact document deduper that scales past RAM-sized hash sets."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_hashes (hash TEXT PRIMARY KEY)"
        )
        self.conn.commit()

    def add_if_new(self, text: str) -> bool:
        digest = hashlib.sha1(normalize_document(text).encode("utf-8")).hexdigest()
        try:
            self.conn.execute(
                "INSERT INTO seen_hashes(hash) VALUES (?)",
                (digest,),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


@dataclass
class SourceSpec:
    name: str
    repo_id: str
    split: str = "train"
    subset: str | None = None
    weight: float = 1.0
    text_fields: list[str] | None = None
    min_chars: int = 32
    shuffle: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceSpec":
        return cls(
            name=raw["name"],
            repo_id=raw["repo_id"],
            split=raw.get("split", "train"),
            subset=raw.get("subset"),
            weight=float(raw.get("weight", 1.0)),
            text_fields=list(raw.get("text_fields", [])) or None,
            min_chars=int(raw.get("min_chars", 32)),
            shuffle=bool(raw.get("shuffle", True)),
        )


def _tokenizer_kind(name: str) -> str:
    return "neox" if name in ("neox", "gpt-neox", "gpt-neox-20b") else "gpt2"


def _eos_token_id(tokenizer: BPETokenizer) -> int | None:
    if getattr(tokenizer, "_kind", "") == "hf":
        return int(tokenizer._enc.eos_token_id)  # type: ignore[attr-defined]
    token = getattr(tokenizer._enc, "eot_token", None)  # type: ignore[attr-defined]
    return int(token) if token is not None else None


def iter_hf_documents(
    spec: SourceSpec,
    *,
    cache_dir: str | Path,
    seed: int,
):
    """Yield candidate text documents from a Hugging Face dataset stream."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": spec.repo_id,
        "split": spec.split,
        "streaming": True,
        "cache_dir": str(cache_dir),
    }
    if spec.subset:
        kwargs["name"] = spec.subset
    ds = load_dataset(**kwargs)
    if spec.shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        text = infer_text(row, spec.text_fields)
        if len(text) >= spec.min_chars:
            yield text


class TokenShardWriter:
    """Append tokenized documents into fixed-size ``.bin`` shards."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        split: str,
        dtype: str,
        shard_tokens: int,
    ):
        self.output_dir = Path(output_dir)
        self.split = split
        self.dtype = np.dtype(dtype)
        self.shard_tokens = shard_tokens
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: list[np.ndarray] = []
        self.buffer_tokens = 0
        self.shard_index = 0
        self.shards: list[dict[str, Any]] = []

    def _flush(self) -> None:
        if self.buffer_tokens == 0:
            return
        shard_path = self.output_dir / f"{self.split}-{self.shard_index:05d}.bin"
        arr = np.concatenate(self.buffer).astype(self.dtype, copy=False)
        arr.tofile(shard_path)
        self.shards.append(
            {
                "path": str(shard_path),
                "num_tokens": int(arr.shape[0]),
                "dtype": self.dtype.name,
            }
        )
        self.buffer = []
        self.buffer_tokens = 0
        self.shard_index += 1

    def write(self, token_ids: np.ndarray) -> int:
        if token_ids.size == 0:
            return 0
        self.buffer.append(token_ids.astype(self.dtype, copy=False))
        self.buffer_tokens += int(token_ids.size)
        while self.buffer_tokens >= self.shard_tokens:
            joined = np.concatenate(self.buffer)
            keep = joined[: self.shard_tokens]
            rest = joined[self.shard_tokens :]
            shard_path = self.output_dir / f"{self.split}-{self.shard_index:05d}.bin"
            keep.astype(self.dtype, copy=False).tofile(shard_path)
            self.shards.append(
                {
                    "path": str(shard_path),
                    "num_tokens": int(keep.shape[0]),
                    "dtype": self.dtype.name,
                }
            )
            self.shard_index += 1
            self.buffer = [rest.astype(self.dtype, copy=False)] if rest.size else []
            self.buffer_tokens = int(rest.size)
        return int(token_ids.size)

    def finalize(self) -> list[dict[str, Any]]:
        self._flush()
        return self.shards


def _normalized_weights(specs: list[SourceSpec]) -> dict[str, float]:
    total = sum(max(0.0, spec.weight) for spec in specs)
    if total <= 0:
        raise ValueError("At least one source weight must be positive.")
    return {spec.name: max(0.0, spec.weight) / total for spec in specs}


def _pick_source(
    specs: list[SourceSpec],
    targets: dict[str, float],
    produced: dict[str, int],
    exhausted: set[str],
) -> SourceSpec | None:
    best: SourceSpec | None = None
    best_deficit = float("-inf")
    for spec in specs:
        if spec.name in exhausted:
            continue
        deficit = targets[spec.name] - produced[spec.name]
        if deficit > best_deficit:
            best = spec
            best_deficit = deficit
    return best


def build_corpus(
    config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Stream, deduplicate, globally shuffle, tokenize, and shard a corpus."""
    data_cfg = config["data"]
    build_cfg = data_cfg.get("build", {})
    out_dir = Path(output_dir or data_cfg.get("output_dir") or data_cfg["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_name = _tokenizer_kind(data_cfg.get("tokenizer", "neox"))
    tokenizer = BPETokenizer(encoding=tokenizer_name)
    eos_id = _eos_token_id(tokenizer)
    dtype = build_cfg.get("dtype", "uint16")
    shard_tokens = int(build_cfg.get("shard_tokens", 268_435_456))
    train_tokens = int(build_cfg.get("train_tokens", data_cfg.get("token_budget", 0)))
    val_tokens = int(build_cfg.get("val_tokens", data_cfg.get("val_tokens", 100_000_000)))
    shuffle_buffer_docs = int(build_cfg.get("shuffle_buffer_docs", 8192))
    seed = int(config.get("training", {}).get("seed", 42))
    dedupe_path = Path(build_cfg.get("dedupe_db", out_dir / "dedupe.sqlite"))
    dedupe = SqliteDeduper(dedupe_path) if build_cfg.get("dedupe", True) else None

    specs = [SourceSpec.from_dict(raw) for raw in data_cfg["sources"]]
    weights = _normalized_weights(specs)
    source_iters = {
        spec.name: iter_hf_documents(spec, cache_dir=out_dir / "hf_cache", seed=seed)
        for spec in specs
    }
    exhausted: set[str] = set()
    source_split_targets = {
        "train": {spec.name: train_tokens * weights[spec.name] for spec in specs},
        "val": {spec.name: val_tokens * weights[spec.name] for spec in specs},
    }
    source_split_produced = {
        "train": {spec.name: 0 for spec in specs},
        "val": {spec.name: 0 for spec in specs},
    }
    split_targets = {"train": train_tokens, "val": val_tokens}
    split_produced = {"train": 0, "val": 0}
    writers = {
        "train": TokenShardWriter(
            output_dir=out_dir,
            split="train",
            dtype=dtype,
            shard_tokens=shard_tokens,
        ),
        "val": TokenShardWriter(
            output_dir=out_dir,
            split="val",
            dtype=dtype,
            shard_tokens=shard_tokens,
        ),
    }

    rng = random.Random(seed)
    doc_buffer: list[tuple[str, str]] = []
    docs_seen = 0
    docs_kept = 0

    def write_buffered(split: str) -> bool:
        nonlocal docs_kept
        if not doc_buffer:
            return False
        source_name, text = doc_buffer.pop(rng.randrange(len(doc_buffer)))
        token_ids = tokenizer.encode(text)
        if eos_id is not None:
            token_ids = token_ids + [eos_id]
        arr = np.asarray(token_ids, dtype=np.int64)
        if arr.size == 0:
            return True
        written = writers[split].write(arr.astype(dtype, copy=False))
        split_produced[split] += written
        source_split_produced[split][source_name] += written
        docs_kept += 1
        return True

    for split in ("val", "train"):
        while split_produced[split] < split_targets[split]:
            spec = _pick_source(
                specs,
                source_split_targets[split],
                source_split_produced[split],
                exhausted,
            )
            if spec is None:
                break
            try:
                text = next(source_iters[spec.name])
            except StopIteration:
                exhausted.add(spec.name)
                continue
            docs_seen += 1
            if dedupe is not None and not dedupe.add_if_new(text):
                continue
            doc_buffer.append((spec.name, text))
            if dedupe is not None and docs_seen % 1000 == 0:
                dedupe.commit()
            while len(doc_buffer) >= shuffle_buffer_docs and split_produced[split] < split_targets[split]:
                write_buffered(split)

        while split_produced[split] < split_targets[split] and write_buffered(split):
            pass

    train_shards = writers["train"].finalize()
    val_shards = writers["val"].finalize()
    if dedupe is not None:
        dedupe.close()

    manifest = {
        "format": "aria_corpus_v1",
        "tokenizer": tokenizer_name,
        "tokenizer_repo": (
            "EleutherAI/gpt-neox-20b" if tokenizer_name == "neox" else "gpt2"
        ),
        "dtype": np.dtype(dtype).name,
        "seed": seed,
        "seq_len": int(data_cfg["seq_len"]),
        "docs_seen": docs_seen,
        "docs_kept": docs_kept,
        "splits": {
            "train": {
                "num_tokens": split_produced["train"],
                "source_tokens": source_split_produced["train"],
                "shards": train_shards,
            },
            "val": {
                "num_tokens": split_produced["val"],
                "source_tokens": source_split_produced["val"],
                "shards": val_shards,
            },
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


class ShardedTokenSampler:
    """Fixed-shape random window sampler over mmap shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        seed: int = 42,
    ):
        manifest = load_manifest(manifest_path)
        split_cfg = manifest["splits"][split]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.shards: list[np.memmap] = []
        self.valid_lengths: list[int] = []
        self.dtype = np.dtype(manifest.get("dtype", "uint16"))

        for shard in split_cfg["shards"]:
            data = np.memmap(shard["path"], mode="r", dtype=np.dtype(shard["dtype"]))
            valid = int(data.shape[0]) - seq_len - 1
            if valid > 0:
                self.shards.append(data)
                self.valid_lengths.append(valid)

        if not self.shards:
            raise ValueError(
                f"No usable shards found for split='{split}' with seq_len={seq_len}."
            )
        weights = np.asarray(self.valid_lengths, dtype=np.float64)
        self.shard_probs = weights / weights.sum()

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        shard_ids = self.rng.choice(
            len(self.shards),
            size=self.batch_size,
            p=self.shard_probs,
        )
        xs = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        ys = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        for row, shard_id in enumerate(shard_ids):
            start = int(self.rng.integers(0, self.valid_lengths[shard_id]))
            window = np.asarray(
                self.shards[shard_id][start : start + self.seq_len + 1],
                dtype=np.int64,
            )
            xs[row] = window[:-1]
            ys[row] = window[1:]
        return (
            torch.from_numpy(xs).to(self.device, non_blocking=True),
            torch.from_numpy(ys).to(self.device, non_blocking=True),
        )

