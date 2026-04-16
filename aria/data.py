"""Character-level dataset and loaders.

Phase 0 uses the tiny-shakespeare dataset (~1MB, ~65 unique characters).
Small enough to fit fully in RAM, fast to tokenize, standard benchmark.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def load_tinyshakespeare(cache_dir: str | Path = "./data") -> str:
    """Download (if needed) and return the full tiny-shakespeare text."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "tinyshakespeare.txt"
    if not path.exists():
        print(f"Downloading tinyshakespeare to {path}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """Simple character-level tokenizer built from a reference corpus."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)


class CharDataset(Dataset):
    """Sliding-window character dataset.

    Yields (input_ids, targets) where targets = input_ids shifted by 1.
    For character-level training, each sample is `seq_len` contiguous chars.
    """

    def __init__(self, data: np.ndarray, seq_len: int):
        assert data.ndim == 1
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx: idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def build_datasets(cache_dir: str | Path = "./data",
                   seq_len: int = 256,
                   train_split: float = 0.9) -> tuple[CharDataset, CharDataset, CharTokenizer]:
    """Load tiny-shakespeare, build tokenizer, split train/val, return datasets."""
    text = load_tinyshakespeare(cache_dir)
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text), dtype=np.int32)

    n_train = int(len(data) * train_split)
    train_data = data[:n_train]
    val_data = data[n_train:]

    train_ds = CharDataset(train_data, seq_len)
    val_ds = CharDataset(val_data, seq_len)
    return train_ds, val_ds, tokenizer


# ---------------------------------------------------------------------------
# BPE tokenization + larger corpora (Kaggle / TRC scale)
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Wraps tiktoken's GPT-2 BPE as a drop-in replacement for CharTokenizer.

    Vocabulary: 50257 tokens. Uses gpt2 encoding by default which is well-tested
    and stable across tiktoken versions.
    """

    def __init__(self, encoding: str = "gpt2"):
        import tiktoken  # imported lazily so char-level path has no extra deps
        self._enc = tiktoken.get_encoding(encoding)
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def encode_chunked(self, text: str, chunk_chars: int = 1_000_000) -> list[int]:
        """Parallel tokenize a large blob by splitting on newlines into ~1MB chunks.

        Uses encode_ordinary_batch which releases the GIL and runs across cores.
        Roughly 10-30x faster than a single encode_ordinary call on 100M+ char inputs.
        """
        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_chars, n)
            if end < n:
                nl = text.rfind("\n", start, end)
                if nl > start:
                    end = nl + 1
            chunks.append(text[start:end])
            start = end
        ids: list[int] = []
        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            for enc in self._enc.encode_ordinary_batch(batch):
                ids.extend(enc)
            done = min(i + batch_size, len(chunks))
            print(f"  tokenized {done}/{len(chunks)} chunks", flush=True)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


def _tokenize_to_binfile(text: str, tokenizer: BPETokenizer,
                         out_path: Path) -> np.ndarray:
    """Tokenize a text blob once and memoize as an int32 binary file."""
    if out_path.exists():
        return np.fromfile(out_path, dtype=np.int32)
    print(f"Tokenizing {len(text):,} chars -> {out_path}", flush=True)
    ids = tokenizer.encode_chunked(text) if len(text) > 2_000_000 else tokenizer.encode(text)
    arr = np.array(ids, dtype=np.int32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out_path)
    return arr


def load_wikitext103(cache_dir: str | Path, max_train_tokens: int | None = None
                     ) -> tuple[str, str]:
    """Load wikitext-103 train and validation texts via HuggingFace datasets.

    Returns (train_text, val_text). Streams from the HF hub on first call.
    """
    from datasets import load_dataset  # lazy import

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=str(cache_dir))

    train_pieces: list[str] = []
    total_chars = 0
    char_budget = max_train_tokens * 5 if max_train_tokens else None  # ~5 chars/tok
    for row in ds["train"]:
        t = row["text"]
        if not t:
            continue
        train_pieces.append(t)
        total_chars += len(t)
        if char_budget is not None and total_chars >= char_budget:
            break
    train_text = "".join(train_pieces)

    val_text = "".join(row["text"] for row in ds["validation"] if row["text"])
    return train_text, val_text


def load_fineweb_edu(cache_dir: str | Path, max_train_tokens: int | None = None,
                     val_tokens: int = 500_000) -> tuple[str, str]:
    """Load FineWeb-Edu via HuggingFace streaming.

    Streams from ``HuggingFaceFW/fineweb-edu`` (1.3T tokens total). Only
    downloads as much text as needed for ``max_train_tokens`` (at ~4 chars/BPE
    token). A small val split is carved from the end of the streamed data.

    FineWeb-Edu is the highest-quality filtered web corpus available (April
    2026). 10% of its tokens match the performance of 350B unfiltered tokens.
    """
    from datasets import load_dataset  # lazy import

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use the "sample-10BT" subset for tractable downloads; full corpus is 1.3T
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        cache_dir=str(cache_dir),
    )

    char_budget = (max_train_tokens or 50_000_000) * 4  # ~4 chars per BPE token
    val_char_budget = val_tokens * 4

    pieces: list[str] = []
    total_chars = 0
    for row in ds:
        t = row.get("text", "")
        if not t or len(t) < 50:
            continue
        pieces.append(t)
        total_chars += len(t)
        if total_chars >= char_budget + val_char_budget:
            break
        if len(pieces) % 10000 == 0:
            print(f"  streamed {len(pieces):,} docs, {total_chars:,} chars", flush=True)

    all_text = "".join(pieces)
    # Split: last val_char_budget chars for validation, rest for training
    split_point = len(all_text) - val_char_budget
    train_text = all_text[:split_point]
    val_text = all_text[split_point:]
    print(f"FineWeb-Edu: train {len(train_text):,} chars, val {len(val_text):,} chars "
          f"({len(pieces):,} docs)", flush=True)
    return train_text, val_text


def build_bpe_datasets(dataset: str = "tinyshakespeare",
                       cache_dir: str | Path = "./data",
                       seq_len: int = 512,
                       train_split: float = 0.9,
                       max_train_tokens: int | None = None
                       ) -> tuple[CharDataset, CharDataset, BPETokenizer]:
    """BPE-tokenized dataset loader for Phase 1+ experiments.

    Supported datasets:
        - tinyshakespeare: small toy corpus, ~300K BPE tokens
        - wikitext-103: ~100M+ train tokens, standard LM benchmark
        - fineweb-edu: high-quality educational web text (streams from HF Hub)

    Returns datasets compatible with the existing CharDataset interface
    (it stores int32 token IDs — label is misleading but the type is the same).
    """
    cache_dir = Path(cache_dir)
    tokenizer = BPETokenizer()

    bin_dir = cache_dir / "bpe_tokens" / dataset
    bin_dir.mkdir(parents=True, exist_ok=True)
    train_bin = bin_dir / "train.bin"
    val_bin = bin_dir / "val.bin"

    if train_bin.exists() and val_bin.exists():
        train_arr = np.fromfile(train_bin, dtype=np.int32)
        val_arr = np.fromfile(val_bin, dtype=np.int32)
    elif dataset == "tinyshakespeare":
        text = load_tinyshakespeare(cache_dir)
        all_arr = _tokenize_to_binfile(text, tokenizer, bin_dir / "_all.bin")
        n_train = int(len(all_arr) * train_split)
        train_arr = all_arr[:n_train]
        val_arr = all_arr[n_train:]
        train_arr.tofile(train_bin)
        val_arr.tofile(val_bin)
    elif dataset == "wikitext-103":
        train_text, val_text = load_wikitext103(cache_dir, max_train_tokens)
        train_arr = _tokenize_to_binfile(train_text, tokenizer, train_bin)
        val_arr = _tokenize_to_binfile(val_text, tokenizer, val_bin)
    elif dataset == "fineweb-edu":
        train_text, val_text = load_fineweb_edu(cache_dir, max_train_tokens)
        train_arr = _tokenize_to_binfile(train_text, tokenizer, train_bin)
        val_arr = _tokenize_to_binfile(val_text, tokenizer, val_bin)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    if max_train_tokens is not None and len(train_arr) > max_train_tokens:
        train_arr = train_arr[:max_train_tokens]

    print(f"BPE dataset '{dataset}': train {len(train_arr):,} tokens, "
          f"val {len(val_arr):,} tokens, vocab {tokenizer.vocab_size}")

    train_ds = CharDataset(train_arr, seq_len)
    val_ds = CharDataset(val_arr, seq_len)
    return train_ds, val_ds, tokenizer


class RandomWindowSampler:
    """Infinite iterator that yields random windows from a CharDataset.

    More efficient than DataLoader for our use case: no worker overhead,
    draws random indices directly. Matches Karpathy's nanoGPT data loop.
    """

    def __init__(self, dataset: CharDataset, batch_size: int, device: torch.device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.dataset)
        ix = torch.randint(0, n, (self.batch_size,))
        xs = torch.stack([self.dataset[int(i)][0] for i in ix])
        ys = torch.stack([self.dataset[int(i)][1] for i in ix])
        return xs.to(self.device, non_blocking=True), ys.to(self.device, non_blocking=True)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield self.sample()
