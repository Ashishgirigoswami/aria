from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aria.lsa import LSALanguageModel, _FullAttentionBlock  # noqa: E402
from aria.corpus import ShardedTokenSampler  # noqa: E402


def test_1b_config_exposes_expected_shape():
    cfg = yaml.safe_load((ROOT / "configs" / "aria_v3_1b_model.yaml").read_text())
    model_cfg = cfg["model"]
    assert model_cfg["d_model"] == 2048
    assert model_cfg["n_layers"] == 20
    assert model_cfg["n_heads"] == 16
    assert model_cfg["n_kv_heads"] == 4
    assert model_cfg["d_head"] == 128
    assert model_cfg["d_kv_latent"] == 512
    assert model_cfg["full_attention_layers"] == [3, 6, 9, 12, 15, 18]


def test_explicit_full_attention_layers_are_respected():
    model = LSALanguageModel(
        vocab_size=512,
        d_model=128,
        n_layers=20,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        d_ff=384,
        d_kv_latent=64,
        d_state=32,
        max_seq_len=64,
        rope_base=500000.0,
        norm_eps=1e-6,
        full_attention_layers=[3, 6, 9, 12, 15, 18],
        qk_norm=True,
        tie_weights=True,
    )
    full_blocks = [block for block in model.blocks if isinstance(block, _FullAttentionBlock)]
    assert model.full_attention_layers == (3, 6, 9, 12, 15, 18)
    assert len(full_blocks) == 6


def test_sharded_sampler_returns_static_shapes(tmp_path: Path):
    shard0 = tmp_path / "train-00000.bin"
    shard1 = tmp_path / "val-00000.bin"
    torch.arange(0, 256, dtype=torch.int64).numpy().astype("uint16").tofile(shard0)
    torch.arange(256, 512, dtype=torch.int64).numpy().astype("uint16").tofile(shard1)
    manifest = {
        "dtype": "uint16",
        "splits": {
            "train": {
                "shards": [{"path": str(shard0), "dtype": "uint16", "num_tokens": 256}],
            },
            "val": {
                "shards": [{"path": str(shard1), "dtype": "uint16", "num_tokens": 256}],
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    sampler = ShardedTokenSampler(
        manifest_path=manifest_path,
        split="train",
        seq_len=16,
        batch_size=4,
        device=torch.device("cpu"),
        seed=123,
    )
    x, y = sampler.sample()
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
