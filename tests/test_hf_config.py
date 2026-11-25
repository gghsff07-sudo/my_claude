"""Tests for the Fable HF config wrapper (step 2)."""

from __future__ import annotations

import json
from pathlib import Path

from fable_lm.model import ModelConfig
from hf_fable import FableConfig

REPO = Path(__file__).resolve().parents[1]


def test_config_defaults_match_repo_default():
    """FableConfig defaults == configs/small.py == ModelConfig defaults."""
    from configs.small import config as small_cfg  # repo default (3.29M model)

    hf = FableConfig()
    assert hf.model_type == "fable"
    assert hf.vocab_size == small_cfg.vocab_size == 2048
    assert hf.max_position_embeddings == small_cfg.context_len == 512
    assert hf.num_hidden_layers == small_cfg.n_layers == 4
    assert hf.hidden_size == small_cfg.d_model == 256
    assert hf.num_attention_heads == small_cfg.n_heads == 4
    assert hf.intermediate_size == small_cfg.d_ff == 768
    assert hf.dropout == 0.0
    assert hf.tie_word_embeddings is True
    # special token ids from the byte-BPE layout: pad=256, eos=257, doc=258
    assert (hf.pad_token_id, hf.eos_token_id, hf.bos_token_id) == (256, 257, 258)


def test_config_roundtrip_with_model_config():
    """FableConfig <-> fable_lm.ModelConfig convert losslessly both ways."""
    repo_cfg = ModelConfig(vocab_size=1111, context_len=64, n_layers=2,
                           d_model=32, n_heads=2, d_ff=64)
    hf = FableConfig(
        vocab_size=repo_cfg.vocab_size,
        max_position_embeddings=repo_cfg.context_len,
        num_hidden_layers=repo_cfg.n_layers,
        hidden_size=repo_cfg.d_model,
        num_attention_heads=repo_cfg.n_heads,
        intermediate_size=repo_cfg.d_ff,
        dropout=repo_cfg.dropout,
    )
    assert ModelConfig(
        vocab_size=hf.vocab_size,
        context_len=hf.max_position_embeddings,
        n_layers=hf.num_hidden_layers,
        d_model=hf.hidden_size,
        n_heads=hf.num_attention_heads,
        d_ff=hf.intermediate_size,
        dropout=hf.dropout,
    ) == repo_cfg


def test_config_save_load_roundtrip(tmp_path):
    """save_pretrained -> from_pretrained (local dir, no remote code)."""
    hf = FableConfig(dropout=0.1, num_hidden_layers=3)
    hf.save_pretrained(tmp_path)
    disk = json.loads((tmp_path / "config.json").read_text())
    assert disk["model_type"] == "fable"
    assert disk["num_hidden_layers"] == 3
    assert disk["dropout"] == 0.1
    # aliases must not leak into the serialized config
    assert "context_len" not in disk and "d_model" not in disk
    loaded = FableConfig.from_pretrained(tmp_path)
    assert loaded.num_hidden_layers == 3 and loaded.dropout == 0.1
    # AutoConfig works locally after `import hf_fable` (registered above)
    from transformers import AutoConfig

    auto = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(auto, FableConfig), type(auto)
    assert auto.hidden_size == hf.hidden_size


def test_config_from_checkpoint_cfg_dict():
    """A checkpoint's cfg dict (as stored in .pt files) maps onto FableConfig."""
    ck_cfg = {"vocab_size": 2048, "context_len": 512, "n_layers": 4,
              "d_model": 256, "n_heads": 4, "d_ff": 768, "dropout": 0.0}
    hf = FableConfig(
        vocab_size=ck_cfg["vocab_size"],
        max_position_embeddings=ck_cfg["context_len"],
        num_hidden_layers=ck_cfg["n_layers"],
        hidden_size=ck_cfg["d_model"],
        num_attention_heads=ck_cfg["n_heads"],
        intermediate_size=ck_cfg["d_ff"],
    )
    assert (hf.vocab_size, hf.max_position_embeddings, hf.hidden_size) == (2048, 512, 256)
    assert hf.context_len == 512 and hf.d_model == 256  # repo-style aliases
