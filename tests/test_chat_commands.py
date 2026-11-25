"""Tests for chat REPL command parsing and checkpoint tokenizer fallback."""

from dataclasses import asdict
from pathlib import Path

import torch

from fable_lm.chat import parse_command
from fable_lm.model import FableLM, ModelConfig

REPO = Path(__file__).resolve().parents[1]


def test_parse_command_valid():
    assert parse_command("/temp 0.5") == ("temp", "0.5")
    assert parse_command("/topp 0.9") == ("topp", "0.9")
    assert parse_command("/seed 123") == ("seed", "123")
    assert parse_command("/system be brief") == ("system", "be brief")
    assert parse_command("/save chat.txt") == ("save", "chat.txt")
    assert parse_command("/reset") == ("reset", "")
    assert parse_command("/quit") == ("quit", "")
    assert parse_command("/help") == ("help", "")
    assert parse_command("/SAVE x.txt") == ("save", "x.txt")  # case-insensitive


def test_parse_command_malformed_never_raises():
    # missing arguments / non-numeric values -> usage errors, not exceptions
    for line in ("/temp", "/topp", "/seed", "/save", "/system",
                 "/temp abc", "/topp x", "/seed 1.5", "/frobnicate 3"):
        cmd = parse_command(line)
        assert cmd is not None and cmd[0] == "error", line
        assert "usage" in cmd[1].lower() or "unknown" in cmd[1].lower(), line


def test_parse_command_plain_text_is_not_a_command():
    assert parse_command("hello there") is None
    assert parse_command("temp 0.5") is None
    assert parse_command("") is None


def test_parse_command_non_numeric_arg_is_error():
    cmd = parse_command("/temp 0.5 trailing")  # arg must be exactly a number
    assert cmd is not None and cmd[0] == "error"


def test_load_model_falls_back_when_stored_tokenizer_path_missing(tmp_path):
    """v0.1.0 bug: a checkpoint stores an absolute path from the training
    machine; on any other box loading crashed. Now it falls back to
    data/tokenizer.json."""
    from fable_lm.chat import load_model

    cfg = ModelConfig(vocab_size=2048, context_len=32, n_layers=1,
                      d_model=16, n_heads=2, d_ff=16)
    model = FableLM(cfg)
    ck = {"model": model.state_dict(), "cfg": asdict(cfg), "step": 1,
          "best_val": 4.0, "tokenizer": str(tmp_path / "does-not-exist.json")}
    ckpt = tmp_path / "bad-path.pt"
    torch.save(ck, ckpt)

    loaded, tok = load_model(ckpt)
    assert tok is not None and tok.vocab_size == 2048
    assert loaded.cfg.vocab_size == 2048
