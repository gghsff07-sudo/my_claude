"""Tests for chat REPL command parsing, prompt cropping, and tokenizer fallback."""

from dataclasses import asdict
from pathlib import Path

import torch

from fable_lm.chat import _prompt_ids, parse_command, reply
from fable_lm.model import FableLM, ModelConfig
from fable_lm.tokenizer import ByteBPETokenizer

REPO = Path(__file__).resolve().parents[1]


def test_parse_command_valid():
    assert parse_command("/temp 0.5") == ("temp", "0.5")
    assert parse_command("/topp 0.9") == ("topp", "0.9")
    assert parse_command("/topk 40") == ("topk", "40")
    assert parse_command("/maxnew 128") == ("maxnew", "128")
    assert parse_command("/seed 123") == ("seed", "123")
    assert parse_command("/system be brief") == ("system", "be brief")
    assert parse_command("/save chat.txt") == ("save", "chat.txt")
    assert parse_command("/reset") == ("reset", "")
    assert parse_command("/quit") == ("quit", "")
    assert parse_command("/help") == ("help", "")
    assert parse_command("/SAVE x.txt") == ("save", "x.txt")  # case-insensitive


def test_parse_command_malformed_never_raises():
    # missing arguments / non-numeric values -> usage errors, not exceptions
    for line in ("/temp", "/topp", "/topk", "/maxnew", "/seed", "/save", "/system",
                 "/temp abc", "/topp x", "/seed 1.5", "/topk few", "/maxnew lots",
                 "/frobnicate 3"):
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


def _tiny_tok_model(ctx: int = 48):
    tok = ByteBPETokenizer()
    cfg = ModelConfig(vocab_size=tok.vocab_size, context_len=ctx,
                      n_layers=1, d_model=16, n_heads=2, d_ff=16)
    torch.manual_seed(0)
    return tok, FableLM(cfg)


def test_prompt_crop_leaves_room_for_reply():
    """Budget = ctx - max_new - 1; a long prompt is cropped to fit."""
    tok, model = _tiny_tok_model(ctx=48)
    long_history = [("user", "word " * 40), ("assistant", "word " * 40)]
    idx = _prompt_ids(model, tok, long_history, "hello", "be brief", max_new=12)
    assert idx.shape == (1, 48 - 12 - 1)  # budget is exact


def test_prompt_crop_survives_max_new_over_context():
    """v0.3.0 fix: max_new >= context_len used to turn the slice positive and
    keep a long prompt; now the prompt is cropped to the last ctx tokens."""
    tok, model = _tiny_tok_model(ctx=48)
    long_history = [("user", "word " * 40), ("assistant", "word " * 40)]
    idx = _prompt_ids(model, tok, long_history, "hello", "be brief", max_new=500)
    assert idx.shape == (1, 48)  # clamped, generation restarts the cache
    # and short prompts pass through untouched
    idx2 = _prompt_ids(model, tok, [], "hi", "", max_new=500)
    assert idx2.shape[1] <= 48


def test_reply_accepts_top_k():
    """top_k is exposed through reply() (v0.3.0) — both limits and disables."""
    tok, model = _tiny_tok_model()
    torch.manual_seed(1)
    a = reply(model, tok, [], "hi", "", 0.8, 0.9, 8, top_k=5)
    torch.manual_seed(1)
    b = reply(model, tok, [], "hi", "", 0.8, 0.9, 8, top_k=0)  # disabled
    assert isinstance(a, str) and isinstance(b, str)
    # with the same seed and a tight top_k the outputs coincide
    torch.manual_seed(1)
    c = reply(model, tok, [], "hi", "", 0.8, 0.9, 8, top_k=5)
    assert a == c
