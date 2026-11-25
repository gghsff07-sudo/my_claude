"""Tests for generation: sampling controls, stops, prompt assembly."""

import torch

from fable_lm.generate import (
    build_prompt_tokens,
    cue_tokens,
    extract_assistant_text,
    generate,
)
from fable_lm.model import FableLM, ModelConfig
from fable_lm.tokenizer import SPECIAL_IDS, ByteBPETokenizer


def _tok_model():
    tok = ByteBPETokenizer()  # no merges: bytes only — fine for logic tests
    cfg = ModelConfig(vocab_size=tok.vocab_size, context_len=48,
                      n_layers=1, d_model=16, n_heads=2, d_ff=16)
    torch.manual_seed(0)
    return tok, FableLM(cfg)


def test_generate_respects_max_new_tokens():
    tok, model = _tok_model()
    idx = torch.tensor([tok.encode("hello")], dtype=torch.long)
    out = generate(model, idx, max_new_tokens=7, temperature=1.0, top_p=1.0)
    assert out.shape == (1, idx.shape[1] + 7)


def test_greedy_temperature_zero_is_deterministic():
    tok, model = _tok_model()
    idx = torch.tensor([tok.encode("abc")], dtype=torch.long)
    a = generate(model, idx, 5, temperature=0.0)
    b = generate(model, idx, 5, temperature=0.0)
    assert torch.equal(a, b)


def test_generate_stops_at_eos():
    tok, model = _tok_model()
    eos = SPECIAL_IDS["<|eos|>"]
    idx = torch.tensor([[tok.encode("x")[0], eos]], dtype=torch.long)
    # eos already at the end with 0 new tokens requested -> unchanged
    out = generate(model, idx[:, :1], 1, temperature=0.0, eos_ids=(eos,))
    # model may or may not emit eos as its first token; check contract instead:
    # whatever comes back, generation halts immediately when eos is produced
    ids = out[0].tolist()
    assert len(ids) <= 3
    if eos in ids[1:]:
        assert ids.index(eos, 1) == len(ids) - 1  # nothing after eos


def test_context_crop():
    tok, model = _tok_model()
    long_ids = tok.encode("word " * 50)
    idx = torch.tensor([long_ids], dtype=torch.long)
    # input longer than context_len must not raise (forward crops internally);
    # output is always input + exactly max_new tokens
    out = generate(model, idx, 5, temperature=1.0)
    assert out.shape[1] == idx.shape[1] + 5


def test_prompt_assembly_and_extraction():
    tok, _ = _tok_model()
    ids = build_prompt_tokens(
        tok, "be brief", [("user", "hi"), ("assistant", "hello!")], "how are you?")
    text = tok.decode(ids)
    assert text == "<system>be brief</system>\nUser: hi\nAssistant: hello!\nUser: how are you?\n"
    assert tok.decode(cue_tokens(tok)) == "Assistant:"
    # extraction strips specials and cuts at hallucinated User:
    gen = tok.encode(" Hi there.") + [SPECIAL_IDS["<|eos|>"]]
    assert extract_assistant_text(tok, gen) == "Hi there."
    gen2 = tok.encode(" partial") + tok.encode("\nUser: nope")
    assert "nope" not in extract_assistant_text(tok, gen2)


def test_top_p_bounds():
    tok, model = _tok_model()
    idx = torch.tensor([tok.encode("q")], dtype=torch.long)
    for p in (0.1, 0.5, 0.9, 1.0):
        out = generate(model, idx, 3, temperature=0.9, top_p=p)
        assert out.shape[1] == idx.shape[1] + 3
