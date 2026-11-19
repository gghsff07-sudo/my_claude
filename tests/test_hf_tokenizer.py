"""Tests for the HF fast tokenizer port (step 5).

Gates: id parity with the original ByteBPETokenizer on the corpus + edge
cases, lossless round-trip, special-token handling, chat template equals
fable_lm's build_prompt_tokens + cue_tokens, and the saved hf_model/ folder
loads via AutoTokenizer with the same behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fable_lm.generate import build_prompt_tokens, cue_tokens  # noqa: E402
from fable_lm.tokenizer import SPECIALS, ByteBPETokenizer  # noqa: E402
from scripts.convert_tokenizer_to_hf import (  # noqa: E402
    build_fast_tokenizer,
    corpus_texts,
)

HF_MODEL_DIR = REPO / "hf_model"


@pytest.fixture(scope="module")
def repo_tok():
    return ByteBPETokenizer.load(REPO / "data" / "tokenizer.json")


@pytest.fixture(scope="module")
def fast(repo_tok):
    return build_fast_tokenizer(repo_tok)


def test_id_parity_on_corpus(repo_tok, fast):
    fails = []
    for name, text in corpus_texts():
        if repo_tok.encode(text) != fast.encode(text, add_special_tokens=False):
            fails.append(name)
    assert fails == [], f"encode mismatch in: {fails}"


def test_decode_parity_on_corpus(repo_tok, fast):
    for name, text in corpus_texts():
        ids = repo_tok.encode(text)
        assert fast.decode(ids) == repo_tok.decode(ids), name


def test_special_tokens_encode_to_exact_ids(repo_tok, fast):
    for s in SPECIALS:
        assert fast.encode(s, add_special_tokens=False) == [repo_tok.encode(s)[0]] == [SPECIALS.index(s) + 256]
    assert fast.decode([256, 257, 258]) == "<|pad|><|eos|><|doc|>"
    # specials inside text are never split or merged through
    text = "before <|eos|> after"
    assert repo_tok.encode(text) == fast.encode(text, add_special_tokens=False)


def test_vocab_size_and_special_ids(repo_tok, fast):
    assert fast.vocab_size == repo_tok.vocab_size == 2048
    assert fast.pad_token_id == 256 and fast.pad_token == "<|pad|>"
    assert fast.eos_token_id == 257 and fast.eos_token == "<|eos|>"
    assert fast.bos_token_id == 258 and fast.bos_token == "<|doc|>"
    assert fast.unk_token is None


def test_byte_fuzz_roundtrip(fast):
    import random

    rng = random.Random(0)
    for _ in range(300):
        b = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 80)))
        text = b.decode("utf-8", errors="replace")
        ids = fast.encode(text, add_special_tokens=False)
        assert fast.decode(ids) == text, b


def test_chat_template_matches_build_prompt_tokens(repo_tok, fast):
    cases = [
        ([{"role": "system", "content": "You are concise."},
          {"role": "user", "content": "Hi!"}], True),
        ([{"role": "user", "content": "Hi!"}], True),
        ([{"role": "system", "content": "Be brief."},
          {"role": "user", "content": "Q1?"},
          {"role": "assistant", "content": "A1."},
          {"role": "user", "content": "Q2?"}], True),
        ([{"role": "user", "content": "Hi!"},
          {"role": "assistant", "content": "Hello!"}], False),
    ]
    for msgs, gen in cases:
        rendered = fast.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
        system = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
        rest = msgs[1:] if system else msgs
        if gen:
            history = [(m["role"], m["content"]) for m in rest[:-1]]
            expected_ids = build_prompt_tokens(repo_tok, system, history, rest[-1]["content"])
            expected_ids += cue_tokens(repo_tok)
        else:
            history = [(m["role"], m["content"]) for m in rest]
            expected_ids = build_prompt_tokens(repo_tok, system, history, None)
        assert fast.encode(rendered, add_special_tokens=False) == expected_ids, msgs
        # tokenize=True path gives the same ids
        out = fast.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)
        tok_ids = list(out.input_ids) if hasattr(out, "input_ids") else out
        assert tok_ids == expected_ids, msgs


@pytest.mark.skipif(not (HF_MODEL_DIR / "tokenizer.json").is_file(), reason="hf_model/ not built")
def test_saved_folder_loads_via_autotokenizer(repo_tok):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
    assert tok.vocab_size == 2048
    assert (tok.pad_token_id, tok.eos_token_id, tok.bos_token_id) == (256, 257, 258)
    assert tok.chat_template is not None
    # behavior parity through the real loading path
    for name, text in corpus_texts()[:12]:
        assert tok.encode(text, add_special_tokens=False) == repo_tok.encode(text), name
    # specials map file exists with the hub-standard content
    import json

    sm = json.loads((HF_MODEL_DIR / "special_tokens_map.json").read_text())
    assert sm["eos_token"]["content"] == "<|eos|>"
    assert sm["unk_token"] is None
