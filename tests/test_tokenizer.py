import pytest

from fable_lm.tokenizer import SPECIAL_IDS, SPECIALS, ByteBPETokenizer

TRAIN_TEXTS = [
    "User: Hello!\nAssistant: Hi there. How can I help you today?\n",
    "User: What is a byte-level tokenizer?\n"
    "Assistant: It splits text into bytes first, then learns merges. "
    "That way encoding never fails on unusual characters.\n",
    "User: Write me a poem.\n"
    "Assistant: I can't reproduce copyrighted poems, but I can write an "
    "original short verse about the sea for you.\n",
    "The quick brown fox jumps over the lazy dog. " * 20,
    "<|doc|>Claude Fable keeps responses concise and warm. " * 10,
]


@pytest.fixture(scope="module")
def tok():
    t = ByteBPETokenizer()
    t.train(TRAIN_TEXTS, vocab_size=350)
    return t


def test_roundtrip_plain(tok):
    for text in TRAIN_TEXTS:
        if "<|doc|>" in text:
            continue
        ids = tok.encode(text)
        assert ids == tok.encode(text)  # deterministic
        assert tok.decode(ids) == text


def test_roundtrip_unusual_chars(tok):
    # byte-level must be lossless on arbitrary input, incl. emoji and CJK
    for text in ["héllo wörld", "日本語テスト", "emoji 🚀🔥", "tabs\tand\nnewlines", ""]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text


def test_specials_encode_to_single_ids(tok):
    text = "before<|eos|>after<|doc|>end"
    ids = tok.encode(text)
    assert ids.count(SPECIAL_IDS["<|eos|>"]) == 1
    assert ids.count(SPECIAL_IDS["<|doc|>"]) == 1
    # specials never merge with neighbors: neighbors are ordinary token ids
    eos_pos = ids.index(SPECIAL_IDS["<|eos|>"])
    assert ids[eos_pos - 1] != SPECIAL_IDS["<|doc|>"]
    assert tok.decode(ids) == text
    # add_eos appends exactly one eos
    assert tok.encode("hi", add_eos=True)[-1] == SPECIAL_IDS["<|eos|>"]


def test_merges_reduce_length(tok):
    plain = "Assistant: Hi there. How can I help you today?"
    n_bytes = len(plain.encode("utf-8"))
    assert len(tok.encode(plain)) < n_bytes


def test_vocab_size_matches(tok):
    assert tok.vocab_size == 256 + len(SPECIALS) + len(tok.merges)
    assert tok.vocab_size <= 350


def test_save_load_roundtrip(tmp_path, tok):
    p = tmp_path / "tok.json"
    tok.save(p)
    t2 = ByteBPETokenizer.load(p)
    for text in TRAIN_TEXTS:
        assert t2.encode(text) == tok.encode(text)
    assert t2.decode(tok.encode("reload works")) == "reload works"


def test_train_grows_vocab_deterministically():
    a = ByteBPETokenizer()
    a.train(TRAIN_TEXTS, 300)
    b = ByteBPETokenizer()
    b.train(TRAIN_TEXTS, 300)
    assert a.merges == b.merges
