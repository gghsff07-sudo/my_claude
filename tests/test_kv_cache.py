"""Tests for the KV cache and cached generation (v0.2.0 upgrade)."""

import torch

from fable_lm.generate import _apply_repetition_penalty, generate, generate_stream
from fable_lm.model import FableLM, ModelConfig
from fable_lm.tokenizer import ByteBPETokenizer


def _tok_model():
    tok = ByteBPETokenizer()  # no merges: bytes only — fine for logic tests
    cfg = ModelConfig(vocab_size=tok.vocab_size, context_len=48,
                      n_layers=2, d_model=16, n_heads=2, d_ff=16)
    torch.manual_seed(0)
    model = FableLM(cfg)
    model.eval()
    return tok, model


def test_cache_logits_match_full_forward():
    """Prefill + single-token steps must reproduce the uncached forward."""
    tok, model = _tok_model()
    torch.manual_seed(1)
    ids = torch.randint(0, tok.vocab_size, (1, 12))
    full_logits, _ = model(ids)

    logits, cache = model.forward_with_cache(ids[:, :5])  # prefill
    assert logits.shape == (1, 5, tok.vocab_size)
    outs = [logits]
    for t in range(5, 12):
        logits, cache = model.forward_with_cache(ids[:, t : t + 1], cache)
        assert logits.shape == (1, 1, tok.vocab_size)
        outs.append(logits)
    assert cache.seq_len() == 12
    stepped = torch.cat(outs, dim=1)
    assert torch.allclose(stepped, full_logits, atol=1e-5), (
        f"max diff {(stepped - full_logits).abs().max().item():.2e}"
    )


def test_cache_layer_shapes():
    tok, model = _tok_model()
    ids = torch.randint(0, tok.vocab_size, (1, 7))
    _, cache = model.forward_with_cache(ids)
    assert len(cache.kv) == model.cfg.n_layers
    for entry in cache.kv:
        k, v = entry
        assert k.shape == (1, model.cfg.n_heads, 7, model.cfg.d_model // model.cfg.n_heads)
        assert v.shape == k.shape
    _, cache = model.forward_with_cache(ids[:, -1:], cache)
    assert cache.seq_len() == 8


def test_greedy_cached_equals_uncached():
    tok, model = _tok_model()
    idx = torch.tensor([tok.encode("hello world")], dtype=torch.long)
    a = generate(model, idx, 20, temperature=0.0, use_cache=True)
    b = generate(model, idx, 20, temperature=0.0, use_cache=False)
    assert torch.equal(a, b), "cached greedy decoding diverged from uncached"


def test_generate_stream_equals_generate():
    tok, model = _tok_model()
    idx = torch.tensor([tok.encode("streaming test")], dtype=torch.long)
    torch.manual_seed(3)
    stream_ids = [int(n[0, 0]) for n in generate_stream(model, idx, 12, temperature=1.0)]
    torch.manual_seed(3)
    out = generate(model, idx, 12, temperature=1.0)
    assert stream_ids == out[0, idx.shape[1]:].tolist()
    assert len(stream_ids) == 12


def test_context_full_generation_continues():
    """Prompt already fills the context; the cache restarts, nothing raises."""
    tok, model = _tok_model()
    long_ids = tok.encode("word " * 30)[-model.cfg.context_len:]
    idx = torch.tensor([long_ids], dtype=torch.long)
    out = generate(model, idx, 4, temperature=1.0)
    assert out.shape[1] == idx.shape[1] + 4


def test_repetition_penalty_divides_positive_multiplies_negative():
    """HF GenerationMixin semantics: >0 divided, <0 multiplied."""
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = _apply_repetition_penalty(logits.clone(), {0, 1}, 2.0)
    assert out[0, 0].item() == 1.0  #  2.0 / 2
    assert out[0, 1].item() == -4.0  # -2.0 * 2
    assert out[0, 2].item() == 1.0  # untouched
    # seen token whose logit is 0 stays 0 under either branch
    flat = torch.tensor([[0.0, 5.0]])
    out2 = _apply_repetition_penalty(flat.clone(), {0, 1}, 2.0)
    assert out2[0, 0].item() == 0.0
