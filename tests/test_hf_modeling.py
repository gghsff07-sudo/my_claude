"""Tests for the Fable HF modeling wrapper (step 3).

Parity vs the original fable_lm.FableLM is the core gate: same module tree,
same init, same logits (bit-level, since ops are identical), plus KV-cache
correctness and generate() with the required sampling controls.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fable_lm.model import FableLM, ModelConfig
from hf_fable import FableConfig, FableLMHeadModel

REPO = Path(__file__).resolve().parents[1]

# small config for speed, but non-default shapes to catch hardcoding
TINY = dict(vocab_size=300, context_len=48, n_layers=2, d_model=32, n_heads=2, d_ff=64)


def tiny_hf_cfg(**over):
    kw = dict(
        vocab_size=TINY["vocab_size"], max_position_embeddings=TINY["context_len"],
        num_hidden_layers=TINY["n_layers"], hidden_size=TINY["d_model"],
        num_attention_heads=TINY["n_heads"], intermediate_size=TINY["d_ff"],
    )
    kw.update(over)
    return FableConfig(**kw)


def tiny_repo_cfg():
    return ModelConfig(**TINY)


def make_pair(seed=0):
    """Same seed -> same init draw: both classes use normal(0, 0.02) in the
    same module order, so weights are identical."""
    torch.manual_seed(seed)
    hf = FableLMHeadModel(tiny_hf_cfg())
    torch.manual_seed(seed)
    orig = FableLM(tiny_repo_cfg())
    hf.eval()
    orig.eval()
    return hf, orig


def test_same_module_tree_and_init():
    hf, orig = make_pair()
    hk = [k for k, _ in hf.named_parameters()]
    ok_ = [k for k, _ in orig.named_parameters()]
    assert hk == ok_, (hk, ok_)
    for (kn, hp), (_, op) in zip(hf.named_parameters(), orig.named_parameters()):
        # tied head.weight is the same tensor as tok_emb.weight in both
        assert torch.equal(hp, op), kn
    # tying actually happened on both sides
    assert hf.head.weight is hf.tok_emb.weight
    assert orig.head.weight is orig.tok_emb.weight


def test_logits_parity_no_mask():
    hf, orig = make_pair()
    torch.manual_seed(1)
    ids = torch.randint(0, TINY["vocab_size"], (2, 17))
    with torch.no_grad():
        lo, _ = orig(ids)
        lh = hf(input_ids=ids).logits
    assert lo.shape == lh.shape == (2, 17, TINY["vocab_size"])
    assert torch.equal(lo, lh)


def test_logits_parity_with_attention_mask():
    hf, orig = make_pair()
    torch.manual_seed(2)
    ids = torch.randint(0, TINY["vocab_size"], (2, 20))
    mask = torch.ones_like(ids)
    mask[:, 12:] = 0  # padded tail
    with torch.no_grad():
        lh = hf(input_ids=ids, attention_mask=mask).logits
        lo, _ = orig(ids)
    # positions before the pad boundary must be identical to the unmasked run:
    # columns 0..11 attend only to 0..11 in both cases (causal), and the mask
    # only removes attention to positions 12+ which causal already excluded.
    assert torch.allclose(lh[:, :12, :], lo[:, :12, :], atol=0, rtol=0)
    # padded tail rows: logits differ from the unmasked run (pad ids excluded)
    assert not torch.allclose(lh[:, 12:, :], lo[:, 12:, :])


def test_kv_cache_matches_full_forward():
    hf, _ = make_pair()
    torch.manual_seed(3)
    ids = torch.randint(0, TINY["vocab_size"], (1, 20))
    with torch.no_grad():
        full = hf(input_ids=ids).logits
        # chunked prefill + step-by-step with cache
        out = hf(input_ids=ids[:, :9], use_cache=True)
        cache = out.past_key_values
        assert cache.get_seq_length() == 9
        rest = []
        for t in range(9, 20):
            o = hf(input_ids=ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            rest.append(o.logits)
        cached = torch.cat([out.logits] + rest, dim=1)
    assert torch.allclose(full, cached, atol=1e-6), (full - cached).abs().max().item()


def test_labels_shift_and_loss_value():
    hf, _ = make_pair()
    torch.manual_seed(4)
    ids = torch.randint(0, TINY["vocab_size"], (2, 16))
    out = hf(input_ids=ids, labels=ids)
    assert out.loss is not None
    # manual shifted CE
    import torch.nn.functional as F

    logits = out.logits
    manual = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
    )
    assert torch.allclose(out.loss, manual, atol=1e-6)
    # -100 positions are ignored
    lab = ids.clone()
    lab[:, 5:] = -100
    out2 = hf(input_ids=ids, labels=lab)
    assert torch.isfinite(out2.loss)


def test_generate_with_sampling_controls():
    hf, _ = make_pair()
    torch.manual_seed(5)
    ids = torch.randint(0, TINY["vocab_size"], (1, 5))
    out = hf.generate(
        ids,
        max_new_tokens=10,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=TINY and 256 if TINY["vocab_size"] > 256 else 0,
    )
    assert out.shape[1] == 15
    # greedy path: deterministic, stops at eos if drawn
    g1 = hf.generate(ids, max_new_tokens=6, do_sample=False, num_beams=1,
                     pad_token_id=1, eos_token_id=None)
    g2 = hf.generate(ids, max_new_tokens=6, do_sample=False, num_beams=1,
                     pad_token_id=1, eos_token_id=None)
    assert torch.equal(g1, g2)


def test_context_length_guard():
    hf, _ = make_pair()
    ids = torch.zeros(1, TINY["context_len"] + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        hf(input_ids=ids)


@pytest.mark.skipif(not (REPO / "checkpoints" / "best.pt").is_file(), reason="no checkpoint")
def test_real_checkpoint_loads_and_matches_original():
    """The shipped checkpoint loads into the HF wrapper with zero renames and
    produces bit-identical logits to fable_lm.FableLM on the same inputs."""
    ck = torch.load(REPO / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg_d = ck["cfg"]
    hf = FableLMHeadModel(FableConfig(
        vocab_size=cfg_d["vocab_size"], max_position_embeddings=cfg_d["context_len"],
        num_hidden_layers=cfg_d["n_layers"], hidden_size=cfg_d["d_model"],
        num_attention_heads=cfg_d["n_heads"], intermediate_size=cfg_d["d_ff"],
    ))
    missing, unexpected = hf.load_state_dict(ck["model"], strict=False)
    # head.weight is the only allowed "missing" (it is tied to tok_emb.weight)
    assert unexpected == [], unexpected
    assert set(missing) <= {"head.weight"}, missing
    hf.eval()

    orig = FableLM(ModelConfig(**cfg_d))
    orig.load_state_dict(ck["model"])  # strict=True works on the original
    orig.eval()

    torch.manual_seed(6)
    ids = torch.randint(0, cfg_d["vocab_size"], (2, 33))
    with torch.no_grad():
        lo, _ = orig(ids)
        lh = hf(input_ids=ids).logits
    assert torch.equal(lo, lh)
    assert hf.num_params() == orig.num_params() == 3_285_504
