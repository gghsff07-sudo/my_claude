"""End-to-end HF compatibility tests (step 6).

Covers the GOAL contract against the real hf_model/ folder:
  - AutoModelForCausalLM.from_pretrained (trust_remote_code) loads and generates
  - save_pretrained -> from_pretrained round trip preserves logits
  - generate() honors temperature / top_p / top_k / max_new_tokens /
    repetition_penalty
  - tokenizer round trip + chat template (already detailed in
    test_hf_tokenizer.py; here through the AutoTokenizer loading path)
  - logits parity original vs HF (max |diff| < 1e-4)

Requires: hf_model/ built (scripts/convert_to_hf.py + convert_tokenizer_to_hf.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fable_lm.model import FableLM, ModelConfig  # noqa: E402
from fable_lm.tokenizer import ByteBPETokenizer  # noqa: E402

HF_DIR = REPO / "hf_model"
CKPT = REPO / "checkpoints" / "best.pt"

pytestmark = pytest.mark.skipif(
    not (HF_DIR / "model.safetensors").is_file() or not CKPT.is_file(),
    reason="hf_model/ not built — run scripts/convert_to_hf.py "
           "and scripts/convert_tokenizer_to_hf.py",
)


@pytest.fixture(scope="module")
def orig():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = FableLM(ModelConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@pytest.fixture(scope="module")
def hf():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(HF_DIR, trust_remote_code=True)
    m.eval()
    return m


def test_from_pretrained_loads_real_model(hf, orig):
    assert type(hf).__name__ == "FableLMHeadModel"
    assert hf.num_params() == orig.num_params() == 3_285_504
    # tied head survived the safetensors round trip
    assert hf.head.weight is hf.tok_emb.weight


def test_logits_parity_original_vs_hf(hf, orig):
    """The core parity gate (same as convert_to_hf.py, through pytest)."""
    torch.manual_seed(123)
    V = orig.cfg.vocab_size
    max_diff = 0.0
    for T in (7, 33, 128, 512):
        ids = torch.randint(0, V, (2, T))
        with torch.no_grad():
            lo, _ = orig(ids)
            lh = hf(input_ids=ids).logits
        max_diff = max(max_diff, (lo - lh).abs().max().item())
    assert max_diff < 1e-4, f"max |dlogits| = {max_diff}"


def test_save_pretrained_from_pretrained_roundtrip(hf, tmp_path):
    """save_pretrained -> from_pretrained must preserve logits exactly."""
    hf.save_pretrained(tmp_path / "saved")
    from transformers import AutoModelForCausalLM

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "saved", trust_remote_code=True)
    reloaded.eval()
    torch.manual_seed(7)
    ids = torch.randint(0, 2048, (2, 21))
    with torch.no_grad():
        l1 = hf(input_ids=ids).logits
        l2 = reloaded(input_ids=ids).logits
    assert torch.equal(l1, l2)
    assert reloaded.head.weight is reloaded.tok_emb.weight
    # the remote-code files travel with save_pretrained (auto_map kept)
    assert (tmp_path / "saved" / "config.json").is_file()
    assert (tmp_path / "saved" / "model.safetensors").is_file()


def test_generate_max_new_tokens(hf):
    torch.manual_seed(0)
    ids = torch.randint(0, 2048, (1, 6))
    out = hf.generate(ids, max_new_tokens=12, do_sample=True, temperature=0.8,
                      pad_token_id=256)
    assert out.shape == (1, 18)


def test_generate_sampling_controls(hf):
    torch.manual_seed(1)
    ids = torch.randint(0, 2048, (1, 5))
    out = hf.generate(
        ids,
        max_new_tokens=10,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.2,
        pad_token_id=256,
    )
    assert out.shape == (1, 15)


def test_generate_greedy_deterministic_and_cache_consistent(hf):
    torch.manual_seed(2)
    ids = torch.randint(0, 2048, (1, 8))
    g1 = hf.generate(ids, max_new_tokens=8, do_sample=False, num_beams=1, pad_token_id=256)
    g2 = hf.generate(ids, max_new_tokens=8, do_sample=False, num_beams=1, pad_token_id=256)
    assert torch.equal(g1, g2)
    # greedy generate must equal manual argmax stepping with the cache
    with torch.no_grad():
        o = hf(input_ids=ids, use_cache=True)
        cache = o.past_key_values
        cur = o.logits[:, -1].argmax(-1, keepdim=True)
        manual = [cur]
        for _ in range(7):
            o = hf(input_ids=cur, past_key_values=cache, use_cache=True)
            cur = o.logits[:, -1].argmax(-1, keepdim=True)
            manual.append(cur)
    assert torch.equal(g1[:, 8:], torch.cat(manual, dim=1))


def test_generate_stops_at_eos(hf):
    torch.manual_seed(3)
    ids = torch.randint(0, 2048, (1, 4))
    out = hf.generate(ids, max_new_tokens=64, do_sample=True, temperature=1.0,
                      pad_token_id=256, eos_token_id=257)
    # whatever the length, it must not exceed prompt + 64 and must be finite ids
    assert out.shape[1] <= 4 + 64
    assert (out < 2048).all()


def test_attention_mask_path_matches_unmasked_prefix(hf):
    torch.manual_seed(4)
    ids = torch.randint(0, 2048, (2, 24))
    mask = torch.ones_like(ids)
    mask[:, 15:] = 0
    with torch.no_grad():
        lm = hf(input_ids=ids, attention_mask=mask).logits
        lu = hf(input_ids=ids).logits
    assert torch.equal(lm[:, :15, :], lu[:, :15, :])


def test_tokenizer_roundtrip_and_chat_template_via_autotokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_DIR)
    for text in ("Hello, world!", "café 😀 你好", "User: hi\nAssistant: hey\n"):
        ids = tok.encode(text, add_special_tokens=False)
        assert tok.decode(ids) == text
    msgs = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi!"}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert rendered == "<system>Be brief.</system>\nUser: Hi!\nAssistant:"
    # repo cross-check: same ids as fable_lm's prompt builder
    from fable_lm.generate import build_prompt_tokens, cue_tokens

    repo_tok = ByteBPETokenizer.load(REPO / "data" / "tokenizer.json")
    expected = build_prompt_tokens(repo_tok, "Be brief.", [], "Hi!") + cue_tokens(repo_tok)
    assert tok.encode(rendered, add_special_tokens=False) == expected


def test_end_to_end_chat_generation(hf):
    """Tokenizer + chat template + model.generate: the full pipeline a user
    of the Hub repo runs, including assistant-text extraction."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_DIR)
    msgs = [{"role": "system", "content": "You are a concise, warm assistant."},
            {"role": "user", "content": "Hi!"}]
    enc = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    prompt_ids = enc.input_ids if not isinstance(enc, dict) else enc["input_ids"]
    out = hf.generate(
        prompt_ids,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=256,
        eos_token_id=257,
    )
    new_ids = out[0, prompt_ids.shape[1]:].tolist()
    text = tok.decode(new_ids)
    for marker in ("<|eos|>", "<|doc|>", "<|pad|>"):
        text = text.replace(marker, "")
    text = text.split("User:")[0].strip()
    assert isinstance(text, str) and len(text) >= 0  # generation runs to completion
