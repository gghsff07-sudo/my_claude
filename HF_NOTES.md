# HF_NOTES.md — Hugging Face Compatibility Audit

Audit of the miniature Fable-style model (repo state: commit `7a5ee8a`, 2026-11-19) against the
goal of full `transformers` compatibility:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("<user>/fable-mini", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("<user>/fable-mini", trust_remote_code=True)
```

Scope rule: only the repo plus the installed/official Hugging Face libraries are used. Nothing in
the model's math or weights changes — the upgrade is a wrapper layer on top of `fable_lm`.

---

## 1. What exists today

| Piece | Where | Format |
| --- | --- | --- |
| Model | `fable_lm/model.py` — `FableLM(nn.Module)` | Decoder-only transformer, pre-LN, learned absolute positions, GELU FF, tied embeddings (`head.weight = tok_emb.weight`), fused QKV (no bias), attn/ff proj (no bias) |
| Config | `fable_lm/model.py` — `ModelConfig` dataclass | `vocab_size=2048, context_len=512, n_layers=4, d_model=256, n_heads=4, d_ff=768, dropout=0.0`; exact param count 3,285,504 (45 tensors) |
| Tokenizer | `fable_lm/tokenizer.py` — `ByteBPETokenizer` | From-scratch byte-level BPE; 2048 vocab = 256 byte-chars + 3 specials + 1789 merges; saved as custom JSON `{type, specials, merges}` (`data/tokenizer.json`) |
| Checkpoint | `checkpoints/best.pt` (f32, step 749, best_val 4.1600), `models/fable-mini-final.pt` (fp16 copy) | Pickle dict: `{model: state_dict, cfg: dict, step, best_val, tokenizer: path}` |
| Chat | `fable_lm/chat.py` + `fable_lm/generate.py` | `<system>...</system>\n` header + `User:` / `Assistant:` plain-text turns; `<|eos|>` after each assistant reply; trailing `Assistant:` cue starts generation |
| Tests | `tests/` | 24 tests, all passing (baseline run 2026-11-19, pytest, CPU) |
| Sizes | Largest artifact 7.6 MB (fp16 `.pt`) | No Git LFS needed (GitHub's 100 MB hard limit not approached; LFS not installed in sandbox) |
| Environment | torch 2.15 CPU, transformers 5.23.0, tokenizers 0.23.3, safetensors 0.8.0, gradio 6.3.0, huggingface_hub (installed) | GitHub push works; HF Hub reachable |

Special tokens: `<|pad|>`=256, `<|eos|>`=257, `<|doc|>`=258. There is **no** `<|unk|>` — the
byte-level vocab covers every byte, so an unk token is unnecessary by design (same choice as GPT-2).

## 2. What must change for HF compatibility

### 2.1 Config (step 2)

- Need a `PretrainedConfig` subclass with `model_type` and serializable hyperparameters.
- Name: `fable` → `configuration_fable.py` (plus `__init__.py` for `trust_remote_code` imports).
- Field mapping (HF name ← repo name): `vocab_size`←`vocab_size`, `max_position_embeddings`←`context_len`,
  `num_hidden_layers`←`n_layers`, `hidden_size`←`d_model`, `num_attention_heads`←`n_heads`,
  `intermediate_size`←`d_ff`. Also expose `dropout`, `tie_word_embeddings=True` (embeddings are tied).
- The dataclass `ModelConfig` stays as-is; the HF config converts to/from it. Existing scripts
  (`train.py`, `chat.py`, `eval.py`) are untouched.

### 2.2 Modeling (step 3)

- Wrap `FableLM` as a `PreTrainedModel` subclass in `modeling_fable.py`.
- `forward(input_ids, attention_mask, labels, ...)` returning `CausalLMOutputWithPast`
  (fields: `loss, logits, past_key_values, hidden_states, attentions` — verified in 5.23).
- KV cache: feasible — attention is standard MHA with fused QKV; use HF `DynamicCache`
  (`update(key_states, value_states, layer_idx)` — 5.x signature verified).
- transformers **5.x specifics** (verified against the installed 5.23 source):
  - `PreTrainedModel` no longer inherits `GenerationMixin`; the wrapper must list `GenerationMixin`
    explicitly (as `GPT2LMHeadModel` does) so `can_generate()` is true and `.generate()` exists.
  - 5.x `generate()` drives the loop via `prepare_inputs_for_generation(input_ids,
    next_sequence_length, past_key_values, attention_mask, ..., is_first_iteration, **kwargs)`;
    the stock implementation handles slicing and forwards `attention_mask` and `use_cache` to
    forward. Our forward must accept `attention_mask`, `past_key_values`, `use_cache`, `labels`.
  - There is BC support for remote-code models that expect `cache_position`, but our forward will
    not use `cache_position` at all, avoiding that deprecated path entirely.
  - `generate()` still accepts `temperature, top_p, top_k, max_new_tokens, repetition_penalty`
    as kwargs (absorbed into `GenerationConfig`).
  - `AutoConfig.register(model_type, config)` and `AutoModelForCausalLM.register(config, model)`
    both exist in 5.23 — registering `fable` locally also makes local `from_pretrained` work
    *without* `trust_remote_code`, and `auto_map` in `config.json` covers the Hub case.
- Tied weights: safetensors cannot store shared tensors twice. The wrapper re-ties
  `head.weight = tok_emb.weight` in `post_init`/`tie_weights` and declares
  `_tied_weights_keys = ["head.weight"]` so `save_pretrained` drops the duplicate and
  `from_pretrained` re-ties on load.
- Loss shift: HF CausalLM convention shifts internally
  (`loss = CE(logits[..., :-1, :], labels[..., 1:])`). The repo's trainer shifts in the dataset
  instead. The wrapper follows the HF convention (so `labels=input_ids` just works); logits are
  unaffected by this choice, and logits parity is what we verify (< 1e-4).

### 2.3 Weight conversion (step 4)

- `convert_to_hf.py`: read a `.pt` checkpoint → write `hf_model/` with `config.json`,
  `model.safetensors`, `generation_config.json` (plus tokenizer files after step 5).
  Default input `checkpoints/best.pt` (f32, step 749, best_val 4.1600); `--checkpoint` and `--fp16`
  flags to convert the fp16 artifact instead. Default output: `hf_model/`.
- State-dict key map is identity — the HF wrapper deliberately reuses the exact module tree and
  tensor names (`tok_emb.weight`, `pos_emb.weight`, `blocks.{i}.ln1.*, blocks.{i}.attn.qkv.weight`,
  `blocks.{i}.attn.proj.weight`, `blocks.{i}.ln2.*, blocks.{i}.ff.fc.*, blocks.{i}.ff.proj.weight`,
  `ln_f.*`, `head.weight`). No renames; conversion is a format change only.
- Parity check built into the script: run sample inputs through both the original `FableLM`
  (checkpoint weights) and the HF wrapper (safetensors weights), print max |Δlogits|, require < 1e-4.
- The base `.pt` checkpoints stay untouched (rule: keep existing scripts working).

### 2.4 Tokenizer (step 5)

- Current save format is a custom JSON — not the HF `tokenizers` backend format. Need a
  `PreTrainedTokenizerFast` shipping `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json` (+ `chat_template.jinja` in 5.x).
- Build the HF `tokenizer.json` **from the repo's own data**: vocab = 256 byte-chars (ids 0–255) +
  `<|pad|>`/`<|eos|>`/`<|doc|>` (ids 256–258, as added special tokens) + 1789 merged strings.
- Byte↔char mapping: the repo's `_byte_char/_char_byte` is a bijection where bytes 0–32 →
  chars 256–288 and bytes 127–160 → chars 383–416. This differs from GPT-2's mapping in the
  127–160 range and at byte 173. **ASSUMPTION (A1, tokenizer):** rather than the library's
  `ByteLevel` pre-tokenizer/decoder (which hardcodes the GPT-2 mapping), I will use a plain BPE
  `model` with the repo's exact alphabet and merges, a `Split` pre-tokenizer reproducing the repo's
  stdlib-`re` pattern ` ?\w+| ?[^\w\s]+|\s+`, and a custom-free decode path via
  `PreTrainedTokenizerFast` with a byte-faithful `decoder` or post-decode correction. Exact id
  parity with `ByteBPETokenizer.encode` on the corpus and a byte fuzz set gates this in tests;
  if `Split`'s inverted-regex mode cannot reproduce stdlib `\w`/`\s` semantics exactly, fallback is
  shipping the BPE model without pre-tokenizer merging across boundaries and verifying parity
  empirically — final choice recorded here in step 5.
- Specials: **ASSUMPTION (A2):** `eos_token = <|eos|>`, `pad_token = <|pad|>` (as trained);
  `bos_token = <|doc|>` (re-purposed: it is the document separator, i.e. exactly a BOS in this
  model's training regime); `unk_token = None` (byte-level coverage means unk never occurs;
  GPT-2 ships without unk too). None of this retrains anything.
- Chat template — **ASSUMPTION (A3):** mirror `build_prompt_tokens` exactly:
  `{% for message %}` → `<system>{system}</system>\n` then `User: {text}\n` / `Assistant: {text} <|eos|>\n`,
  and `add_generation_prompt` appends `Assistant:` (no trailing space, matching `cue_tokens()`).
  Round-trip encode/decode and `apply_chat_template` are tested in steps 5–6.
- `data/tokenizer.json` stays untouched; the HF tokenizer files live in `hf_model/` only.

### 2.5 Tests (step 6)

- `tests/test_hf_compat.py`: save_pretrained → from_pretrained round trip, `generate()` with
  temperature/top_p/top_k/max_new_tokens/repetition_penalty, tokenizer round trip, chat template,
  and logits parity original-vs-HF (max diff printed, < 1e-4 required). Runs offline from local paths.
- All 24 pre-existing tests must keep passing (rule: keep existing scripts working).

### 2.6 Model card, upload, Space, docs (steps 7–10)

- Model name: **`fable-mini`** (matches `models/fable-mini-final.pt`).
- Model card YAML: `license: apache-2.0`, `library_name: transformers`, `pipeline_tag:
  text-generation`, `tags: [educational, tiny-model, from-scratch, byte-bpe]`, `language: en`.
  **ASSUMPTION (A4):** apache-2.0 (repo has no LICENSE file; educational intent).
- **ASSUMPTION (A5):** the HF username is unknown to me, so `push_to_hub.py` requires an explicit
  `--repo-id` (no silent default guess); it reads `HF_TOKEN` from the environment, supports
  `--private` and `--dry-run`, and refuses to upload without a token. A token IS available in this
  environment, so a real upload will be attempted to `<whoami>/fable-mini` using the token's own
  identity once the username is read from the Hub API (never hardcoded).
- Space: `sdk: gradio`, `app_file: app.py`, pinned `requirements.txt`
  (transformers, safetensors, torch, gradio, huggingface_hub at the tested versions).
- Step 10: main README "Hugging Face" section + a Colab notebook loading the model via transformers.

## 3. Assumptions register

| ID | Area | Assumption |
| --- | --- | --- |
| A1 | Tokenizer | Repo byte↔char mapping and merges ported verbatim into a plain BPE `tokenizer.json`; parity gated by tests |
| A2 | Tokenizer | bos=`<|doc|>`, unk=None (byte-level coverage; GPT-2 precedent) |
| A3 | Chat template | Template mirrors `build_prompt_tokens` / `cue_tokens` exactly |
| A4 | License | apache-2.0 (no LICENSE file in repo) |
| A5 | Hub repo id | Explicit `--repo-id` required; identity derived from the token via `whoami` at upload time |
| A6 | Git LFS | Not needed: largest file 7.6 MB < 10 MB threshold in the working rules |

## 4. Verification plan (what "done" means)

1. `pytest tests/` — all old + new tests green.
2. Logits parity: max |original − HF| < 1e-4 on sample inputs (printed by `convert_to_hf.py`).
3. `AutoModelForCausalLM.from_pretrained("hf_model/")` and `.generate(...)` work with the required
   sampling controls; tokenizer round-trips; `apply_chat_template` produces the training format.
4. Hub upload (or dry run) via `push_to_hub.py`; Space runs locally with `gradio app.py`.
