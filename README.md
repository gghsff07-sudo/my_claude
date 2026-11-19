# Fable 5.2 (miniature)

A small language model and chat interface built from scratch to follow the
observable design and behavior of "Claude Fable 5.2", scaled down to run on
2 CPUs and 4 GB RAM. **It does not match the real model's capability — it is a
faithful, well-documented miniature.** See `research/system-card.md` for the
model's honest capability summary.

## What's here

| Piece | Where |
| --- | --- |
| Research notes (Fable 5.1 doc analysis, 5.2 assumptions, system card) | `research/` |
| Behavior study (what to imitate, ranked by fidelity) | `STUDY.md` |
| Design (architecture, trait→implementation map, assumptions) | `DESIGN.md` |
| Byte-level BPE tokenizer (from scratch) | `fable_lm/tokenizer.py` |
| Decoder-only transformer (from scratch, configurable) | `fable_lm/model.py` |
| Corpus builder + dataset | `fable_lm/data.py` |
| Training loop (warmup+cosine LR, clipping, checkpoints, CSV logs, early stop) | `fable_lm/train.py` |
| Chat CLI (temperature, top-p, system prompt) | `fable_lm/chat.py` |
| Evaluation (loss, perplexity, samples) | `fable_lm/eval.py` → `EVAL.md` |
| Synthetic Fable-style dialogues (training data) | `data/dialogues/` |
| Shipped checkpoint (fp16, ~7.6 MB) | `models/fable-mini-final.pt` |
| Tests (53: tokenizer, model, data, training, generation, HF compat) | `tests/` |
| HF wrapper (config, modeling — self-contained remote code) | `hf_fable/` |
| HF conversion scripts (weights, tokenizer, upload, Space) | `scripts/` |
| HF-format model folder (Hub artifact) | `hf_model/` (weights gitignored) |
| HF audit + per-step results log | `HF_NOTES.md` |
| Gradio demo (Hub id or local path, streaming) | `app.py` + `space/README.md` |
| Colab notebook (load via transformers) | `notebooks/fable_mini_colab.ipynb` |

## Setup

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest tests/ -q        # 53 tests (24 original + 29 HF compat)
```

Python 3.12, PyTorch 2.15.0+cpu. The HF side additionally needs
`transformers==5.23.0`, `safetensors`, `tokenizers` (see `requirements.txt`).

## Usage

```bash
# train from scratch (tokenizer is trained automatically and cached)
python -m fable_lm.train --steps 600 --run-name run1

# train with the chat data upweighted (the shipped recipe)
python -m fable_lm.train --steps 800 --run-name run2 --dialogue-mult 6 --data-mult 2

# chat (REPL with /temp, /topp, /system, /reset, /save, /quit)
python -m fable_lm.chat

# single shot
python -m fable_lm.chat --prompt "hello" --temperature 0.3

# evaluate
python -m fable_lm.eval
```

The chat CLI loads `checkpoints/best.pt` by default; `--checkpoint` takes any
other path (e.g. `models/fable-mini-final.pt`, the shipped fp16 copy).

## Hugging Face

The model is on the Hub and loadable like any HF model (verified: Hub weights
are bit-identical to the original checkpoint, max |Δlogits| = 0.0):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("dev-sandakelum/fable-mini", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("dev-sandakelum/fable-mini", trust_remote_code=True)

messages = [{"role": "system", "content": "You are a concise, warm assistant."},
            {"role": "user", "content": "Hi!"}]
enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                              return_dict=True, return_tensors="pt")
out = model.generate(enc.input_ids, max_new_tokens=64, do_sample=True,
                     temperature=0.8, top_p=0.9, repetition_penalty=1.2,
                     pad_token_id=256, eos_token_id=257)
print(tok.decode(out[0][enc.input_ids.shape[1]:]))
```

- **Repo**: https://huggingface.co/dev-sandakelum/fable-mini
  (model card with YAML metadata, honest eval, usage snippet).
- **Files**: `config.json` + `auto_map` (remote code `configuration_fable.py` /
  `modeling_fable.py`, also usable locally without remote code via `import hf_fable`),
  `model.safetensors` (fp32; tied `head.weight` dropped and re-tied on load),
  `generation_config.json`, `tokenizer.json` / `tokenizer_config.json` /
  `special_tokens_map.json` / `chat_template.jinja`.
- **Parity**: logits of the HF wrapper vs `fable_lm.FableLM` are bit-identical
  on the same weights (max |Δ| = 0.0, fp32, T ∈ {7, 33, 128, 512}); tokenizer
  ids identical on 40/40 corpus + edge-case texts; chat template renders
  exactly `fable_lm`'s prompt format (`<system>…</system>` + `User:`/`Assistant:` turns).
- **`generate()`** supports `temperature`, `top_p`, `top_k`, `max_new_tokens`,
  `repetition_penalty` with a `DynamicCache` KV cache; greedy `generate()` equals
  manual cached argmax stepping (tested).

### Converting weights

```bash
# .pt checkpoint -> hf_model/ (config.json, model.safetensors, generation_config.json)
python scripts/convert_to_hf.py                          # fp32, parity-checked
python scripts/convert_to_hf.py --checkpoint models/fable-mini-final.pt --fp16

# repo tokenizer -> HF fast tokenizer files (+ corpus-wide parity verification)
python scripts/convert_tokenizer_to_hf.py
```

### Uploading to the Hub

```bash
export HF_TOKEN=hf_...           # or put it in .env (gitignored; see .env.example)
python scripts/push_to_hub.py --repo-id dev-sandakelum/fable-mini --dry-run
python scripts/push_to_hub.py --repo-id dev-sandakelum/fable-mini [--private]
```

The script checks the token, verifies the repo owner matches the token identity,
and only uploads the allowlisted artifact files. This exact flow was used for the
live upload (commit `feb6b86f`).

### Deploying the Space

```bash
python app.py                   # local: loads Hub repo id (or --model hf_model)
python scripts/deploy_space.py --repo-id dev-sandakelum/fable-mini-demo
```

`app.py` is a Gradio Blocks chat (streaming via `TextIteratorStreamer`,
sampling sliders, system-prompt box) that loads through plain transformers.
`space/README.md` carries the Space metadata (`sdk: gradio`, `app_file: app.py`,
pinned `requirements.txt`).

**Known limitation (documented honestly)**: creating a Gradio Space on the free
tier currently returns HTTP 402 — HF requires a PRO subscription for
Gradio/Docker Spaces on free cpu-basic. The Space is fully packaged and the
deploy script works (verified dry-run); hosting it needs a PRO subscription or
a static Space. The demo itself runs locally: `python app.py`.

A Colab notebook that loads the model through transformers is at
`notebooks/fable_mini_colab.ipynb` (also in `download/`).

## Architecture summary

Decoder-only transformer, pre-LN, causal self-attention, learned absolute
positions, GELU feed-forward, tied input/output embeddings.

Shipped config: vocab 2048 (byte-level BPE, 1789 merges), context 512,
4 layers, d_model 256, 4 heads, d_ff 768 — **3,285,504 parameters**.

Trained with AdamW (lr 3e-4, wd 0.1 on matrices), 100-step linear warmup,
cosine decay to 10%, gradient-norm clip 1.0, batch 8×512 tokens.

## Known limitations

- **Tiny corpus, memorization**: trained on ~400 KB of repo text; train loss
  3.6 vs val loss 5.0 in run1 — the model largely memorizes. Samples are
  recognizably in-corpus register but not coherent language.
- **No real capabilities**: no reliable QA, reasoning, tool use, or memory.
  The Fable behaviors in `STUDY.md` are imitated as *format* only; `EVAL.md`
  shows what that actually looks like.
- **Safety**: the miniature has **no** reliable safety behavior. Dialogue data
  contains refusal *styles*, but a 3.3M model cannot be trusted to refuse
  anything. Do not use it for anything real.
- **Context**: 512 tokens; longer conversations are cropped from the left.
- **Speed**: ~4.5k tokens/s training throughput on 2 CPUs; generation is
  interactive-ish for short replies.

## Next steps

- More and better dialogue data (the single highest-leverage change), then a
  larger data multiplier on the Fable doc.
- A second model trained *without* the system-prompt document, to separate
  "XML-soup register" from "chat register".
- Eval additions: distinct-n, a small held-out prompt set scored by format
  adherence rather than eyeballing.
- Try weight tying off / d_ff 1024 at 6 layers if RAM allows.

## Provenance and honesty

The only primary source about Fable is `doc/Claude-Fable-5.1.md` (a system-prompt
style document for "Claude Fable 5.1"). **No Fable 5.2 research notes, system
card, or benchmarks were available** — every 5.2-specific choice is a labeled
ASSUMPTION in `research/fable-5.2-delta-assumptions.md` and `DESIGN.md` §7.
All evaluation numbers in `EVAL.md` are measured on this machine, on this corpus.
