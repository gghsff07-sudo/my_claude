# DESIGN.md — Fable 5.2 (miniature) design

Mapping from the behaviors in `STUDY.md` to concrete implementation choices in this
repo. Everything here is built for: 2 CPUs, 4 GB RAM, PyTorch 2.15.0+cpu.

## 1. Architecture

**Decoder-only transformer, from scratch in PyTorch.**

| Component | Choice | Why |
| --- | --- | --- |
| Token type | Byte-level BPE, vocab ~2,048 (configurable) | Lossless on any text incl. the XML-ish tags in the corpus (A6) |
| Positional encoding | Learned absolute positions | Simplest robust choice at this scale; RoPE adds nothing measurable here (A4) |
| Layers | 6 blocks, `d_model` 256, 4 heads, `d_ff` 1024, pre-LN | ~3.9M params in embeddings+layers — fits RAM/CPU budget with headroom |
| Attention | Causal self-attention, no cross-block flash tricks | Plain matmul attention is fastest to get right on CPU |
| Norm | LayerNorm pre-block + final LayerNorm | Standard pre-LN stability |
| Activation | GELU | Standard, cheap |
| Tied embeddings | Output head shares weight with input embedding | Saves ~524k params; standard for tiny models |
| Context length | 512 tokens (A5) | Corpus documents are chunked to this; longer is pure cost here |
| Attention window | Full causal within 512 | 512² attention is cheap enough on CPU |

### Parameter budget

An early 6-layer sketch landed at ~7.1M — above the "few million" target — so the
shipped default is **4 layers, d_model 256, d_ff 768 (vocab 2048, ctx 512)** →
exactly **3,285,504 params** (`ModelConfig.param_count()`, verified against the
live model in `tests/test_model.py`). The 6-layer variant stays available via
config for a heavier run.

## 2. Trait → implementation map (from STUDY.md)

| Trait | Implementation choice |
| --- | --- |
| B1 concise warm answers | Dialogue corpus in Fable style: short assistant turns, first sentence answers |
| B2 chat format | Fixed turn markers: `User:` / `Assistant:` lines, `<|eos|>` special token ends a turn |
| B3 minimal formatting | Dialogue data written with plain paragraphs, no markdown in chat turns |
| B4 no filler | Corpus written avoiding "gently/honestly/genuinely" filler; eval samples checked |
| B5 system-prompt slot | Prompt = `<system>…</system>\n` + dialogue history; CLI `--system` flag |
| B6 status-then-answer | A few dialogues include a one-line status sentence before the answer |
| B7 "I don't know" | Dialogues containing honest short uncertainty lines |
| B8 refusal-with-alternative | Dialogues in the 5.1 refusal style: boundary sentence + alternative sentence, no bullets |
| B9 no groveling | Mistake-repair dialogue pair in corpus |
| B10 disclaimer format | One legal/financial dialogue with the one-line disclaimer format |
| B11 memory | NOT implemented (out of scope, A8) — EVAL states this |
| B12 evenhandedness | One dialogue showing case-then-counter-case format; no capability claim |
| B13 wellbeing | A small number of format-only dialogues; no safety claim (A9) |
| B14 reply-after-tools | Covered by B6 status-sentence pattern; no real tools exist |

## 3. Training data design

Corpus = concatenation of, in this order, separated by `<|doc|>` markers:

1. `doc/Claude-Fable-5.1.md` (the primary source; teaches the register and vocabulary
   of the behavioral spec)
2. `research/*.md`, `STUDY.md`, `DESIGN.md` (project register)
3. `data/dialogues/*.txt` — synthetic Fable-style dialogues (the Tier 1+2 formats)

Documents are split into 512-token chunks for training; dialogues are kept whole
where possible (they are short). Split: 90% train / 10% val, split at the document
level, deterministic (seed 42).

## 4. Training pipeline

- Optimizer: AdamW (lr 3e-4 default, weight decay 0.1 on non-LN/non-bias params).
- LR schedule: linear warmup (100 steps) → cosine decay to 10% of peak.
- Gradient clipping: global norm 1.0.
- Batch: 8 sequences × 512 tokens, gradient accumulation if needed.
- Checkpointing: every 200 steps + final; `checkpoints/` is gitignored except the
  final small checkpoint if <10 MB (state_dict of 2.9M params fp32 ≈ 12 MB → ship
  fp16 ≈ 6 MB, or document location per step 7).
- Logging: step, loss, val loss, tokens/s, elapsed — CSV at `runs/<name>/log.csv`
  plus stdout.
- Early stop: patience 5 evals on val loss (keep best checkpoint).

## 5. Inference / chat CLI

- Prompt assembly: system prompt (default: a one-line Fable-style persona line) +
  `User:`/`Assistant:` history + `Assistant:` cue.
- Sampling: temperature (default 0.8), top-p (default 0.9), top-k 50, repetition
  penalty 1.2; stop at `<|eos|>` or `User:` marker or max new tokens (default 128).
- REPL commands: `/temp X`, `/topp X`, `/system TEXT`, `/reset`, `/quit`, `/save`.
- Single-shot mode: `python -m fable_lm.chat --prompt "..."`.

## 6. Evaluation design (honest)

1. Final train/val loss + perplexity (e^loss) on held-out corpus chunks.
2. Fixed set of 6 prompts (2 casual, 2 knowledge-ish, 1 decline-shaped, 1 system-prompt
   test), generated at temperature 0.8 and 0.3, recorded verbatim in EVAL.md.
3. Limits stated: no external benchmarks, no human eval, tiny corpus, style only.

## 7. Assumptions (full list with reasons in research/fable-5.2-delta-assumptions.md)

A1–A10 are incorporated by reference. The load-bearing ones for this design:
- A4: architecture family unknown for real 5.2 → standard decoder-only transformer.
- A5: context 512 (environment-driven).
- A6: byte-level BPE.
- A7: corpus restricted to repo text + synthetic dialogues.
- A8/A9: no tools/memory/safety claims.

## 8. Repository layout

```
fable_lm/            # package
  __init__.py
  tokenizer.py       # byte-level BPE
  model.py           # transformer
  data.py            # corpus building + dataset
  train.py           # training loop
  chat.py            # CLI chat
  generate.py        # sampling logic
configs/
  small.json         # default ~2.9M config
tests/
  test_tokenizer.py
  test_model.py
  test_train_smoke.py
data/
  dialogues/*.txt    # synthetic Fable-style dialogues
checkpoints/         # gitignored *.pt
runs/                # gitignored training logs
```
