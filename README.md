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
| Tests (24: tokenizer, model, data, training, generation) | `tests/` |

## Setup

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest tests/ -q        # 24 tests
```

Python 3.13, PyTorch 2.15.0+cpu. No other dependencies (stdlib `re`, `json`,
`csv`, `math`, `argparse` only).

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
