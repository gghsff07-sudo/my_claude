# Changelog

## v0.4.1 — "train it to 30B": the attempt, the squeeze, the measured wall

The follow-up request was to actually train toward 30B. This release does
everything that is physically doable on this machine and records, with
receipts, exactly where the wall is. The headline measurements: the direct
30B construction attempt is SIGKILLed by the OOM killer during layer 2 of 49
(layer 1 alone — 635M params — takes RSS to 2.6 GB; each layer's fp32 AdamW
state alone is 9.87 GB vs 4.36 GB total RAM), and the largest model this
machine can train at all is the 99M rung — 0.33% of the 30B target.

### Added

- **`fable_lm/attempt.py` + `fable-attempt` entry point** — the 30B
  construction attempt, for the record: a child process builds the real
  parameter volume (real Block modules, real initializers, every page
  touched) layer by layer with per-layer RSS reporting; the parent records
  where the OOM killer stops it (SIGKILL is uncatchable in Python, which is
  precisely why the attempt runs as a child). Output:
  `results/attempt_30b.json` — exit code, signal, layers completed, RSS
  trace, and the fraction of the 30B target that existed at death (2.10%).
- **Squeeze rungs `99m` (10L × d896) and `154m` (12L × d1024)** — the ladder
  is densified near the memory ceiling so the automation can climb as close
  to the wall as RAM allows.
- **Calibrated memory model** — `training_peak_bytes` activation term
  12 → 30 · B·T·d·L·4 bytes, back-fit to all four v0.4.0 observations
  (15m@b8 2.2 GB RSS, 59m@b4 2.97 GB survived, 59m@b8 SIGKILLed). The old
  coefficient passed a rung that then killed the process; the new one errs
  on the side of the OOM killer.
- **`--only RUNG` and `--save-ckpt`** for `fable-scale`: train a single rung
  (results go to `results/scale_<name>.*` so the full-ladder record is
  never clobbered) and save its weights to
  `checkpoints/scale/<name>-final.pt`.
- 10 new tests (101 total): squeeze-rung labels and ordering, ladder gap
  filling, and attempt-progress parsing.

### Measured on this machine (2 threads, 4.16 GB RAM)

Squeeze ladder `--target 30b --safety 0.85` (40 steps per rung):

| rung | params | batch | train loss | val loss | tok/s | peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 3.3m | 3,285,504 | 8 | 7.67 → 6.35 | 6.38 | 4,300 | 1.01 GB |
| 15m | 15,167,232 | 8 | 7.70 → 6.31 | 6.36 | 1,243 | 2.28 GB |
| 59m | 58,646,016 | 4 | 7.81 → 6.57 | 6.49 | 429 | 2.97 GB |
| 99m | 98,714,112 | 2 | 7.76 → 6.73 | 6.59 | 249 | 3.09 GB |
| 154m | 153,729,024 | — | BLOCKED: ~3.21 GB estimated peak vs 2.77 GB budget | | | |

The 99M rung was then trained for real: 150 steps at batch 1, 76,800 tokens
(~0.65 epochs of the 118,784-token corpus), train loss 7.75 → 7.09, val
6.385, 224 tok/s, peak RSS 2.48 GB; weights saved to
`checkpoints/scale/99m-final.pt` (see `results/scale_99m.json`). The corpus,
not the architecture, is what limits it — a 30B-class model wants ~605B
tokens, about 5,000,000x more than the repo has. Reaching the recorded run
took three attempts, each honest: the 150-step run at batch 2 was SIGKILLed
mid-run (the ceiling is marginal — the 40-step squeeze pass had survived in
a warm process), a re-run was blocked by a dropped --safety flag, and the
batch-1 run completed cleanly.

The 30B ledger, measured: 98.7M trained = 0.33% of the target; the attempt
died with 2.10% of the parameter volume resident; full 30B AdamW fp32
training needs ~492 GB peak vs 4.36 GB total (~113x); at 1/params-scaled
throughput the 30B rung would run at ~0.5 tok/s (~38,000 years for a
Chinchilla-style budget). Reaching 30B requires multi-GPU clusters and a
trillion-token-scale dataset — not patience on this CPU box.

## v0.4.0 — 30B scale-up automation (config + ladder, hardware-honest)

Adds the machinery the 30B request needs: a verified 30,000,000,000-parameter
configuration and an automated training ladder that walks from the shipped 3.3M
model toward any parameter target, training every rung that fits the machine
and stopping with exact requirements where it does not. The shipped 3.3M model
and all its artifacts are unchanged.

Stated plainly: a 30B model cannot be trained on this class of hardware, and
this release does not pretend otherwise. 30.23B parameters need ~483 GB of RAM
for fp32 AdamW training (16 bytes/param plus activations) and cluster-scale
compute — this machine has 4 GB and 2 CPU threads. What v0.4.0 delivers is the
verified target config, the automation that reaches it on hardware that can,
and the honest measurement of where this hardware stops.

### Added

- **`fable_lm/scale.py` + `fable-scale` entry point** — the automation:
  - a curated rung ladder 3.3M → 15M → 59M → 344M → 1.2B → 4.1B → 30B, each
    rung's parameter count computed by `ModelConfig.param_count()` and
    cross-checked against real instantiation when trained;
  - `--target` accepts `30b`, `15m`, `3e9`, or a raw integer;
  - heuristic peak-RAM estimation (16 bytes/param for AdamW fp32 states +
    activation estimate), auto batch-fitting (8 → 4 → 2 → 1) against a
    safety-capped fraction of available RAM;
  - trains each feasible rung for a smoke budget (`--steps`, default 40),
    recording train/val loss, tokens/s, elapsed, and measured peak RSS;
  - rungs that cannot fit are marked BLOCKED with the exact requirement, and
    the run stops there — "train until the target or the hardware ceiling,
    whichever comes first";
  - `--dry-run` plans the whole 30B path without instantiating any model
    (this is how the 30B rung is planned on a 4 GB machine);
  - results land in `results/scale_ladder.csv` + `results/scale_ladder.json`.
- **The 30B preset** (LADDER["30b"]): 49 layers, d_model 7168, 56 heads
  (head dim 128), d_ff 28672 — 30,233,097,216 parameters exactly, verified by
  the same `param_count()` formula that the tests cross-check against
  instantiation on the smaller rungs.
- 12 tests (`tests/test_scale.py`): param-count arithmetic vs instantiation,
  30B preset bounds and shape constraints, ladder monotonicity and target
  coverage, target parsing, memory-estimate gating, and dry-run safety.

### Measured on this machine (2 threads, 4.16 GB RAM)

`python -m fable_lm.scale --target 30b` (default 40 smoke steps per rung):

| rung | params | batch | train loss | val loss | tok/s | peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 3.3m | 3,285,504 | 8 | 7.67 → 6.35 | 6.38 | 4,554 | 1.01 GB |
| 15m | 15,167,232 | 8 | 7.70 → 6.31 | 6.36 | 1,235 | 2.23 GB |
| 59m | 58,646,016 | 2 (auto-fit) | 7.84 → 6.47 | 6.56 | 401 | 2.23 GB |
| 344m | 343,839,744 | — | BLOCKED: ~5.95 GB estimated peak vs 1.53 GB budget | | | |

The 59M rung is the largest this machine can train — 17.9× the shipped model —
and every trained rung's instantiated parameter count matched
`param_count()` exactly (measured peak RSS is the process high-water mark:
the CPU allocator does not return memory between rungs). Full numbers in
`results/scale_ladder.json`; the automation's console summary states the 30B
requirement outright. A first attempt at the 59M rung was killed by the Linux
OOM killer (SIGKILL, exit 137) under the original 75% RAM budget — the gate
now defaults to 50% and results persist incrementally after every rung.

## v0.3.0 — data upgrade and honest retrain

The v0.2.0 report's top recommendation was to retrain with the leak-free
recipe and more dialogue data. This release does exactly that: the dialogue
corpus roughly doubles, the model and tokenizer are retrained, and every
shipped artifact is regenerated from the new run. The headline number moves
the honest way — val loss 4.16 (leak-flattered) → 5.09 (clean, on a larger
held-out split) — see EVAL.md's run3 section for why that is an improvement.

### Added

- **12 new dialogue files** (`data/dialogues/12-23_*.txt`, ~4.7k words ≈ the
  existing corpus): multi-turn context use, clarification questions, being
  corrected gracefully, structured comparisons, step-by-step instructions,
  plain definitions, summaries, brainstorming, smalltalk variety, gratitude
  and closings, refusals with alternatives, and calibrated hedging.
- **`--top-k` in the chat CLI** and a `top_k` parameter on
  `reply()`/`reply_stream()`; `/topk N` and `/maxnew N` REPL commands.
- **`--checkpoint` in `fable_lm.eval`** (evaluate any checkpoint, not just
  `checkpoints/best.pt`).
- **`--fresh-tokenizer` in `fable_lm.train`**: retrain the BPE merges when the
  corpus changes (the old behavior silently reused the cached tokenizer).
- **`fable-bench` console entry point** — the generation benchmark moved into
  `fable_lm/bench.py` (installed with the package); `scripts/bench_generation.py`
  is now a thin wrapper.

### Fixed

- **Prompt-crop edge case** (`fable_lm/chat.py`): with `--max-new` ≥
  context_len, `prompt[-(ctx - max_new - 1):]` turned into a positive slice
  and kept a long prompt. The budget is now clamped, falling back to the last
  `ctx` tokens (generation restarts its cache internally).
- **Two non-hermetic HF tests** (`tests/test_hf_compat.py`): the greedy test
  compared `generate()` — which silently inherited repetition_penalty 1.2 from
  the saved `generation_config.json` — against a penalty-free manual loop, and
  the sampling test asserted an exact length although a sampled EOS can end
  generation early. Both passed on run2's weights by luck and failed on
  run3's; they now pin the generation parameters explicitly.
- **Shipped-artifact bloat** (`scripts/to_fp16.py`): v0.2.0 added optimizer
  state to every checkpoint, but the fp16 shipper only converted the model
  weights — the fp32 AdamW moments rode along and the shipped model grew from
  7.6 MB to 34 MB. The optimizer state is now dropped; run3's shipped model
  is 7.6 MB again.

### Changed

- **run3 retrain** (leak-free recipe, 800 steps, early stop at 650): best val
  loss 5.0893 (ppl 162.27) at step 399; train 4.3574 (ppl 78.05). Shipped
  checkpoint `models/fable-mini-final.pt`, tokenizer `data/tokenizer.json`,
  HF folder, `results/eval_results.json`, and `results/run3_log.csv` all
  regenerated from it; HF logits parity re-verified at 0.0 and tokenizer
  parity 53/53.
- Corpus 18 → 30 documents; val split 1 → 3 documents. The tokenizer is
  retrained on the expanded corpus (still vocab 2048 / 1789 merges).
- Test suite 76 → 79; `ruff check .` still clean.
- `fable_lm.__version__` 0.2.0 → 0.3.0; EVAL.md gained a run3 section and the
  run2 honesty note was sharpened with the confirmed leak.

## v0.2.0 — audit-driven upgrade

A full-codebase audit found the v0.1.0 pipeline sound (53/53 tests, HF parity
0.0) but with several real bugs and missed opportunities. This release fixes
the bugs and upgrades performance, UX, and packaging. The from-scratch ethos
and all honesty documentation are unchanged.

### Fixed

- **Train/val leak** (`fable_lm/train.py`): `--dialogue-mult` repeated ALL
  dialogue documents — including those assigned to the validation split —
  into the training data, so run2's val loss (4.16) was measured partly on
  memorized text. Extra copies now come from the train split only
  (`fable_lm.data.split_docs` + `is_dialogue`). Numbers in `EVAL.md` for run2
  remain as measured at v0.1.0; re-run to get a leak-free figure.
- **Checkpoint portability** (`fable_lm/chat.py`): checkpoints store the
  tokenizer path as an absolute path from the training machine; loading the
  shipped checkpoint anywhere else raised FileNotFoundError (the fallback only
  triggered when the key was absent). The loader now tries the stored path,
  the repo-relative location, and the checkpoint's directory, in order.
- **Gradio multi-turn history** (`app.py`): the chatbot log was unpacked as
  (user, assistant) PAIRS while gradio 6 supplies a flat list of
  {role, content} dicts — each dict was destructured into a bogus
  {"role": "user", "content": "user"} + {"role": "assistant", ...} pair, so
  the model saw garbage as conversation history. History is now converted
  with `fable_lm.demo_utils.messages_from_history` (dicts or tuples).
- **Dead `--model` flag** (`app.py`): `python app.py --model hf_model` parsed
  the flag only AFTER the model had already loaded from the Hub id, so it was
  ignored. `--model` is now parsed before loading (tolerantly, so
  Space-injected arguments don't break it).
- **REPL crashes** (`fable_lm/chat.py`): `/temp` without a value, `/temp abc`
  or `/save` without a filename raised IndexError/ValueError. Commands are
  parsed by `parse_command` (never raises) and print a usage line instead;
  added `/help` and `/seed N`.
- **Repetition penalty semantics** (`fable_lm/generate.py`): the penalty
  divided ALL logits of seen tokens, which RAISES the probability of
  negative-logit tokens. Now matches HF `GenerationMixin`: positive logits
  divided, negative multiplied — so the CLI and the HF wrapper sample the
  same way.
- **Resume dropped optimizer state** (`fable_lm/train.py`): `--resume` restored
  only weights — AdamW restarted with zero moments and the warmup schedule
  replayed. Checkpoints (best/step/final) now carry `opt` state and resumes
  restore it.
- **CSV log header on resume** (`fable_lm/train.py`): resuming a run whose
  `log.csv` did not exist appended rows with no header.
- **`torch.load` hardening**: `chat.py`, `eval.py`, `train.py` (resume) now
  load checkpoints with `weights_only=True` (torch ≥2.6's safe default).

### Added

- **KV-cache generation** (`fable_lm/model.py` `FableKVCache` +
  `forward_with_cache`): each sampled token costs one single-token forward
  instead of a full re-run of the context. `generate_stream`/`generate`
  default to cached decoding; verified greedy-identical to the uncached path
  on the shipped model. `use_cache=False` still selects the original loop.
- **Streaming output**: `generate_stream` yields tokens as sampled;
  `fable_lm.chat.reply_stream` yields decoded text deltas (partial multibyte
  characters handled via longest-stable-prefix re-emit). The chat REPL now
  streams by default (`--no-stream` for the old behavior).
- **`/help` and `/seed N`** REPL commands.
- **distinct-1/2/3 diversity metrics** in `fable_lm/eval.py` (per sample and
  aggregate) — addresses a documented "next step".
  `results/eval_results.json` was regenerated: loss/ppl are unchanged
  (3.6208/4.1600), sampled texts differ at the token level because cached
  decoding changes low-order float bits in the logits.
- **`scripts/restore_checkpoint.py`**: upcast the shipped fp16 model to a
  fp32 `checkpoints/best.pt` so a fresh clone can run the full suite and the
  HF conversion (also used by CI).
- **`scripts/bench_generation.py`**: reproducible generation-speed benchmark.
- **Packaging**: `pyproject.toml` (installable package, `fable-train` /
  `fable-chat` / `fable-eval` entry points, `[hf]` / `[demo]` / `[dev]`
  extras, pytest + ruff config), root `conftest.py` (pytest from any cwd),
  `Makefile`, `LICENSE` (MIT), GitHub Actions CI (lint + restore + convert +
  full test suite).

### Changed

- BPE merge selection is now one C-level `max` (same deterministic
  tie-break), speeding up tokenizer training.
- `fable_lm.__version__` 0.1.0 → 0.2.0; test suite grew from 53 to 76 tests
  (KV-cache correctness, streaming equality, command parsing, history
  conversion, distinct-n, no-leak split). `ruff check .` is clean and runs in
  CI together with the full suite.

## v0.1.0

Initial release: tokenizer, model, data, training, chat, eval, HF wrapper,
Hub upload, Gradio demo, Colab notebook. See README.md.
