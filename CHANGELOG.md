# Changelog

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
