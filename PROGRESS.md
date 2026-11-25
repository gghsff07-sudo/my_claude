# PROGRESS — running checklist

Steps are marked done when **committed and pushed**.

## Push blocker (RESOLVED)

The initially provided PAT had Contents: read-only (pushes returned 403). The
token's permissions were later updated on GitHub's side; pushes now work.

## Checklist

- [x] Step 0: repo setup — remote added, remote `main` merged (unrelated histories), torch 2.15.0+cpu installed
- [x] Step 1: `research/` — 5.1 document analysis, 5.2 delta assumptions, miniature system card
- [x] Step 2: `STUDY.md` — behaviors/traits to imitate
- [x] Step 3: `DESIGN.md` — architecture, context length, assumptions
- [x] Step 4: byte-level BPE tokenizer + 7 tests
- [x] Step 5: transformer + 7 tests (3,285,504 params exact)
- [x] Step 6: training pipeline + 11 dialogue files + smoke tests (24 tests total)
- [x] Step 7: run1 (600 steps, val 5.0214/ppl 151.6) + run2 (dialogue-upweighted, val 4.1600/ppl 64.07); fp16 checkpoint shipped in `models/`
- [x] Step 8: chat CLI (temperature, top-p, system prompt, REPL commands) + tests
- [x] Step 9: `EVAL.md` — honest evals with limits, verbatim samples, scorecard
- [x] Step 10: `README.md` + deliverables in `download/`

**All steps complete.**

## v0.2.0 (2026-11-25) — audit-driven upgrade

- [x] Full-code audit: 10 findings (train/val leak, checkpoint portability, gradio
      history bug, dead --model flag, REPL crashes, rep-penalty semantics, resume
      dropping optimizer state, CSV header, torch.load hardening, no KV cache)
- [x] KV-cached generation (3.6-4.2x faster), streaming REPL, distinct-n eval,
      packaging (pyproject, entry points), LICENSE, CI, Makefile, CHANGELOG
- [x] 53 -> 76 tests, ruff clean; HF parity 0.0 and eval numbers preserved

## v0.3.0 (2026-11-25) — data upgrade + retrain

- [x] 12 new dialogue files (12-23): multi-turn context, clarifications,
      corrections, comparisons, step-by-step, definitions, summaries,
      brainstorming, smalltalk, closings, boundaries, hedging (~doubles dialogue corpus)
- [x] Code: top_k exposed in CLI + reply(), /topk + /maxnew commands,
      eval --checkpoint, train --fresh-tokenizer, fable-bench entry point,
      prompt-crop hardening for max_new >= context_len
- [x] Retrain run3: fresh tokenizer + leak-free dialogue upweighting
- [x] Artifacts regenerated: fp16 model, HF folder, eval results, logs

## v0.4.0 (2026-11-25) — 30B scale-up automation

- [x] 30B preset: 49L x d7168 x h56 x ff28672 = 30,233,097,216 params exactly
      (arithmetic-verified; cross-checked against instantiation on small rungs)
- [x] fable_lm/scale.py + fable-scale: ladder automation, RAM gating, batch
      auto-fit, OOM-tolerant stop, dry-run planning, results/scale_ladder.*
- [x] 12 new tests (91 total); shipped 3.3M model and artifacts unchanged
- [x] Measured on this box: trains 3.3M (4548 tok/s) / 15M (1270) / 59M
      (430, batch auto-fit) rungs, blocks at 344M (~5.95 GB peak vs 1.57 GB
      budget); 30B needs ~492 GB peak + cluster GPUs; first 59M attempt was
      OOM-killed (SIGKILL) at the 75% budget — gate now 50%, incremental writes

