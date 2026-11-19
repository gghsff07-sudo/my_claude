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
