# PROGRESS — running checklist

Steps are marked done when **committed and pushed**.

## Push blocker (RESOLVED)

The initially provided PAT had Contents: read-only (pushes returned 403). The
token's permissions were later updated on GitHub's side; pushes now work. Steps 1–3
were pushed retroactively once access was granted.

## Checklist

- [x] Step 0: repo setup — remote added, remote `main` merged (unrelated histories), torch 2.15.0+cpu installed
- [x] Step 1: `research/` — 5.1 document analysis, 5.2 delta assumptions, miniature system card (pushed)
- [x] Step 2: `STUDY.md` — behaviors/traits to imitate (pushed)
- [x] Step 3: `DESIGN.md` — architecture, context length, assumptions (pushed)
- [ ] Step 4: tokenizer (byte-level BPE) + tests
- [ ] Step 5: transformer model + forward-pass unit test
- [ ] Step 6: training pipeline (data, LR schedule, clipping, checkpoints, logging)
- [ ] Step 7: training run, final checkpoint, loss logs
- [ ] Step 8: chat/CLI interface (temperature, top-p, system prompt)
- [ ] Step 9: honest evals, `EVAL.md`
- [ ] Step 10: `README.md`, deliverables in `download/`
