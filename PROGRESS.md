# PROGRESS — running checklist

Steps are marked done when **committed locally**. Push status is tracked separately
(see the blocker note below).

## Push blocker (open)

The provided fine-grained PAT authenticates as `gghsff07-sudo` (repo owner) but its
**Contents permission is read-only**. `git push` and GitHub API writes both return
403 ("Resource not accessible by personal access token"). No workaround attempted,
per the task's data rules. All steps are committed locally on `main`; the full
history pushes with a single `git push origin main` once the token has Contents:
read+write on `gghsff07-sudo/my_claude`.

## Checklist

- [x] Step 0: repo setup — remote added, remote `main` merged (unrelated histories), torch 2.15.0+cpu installed
- [x] Step 1: `research/` — 5.1 document analysis, 5.2 delta assumptions, miniature system card (commit `step 1`)
- [x] Step 2: `STUDY.md` — behaviors/traits/capabilities to imitate (commit `step 2`)
- [ ] Step 3: `DESIGN.md` — architecture, context length, assumptions
- [ ] Step 4: tokenizer (byte-level BPE) + tests
- [ ] Step 5: transformer model + forward-pass unit test
- [ ] Step 6: training pipeline (data, LR schedule, clipping, checkpoints, logging)
- [ ] Step 7: training run, final checkpoint, loss logs
- [ ] Step 8: chat/CLI interface (temperature, top-p, system prompt)
- [ ] Step 9: honest evals, `EVAL.md`
- [ ] Step 10: `README.md`, deliverables in `download/`
