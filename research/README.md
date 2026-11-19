# research/ — Fable source material and findings

This folder holds every piece of Fable-related information available for this project,
plus an explicit record of what was **not** available.

## Provenance summary

| Source | Status | Where |
| --- | --- | --- |
| `doc/Claude-Fable-5.1.md` (system-prompt-style document, 2195 lines, ~270 KB) | Available (in repo) | `doc/Claude-Fable-5.1.md` |
| Fable **5.2** research notes | **Not available** — nothing retrieved, not retried per data rules | — |
| Fable **5.2** system card text | **Not available** — nothing retrieved, not retried per data rules | — |
| Fable **5.2** benchmark findings | **Not available** — nothing retrieved, not retried per data rules | — |

Because no Fable 5.2-specific material exists in this environment, the Fable 5.1
document is the **only** primary source. Every Fable-5.2-specific statement in this
project is labeled `ASSUMPTION` in [fable-5.2-delta-assumptions.md](fable-5.2-delta-assumptions.md)
and in `DESIGN.md`. The project makes **no** capability claims about the real Fable 5.2.

## Files

- [fable-5.1-document-analysis.md](fable-5.1-document-analysis.md) — structured findings
  extracted from `doc/Claude-Fable-5.1.md`: model identity, product facts, feature
  inventory, and the full behavioral spec (the part that matters most for a miniature).
- [fable-5.2-delta-assumptions.md](fable-5.2-delta-assumptions.md) — every gap between
  "Fable 5.2" and what we actually know, each with a labeled assumption and reason.

## Training corpus inventory (what text we actually have)

The miniature is trained only on text present in this repository at training time:

1. `doc/Claude-Fable-5.1.md` — the primary source document itself.
2. Everything in `research/` — these notes.
3. `STUDY.md` and `DESIGN.md` — the derived behavior/study and design documents.
4. A set of **synthetic chat dialogues** written for this project
   (`data/dialogues/`), composed to demonstrate the observable Fable-style behaviors
   (concise warm answers, brief refusals with alternatives, honest uncertainty,
   reply-after-tool-call shape) in a `<|user|>` / `<|assistant|>` format.

That is the entire corpus — small by design, and the main reason the trained
miniature is a style imitation, not a capable assistant. See `EVAL.md` for the
consequences.
