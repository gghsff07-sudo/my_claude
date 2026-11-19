# System card — "Fable 5.2 (miniature)" research artifact

**Status: this is a project-authored system card for the miniature model built in this
repo. It is NOT the system card of any real Anthropic model, and it makes no claims
about the real Fable 5.1 or 5.2.** It exists because the task deliverable list
includes "the system card text" and the real 5.2 card was not available (see
[fable-5.2-delta-assumptions.md](fable-5.2-delta-assumptions.md), A2).

## Model identity

- **Name**: Fable 5.2 (miniature) — internal project name.
- **What it is**: a decoder-only transformer language model of a few million
  parameters, written from scratch in PyTorch, trained on the text inside this
  repository (the Fable 5.1 document, these research notes, and synthetic chat
  dialogues).
- **What it is not**: a capable assistant. It will not answer questions reliably,
  will not follow the Fable behavioral spec beyond shallow stylistic imitation, and
  must not be used for anything real.

## Intended use

Research and education: to show the full pipeline — tokenizer, model, training,
inference, evaluation — in a form that runs on 2 CPUs and 4 GB RAM. The only honest
use is reading the code and running the training/eval yourself.

## Capability summary (measured, see EVAL.md when present)

- Trained next-token cross-entropy loss and perplexity on the project corpus: see
  `EVAL.md` for the actual numbers from the final run.
- Qualitative: short samples that mimic the corpus's chat format
  (`User:`/`Assistant:` turns, a system-prompt slot) at best; repetitive or
  degenerate text at worst. No factual QA, no reasoning, no tool use, no memory.

## Safety posture

- The training corpus includes refusals written in the 5.1 document's *style* (brief,
  conversational, offers an alternative). This is stylistic imitation only.
- The miniature has **no** reliable safety behavior. It can and will produce
  nonsense, and nothing it says should be trusted or deployed.
- The real 5.1 document's actual safety measures (child safety, weapons, substances,
  malicious code, copyright, wellbeing guardrails) are recorded in
  [fable-5.1-document-analysis.md](fable-5.1-document-analysis.md) as research
  findings; the miniature does not implement them at any guaranteed level.

## Training data

Only material present in this repository at training time:
`doc/Claude-Fable-5.1.md`, `research/*`, `STUDY.md`, `DESIGN.md`, and synthetic
dialogues in `data/dialogues/`. No external or web-downloaded corpora.

## Evaluation plan

Honest, small: final training/validation loss, perplexity, and a handful of sample
generations with their visible failure modes. Limits stated alongside every number.
No benchmark comparisons to any real model.
