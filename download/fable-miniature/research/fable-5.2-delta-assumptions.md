# Fable 5.2 delta — gaps and labeled assumptions

The task targets "Fable 5.2". The only primary source available is the Fable **5.1**
document (`doc/Claude-Fable-5.1.md`). No Fable 5.2 research notes, system card text,
or benchmark findings were retrieved. Per the data rules, those were not retried and
no workarounds were attempted. Every 5.2-specific design input below is therefore an
ASSUMPTION, each with a one-line reason.

| # | Gap (unknown about real Fable 5.2) | ASSUMPTION made for this project | One-line reason |
| --- | --- | --- | --- |
| A1 | Existence, release status, and naming of "Fable 5.2" | Treat "Fable 5.2" as the project's internal name for a miniature successor to the observable 5.1 behaviors | Nothing about a 5.2 was available, so the only defensible reading is "imitate the 5.1 we can actually read" |
| A2 | 5.2 system card text | A short system card written by this project, describing the **miniature** honestly (capability, corpus, limits) | A system card must exist for the deliverable, and honesty about the miniature is the only safe content |
| A3 | 5.2 benchmark findings | None exist; `EVAL.md` will contain only this project's own measured loss/perplexity and samples | Reporting real numbers measured here is required; inventing benchmarks is forbidden |
| A4 | 5.2 architecture (layers, width, positions, MoE vs dense) | Decoder-only transformer, causal self-attention, learned absolute positions, pre-LN, configurable size — chosen for this environment | Standard, well-understood architecture that fits 2 CPUs / 4 GB RAM and matches the task's suggestion |
| A5 | 5.2 context length | 512 tokens for training and inference | Tiny corpus and tiny memory budget make longer context pure cost with no measured benefit |
| A6 | 5.2 tokenizer | Byte-level BPE with a 2k–4k merge vocabulary trained on the corpus | Byte-level guarantees lossless round-trip on any text, including the XML-ish tags in the corpus |
| A7 | 5.2 training data and scale | Only the text in this repo plus synthetic dialogues written for the project | Data rule: use only information already gathered; no external corpora |
| A8 | 5.2 feature set (tools, memory, artifacts, …) | A chat CLI with sampling controls and a system-prompt slot; no tools, no memory, no artifacts | A few-million-parameter model cannot operate real tool infrastructure, and claiming it would be dishonest |
| A9 | 5.2 safety posture | Keep the 5.1 document's refusal *style* (brief, conversational, offers an alternative); no claim of 5.1/5.2-grade safety | Safety behavior at real scale needs training data and evaluation we do not have; only the surface style is imitable |
| A10 | 5.2 "additional safety measures for dual-use capabilities" specifics | Out of scope entirely | The specifics are not in any available source |

## Handling rules (from the task's DATA RULES)

1. Only the Fable 5.1 document and project-generated text are used as facts.
2. Unavailable items above were skipped, not retried.
3. Every gap got a labeled assumption with a reason, listed above and in `DESIGN.md`.
4. No capability claims beyond what `EVAL.md` measures.
