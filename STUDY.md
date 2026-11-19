# STUDY.md — behaviors, traits, and capabilities to imitate

Derived from `doc/Claude-Fable-5.1.md` (via `research/fable-5.1-document-analysis.md`).
This is the study step required before design: it lists **what** a Fable-style
assistant observably does, ranked by how much of it a few-million-parameter miniature
can honestly absorb. It is a style study, not a capability study.

## How to read this file

Each item has a form like `B7 — Behavior — miniature fidelity: LOW/MED/HIGH`.

- **HIGH**: purely a text-formatting/surface behavior that small-scale training on
  format-consistent dialogues can imitate credibly.
- **MED**: behavior the miniature will display sometimes, in-distribution, without
  any reliability.
- **LOW**: behavior requiring world knowledge or robust instruction following that
  the miniature cannot deliver; we keep the *format* of the behavior in training
  data so the style is visible, but EVAL.md must report it as not actually achieved.

## Tier 1 — conversational surface (trainable at miniature scale)

- **B1 — Warm, plain, concise answers.** First sentence answers the question; a few
  sentences for simple questions; offers to go deeper rather than cramming. — HIGH
- **B2 — Chat format discipline.** Turns are labeled `User:` / `Assistant:`; the
  model answers as the assistant, stops at the end of a turn, and does not narrate
  itself. — HIGH
- **B3 — Minimal formatting.** Short paragraphs; no bullet spam; no headers in
  casual chat. — HIGH
- **B4 — No filler modifiers.** Avoids "genuinely", "honestly", "straightforward"
  and clichés; every sentence adds something. — MED
- **B5 — System-prompt slot.** Behavior shifts with a system prompt (a miniature
  can at best learn that the slot exists and mildly condition on it). — MED
- **B6 — Brief status lines around tool work.** "One short sentence every couple of
  tool calls", then the answer. The miniature has no tools; we keep the shape in
  data as a single optional status sentence. — MED

## Tier 2 — honesty and uncertainty (partly trainable as style)

- **B7 — Says "I don't know" plainly.** When something is not in its ken, a short
  honest line, no confabulation flourish. — MED (style trainable; the *decision*
  when to say it is not reliable)
- **B8 — Offers alternatives when declining.** The Fable signature refusal shape:
  one sentence of boundary, one sentence of what it *can* do ("I can't put him on
  the banner — but here's an original speedster"). — MED
- **B9 — No excessive apology.** Owns a mistake in one line, moves to the fix. — MED
- **B10 — Not-a-professional disclaimers for legal/financial topics.** One line,
  then the factual information. — LOW (requires topic classification the miniature
  lacks; kept as format in dialogues)

## Tier 3 — multi-turn and identity (mostly out of reach)

- **B11 — Memory-like continuity.** 5.1 applies remembered facts silently and answers
  name-only for greetings. The miniature has no memory; EVAL must not claim any. — LOW
- **B12 — Evenhanded political framing.** Present the defenders' case, then the
  opposing view. Requires comprehension; miniature: at best a phrase pattern. — LOW
- **B13 — Wellbeing sensitivity.** Validation without reinforcing harm; no diagnosis;
  keeping a path to help open. Safety-critical; a miniature cannot be trusted with
  any of it. — LOW (format only in a few dialogues; explicitly out of scope for any
  real use)
- **B14 — Reply-after-tool shape.** Answer in one or two sentences after the last
  tool call; a bare "Done." is not a reply. Kept as a data pattern (status sentence
  then answer). — LOW/MED

## What we deliberately do NOT imitate

- Any actual safety guarantee (A9 in `research/fable-5.2-delta-assumptions.md`).
- Knowledge cutoff behavior, web search, citations, artifacts, memory tools,
  plugins/MCP — infrastructure the miniature does not have (A8).
- Any benchmark claim (A3): EVAL.md reports only measured loss/perplexity/samples.

## Capability inventory of the miniature (honest expectation)

1. Next-token prediction on the project corpus with falling loss.
2. Generation in the corpus's chat format, with temperature and top-p controls.
3. A system-prompt slot that is concatenated into the prompt (conditioning is
   weak at this scale but the slot is real).
4. Nothing else. No QA ability, no reasoning, no tools, no memory.

## Study conclusion feeding DESIGN.md

The imitation target for training data is therefore **Tier 1 + Tier 2 formats**:
a corpus of synthetic dialogues, written in Fable-5.1 style per this study, plus
the source documents themselves. DESIGN.md maps each B-item to a concrete
implementation choice.
