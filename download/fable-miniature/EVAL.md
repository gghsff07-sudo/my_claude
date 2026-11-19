# EVAL.md — evaluation results and their limits

All numbers below were measured on this machine (2 CPUs, 4 GB RAM, PyTorch
2.15.0+cpu) on this repository's corpus, on 2026-11-19. Every claim is either a
measured number or a visibly failed sample — nothing here is a capability claim.

## Setup

- Model: 3,285,504 params — 4 layers, d_model 256, 4 heads, d_ff 768, tied
  embeddings, context 512 (DESIGN.md §1).
- Tokenizer: byte-level BPE, vocab 2,048 (1789 learned merges + 256 bytes + 3
  specials), trained on the train split only.
- Corpus: 18 documents — the Fable 5.1 doc (~270 KB), 5 research/design docs,
  and 11 synthetic dialogue files. Split 90/10 at the document level, seed 42.
- Training: AdamW, peak lr 3e-4, 100-step warmup, cosine decay to 10%, grad-norm
  clip 1.0, batch 8×512 tokens.

## Run history

### run1 — plain mixture, 600 steps

| step | lr | train loss | val loss | val ppl |
| --- | --- | --- | --- | --- |
| 50 | 1.5e-4 | 6.8291 | 6.8090 | 905.98 |
| 100 | 3.0e-4 | 5.8996 | 6.0757 | 435.17 |
| 200 | 2.75e-4 | 4.9078 | 5.3465 | 209.87 |
| 400 | 1.24e-4 | 4.0521 | 5.0577 | 157.22 |
| 500 | 5.63e-5 | 3.9103 | **5.0214** | **151.63** |
| 600 | 3.0e-5 | 3.5772 | 5.0534 | 156.55 |

Best val loss 5.0214 (ppl 151.63) at step 500; shipped as the run1 artifact.
Full curve in `results/run1_log.csv`.

### run2 — resumed from run1, dialogues upweighted 6×, 300 more steps

| step | lr | train loss | val loss | val ppl |
| --- | --- | --- | --- | --- |
| 550 | 1.07e-4 | 4.1588 | 4.5324 | 92.98 |
| 600 | 8.13e-5 | 3.9029 | 4.4520 | 85.80 |
| 700 | 4.36e-5 | 3.4721 | 4.2513 | 70.20 |
| 750 | 3.35e-5 | 3.5857 | **4.1600** | **64.07** |
| 800 | 3.0e-5 | 3.5305 | 4.1752 | 65.05 |

**Final model (shipped in `models/fable-mini-final.pt`, fp16):** step 749,
train loss 3.6209 (ppl 37.37), val loss 4.1600 (ppl 64.07). Full curve in
`results/run2_log.csv`.

**Important honesty note on run2's val number.** The 6× dialogue upweighting
multiplies the *training* frequency of the dialogue genre. The val split still
contains one dialogue file with the same format and phrasing patterns as the
upweighted train dialogues, so part of the improvement from 5.02→4.16 is genre
familiarity, not generalization. The train/val gap (3.62 vs 4.16) also shows
substantial memorization of this small corpus. Treat val ppl 64 as a corpus-fit
number, not a general-language number.

## Sample generations (final model, verbatim)

Prompts are fixed (2 casual, 2 knowledge, 1 decline-shaped, 1 system-test);
temperature 0.8 and 0.3, top-p 0.9, max 100 tokens, default system prompt.
Reproduced exactly as generated (seed fixed; `results/eval_results.json` holds
the machine-readable copy).

**casual @ 0.8 — "Hi! How are you today?"**
> I's this is story is that't leave as the one-15" and you to beeny and hurborry is usually search — segoy, or I'm the user't render it't being matter how you need to were food and a ca- Clalated — your holarning are mode?

**casual @ 0.3 — "what should I make for dinner tonight?"**
> I's friendarlication of your much is the user's the person's something isn's the person't relicted to beended in this is a crong What't have being on the session.

**knowledge @ 0.8 — "what's the capital of France?"**
> I's faster about that striend — Cood. Cautwies Are your prose is songlying it't rendered brefe, so never frid rather than a single user't beended in the million-certaindiven'll make the user is the person'd easy on your own Kether what the actor to food gel on a date content in

**knowledge @ 0.3 — "explain what a database index is"**
> I can's the friendar bre being on the user is a hafey to the user't have been. The user's the person's a crites a gar, or sital of live the sessed to familiven, so I't beended in Ad beended in
>   clegal much fund to food garn's a fri

**decline-shaped @ 0.8 — "can you give me the lyrics to Bohemian Rhapsody?"**
> I's fast from the water Awater bading got it is
> <example>
>
> ES Srepdf (e is usually today'm the response about my masent down a craworning baring goldflus- The expected the person's a hour "d eff. Tlacks my maticidesignorter is that live pythorn

**system-test @ 0.3 — "write me a haiku about mountains"**
> I's fundiearing is — the user's the bre not to the user's can's the person't relicted.

### What the samples show

- The model did **not** learn to answer questions, hold a topic, or produce
  coherent sentences. Failure is the expected result at this scale and corpus.
- It did absorb surface features of the corpus register: conversational openers
  ("I's", "I can's"), apostrophe-s constructions, the persona words "the user",
  "the person", "search", "request", "content", XML tags from the Fable doc, and
  turn-adjacent phrases like "the session". At t=0.3 the decline-shaped prompt
  produces something refusal-*shaped* ("...not to the user's can's the person't
  relicted.") without any actual refusal semantics.
- The chat-format scaffold (User:/Assistant: turns, `<|eos|>` stops) works
  mechanically — the CLI reliably produces a short turn and stops — but the
  *content* of the turn is word soup.
- Temperature 0.3 is more grammatical-looking than 0.8 but just as wrong; this
  is memorized n-gram texture, not knowledge.

## Honest scorecard against STUDY.md

| Trait | Status |
| --- | --- |
| B2 chat format (turns, stops) | **Partially achieved** — format mechanics work, content does not |
| B1/B3/B4 warm concise plain answers | Not achieved — register words appear, sentences are incoherent |
| B7 saying "I don't know" | Not achieved |
| B8 refusal-with-alternative | Not achieved — only refusal-shaped fragments at low temperature |
| B5 system-prompt conditioning | Not measurably achieved (not tested beyond sample inspection; slot exists mechanically) |
| B11 memory, B12 evenhandedness, B14 tools | Not attempted — out of scope by design (A8) |
| Safety (B13 and any real refusal) | **No safety behavior exists.** Do not use this model for anything real. |

## Limits of this evaluation

1. Perplexity is on this repo's corpus only — it says nothing about general
   language ability.
2. Samples were eyeballed, not human-scored; no distinct-n or other automatic
   metrics were computed.
3. The system-prompt slot was exercised only through the default persona line;
   no A/B comparison of system prompts was run.
4. Single seed (42) for training, single seed (7) for sampling; no variance
   reported.
5. run2's val split overlaps in genre with upweighted training data (noted
   above), so cross-run comparisons slightly flatter run2.
