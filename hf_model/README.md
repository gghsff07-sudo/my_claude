---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- educational
- tiny-model
- from-scratch
- byte-bpe
- decoder-only
language:
- en
metrics:
- perplexity
base_model: []
---

# fable-mini

A **3,285,504-parameter** decoder-only transformer, trained from scratch as an
educational miniature of the "Fable" assistant style (see the companion GitHub
repo [gghsff07-sudo/my_claude](https://github.com/gghsff07-sudo/my_claude) for
the full study, design, training and evaluation write-ups).

**This is a small educational model, not a real assistant, and not equivalent
to Fable 5.2 or any production LLM.** It was built to demonstrate a complete,
honest, from-scratch pipeline on a CPU budget. It did not learn to answer
questions or produce coherent sentences (see "Evaluation" below).

## Model description

- Architecture: pre-LN decoder-only transformer — 4 layers, hidden size 256,
  4 attention heads (head dim 64), FFN 768, GELU, learned absolute positions,
  bias-free fused QKV, tied input/output embeddings. Context length 512.
- Tokenizer: byte-level BPE (vocab 2,048 = 256 bytes + 1789 learned merges +
  3 specials), trained from scratch on the training split only.
- Weights: run2 final checkpoint (step 749), fp32 safetensors. The tied LM head
  weight is stored once (`tok_emb.weight`) and re-tied on load.
- Generation: `model.generate()` supports `temperature`, `top_p`, `top_k`,
  `max_new_tokens`, `repetition_penalty` with a KV cache.

Special tokens: `<|pad|>`=256, `<|eos|>`=257, `<|doc|>`=258 (document separator,
doubling as BOS). No `<|unk|>` — byte-level coverage means every byte has an id.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("<user>/fable-mini", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("<user>/fable-mini", trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are a concise, warm assistant."},
    {"role": "user", "content": "Hi!"},
]
inputs = tok.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
)

out = model.generate(
    inputs.input_ids,
    max_new_tokens=64,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    pad_token_id=256,
    eos_token_id=257,
)
text = tok.decode(out[0][inputs.input_ids.shape[1]:])
print(text)
```

The model's chat format (what the chat template renders) is plain-text turns:

```
<system>You are a concise, warm assistant.</system>
User: Hi!
Assistant: ...
```

## Training data

Trained on an 18-document in-repo corpus (90/10 document-level split, seed 42):

- The "Claude Fable 5.1" reference document (~270 KB)
- 5 research/design documents written during the project
- 11 short synthetic Fable-style dialogue files (User:/Assistant: turns)

run2 upweighted the dialogue files 6× to reinforce the chat register. The total
corpus is a few hundred KB of English text — far too small for general language
ability, by design. No web scrape, no external datasets.

Training: AdamW, peak lr 3e-4, 100-step warmup, cosine decay to 10%, gradient
clip 1.0, batch 8×512 tokens, CPU only, ~750 steps in total across two runs
(600 plain + 150 resumed with dialogue upweighting).

## Evaluation (honest numbers)

| Split | Loss | Perplexity |
| --- | --- | --- |
| train | 3.6209 | 37.37 |
| val | 4.1600 | 64.07 |

Perplexity is **on this corpus only** — a corpus-fit number, not a
general-language number. Caveats, stated plainly:

- run2's val split shares genre and phrasing patterns with the upweighted
  training dialogues, so val ppl 64.07 flatters the model somewhat.
- The train/val gap (3.62 vs 4.16) shows substantial memorization.
- Sample generations are register-imitation, not coherent answers. A verbatim
  example (temperature 0.8): *"I's this is story is that't leave as the
  one-15" and you to beeny and hurborry is usually search — segoy, or I'm the
  user't render it't being matter how you need to were food and a ca- Clalated
  — your holarning are mode?"*
- The chat scaffold (User:/Assistant: turns, `<|eos|>` stops) works
  mechanically — short turns, clean stops — but the content is word soup.

The full evaluation with sample tables is [EVAL.md in the GitHub repo]
(https://github.com/gghsff07-sudo/my_claude/blob/main/EVAL.md).

## Intended use

Education and research: studying a complete from-scratch LLM pipeline
(tokenizer training, transformer implementation, training loop, evaluation,
Hugging Face packaging) at a size that trains in minutes on a laptop CPU.
Good for: unit-testing HF integration code, experimenting with decoding
strategies, teaching.

## Limitations and safety

- **Not a chatbot.** Output is incoherent word salad imitating the training
  register. It cannot answer questions, follow instructions, or hold a topic.
- **No safety behavior exists.** The model has no refusal capability, no
  calibration, and no alignment training. Do not use it for anything real.
- **Not equivalent to Fable 5.2** or any production assistant; the name refers
  to the style study, not a capability claim.
- English-only training data; 512-token context; 2K vocab.

## Acknowledging the source style

The "Fable" reference documents analyzed during this project are Anthropic-style
assistant-writing material; this model is an independent educational miniature
with no connection to, or endorsement by, Anthropic.
