"""Build notebooks/fable_mini_colab.ipynb — a Colab notebook that loads
fable-mini through transformers, and verify its code cells actually run."""

import json
import nbformat
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "name": "fable-mini via transformers"},
    "accelerator": "None",
}

cells = []
cells.append(nbformat.v4.new_markdown_cell(
"""# fable-mini — load through `transformers`

This notebook loads [**fable-mini**](https://huggingface.co/dev-sandakelum/fable-mini),
a 3,285,504-parameter decoder-only transformer trained from scratch as an
educational miniature of the "Fable" assistant style
([GitHub: gghsff07-sudo/my_claude](https://github.com/gghsff07-sudo/my_claude)).

**Set your expectations honestly**: this model does not produce coherent
answers. It was trained on a few hundred KB of in-repo text; output is
register-imitating word soup. That is the documented, expected result at this
scale (see the model card for eval numbers: val ppl 64.07 *on its own corpus*).
The point of this notebook is the loading pipeline, not the output quality.
"""))
cells.append(nbformat.v4.new_code_cell(
"""# Transformers 5.x is required (the model's remote code targets the 5.x API)
%pip install -q "transformers>=5.23.0" safetensors tokenizers"""))
cells.append(nbformat.v4.new_markdown_cell("## Load tokenizer + model"))
cells.append(nbformat.v4.new_code_cell(
"""from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "dev-sandakelum/fable-mini"

tok = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(REPO_ID, trust_remote_code=True)
model.eval()

print(type(model).__name__, f"{model.num_params():,} params")
print("vocab:", tok.vocab_size, "| pad/eos/bos:", tok.pad_token_id, tok.eos_token_id, tok.bos_token_id)"""))
cells.append(nbformat.v4.new_markdown_cell(
"""## Chat template + generation

The chat template renders the training format: a `<system>…</system>` line,
then `User:` / `Assistant:` plain-text turns. Generation supports temperature,
top-p, top-k and repetition penalty, with a KV cache."""))
cells.append(nbformat.v4.new_code_cell(
"""import torch

messages = [
    {"role": "system", "content": "You are a concise, warm assistant."},
    {"role": "user", "content": "Hi!"},
]

enc = tok.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
print("prompt text:", repr(tok.decode(enc.input_ids[0])))

torch.manual_seed(42)
out = model.generate(
    enc.input_ids,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)
text = tok.decode(out[0][enc.input_ids.shape[1]:])
print("generated:", repr(text))"""))
cells.append(nbformat.v4.new_markdown_cell(
"""## What you should see

The generated text is **word soup** — fragments in the training register
("the user", "the person", apostrophe-s constructions, XML-ish tags) with no
coherent meaning. A typical sample looks like:

> *"I's this is story is that't leave as the one-15" and you to beeny and
> hurborry is usually search — segoy, or I'm the user't render it't being
> matter how you need to were food and a ca- Clalated — your holarning are
> mode?"*

The mechanics work — prompt formatting, sampling controls, `<|eos|>` stopping —
but a 3.3M model on ~400 KB of text cannot learn language. For the full honest
evaluation see [EVAL.md](https://github.com/gghsff07-sudo/my_claude/blob/main/EVAL.md)."""))
cells.append(nbformat.v4.new_markdown_cell(
"""## Bonus: logits parity with the original implementation

The HF artifact is a format conversion of the original from-scratch model, not
a retrain — logits are bit-identical. (Optional: requires the original repo's
checkpoint; download it from the GitHub repo if you want to run this cell.)"""))
cells.append(nbformat.v4.new_code_cell(
"""# Optional: verify parity against the original fable_lm implementation.
# Clone the source repo and place checkpoints/best.pt where the repo expects it.
try:
    import sys, torch
    sys.path.insert(0, "/content/my_claude")  # adjust if needed
    from fable_lm.model import FableLM, ModelConfig

    ck = torch.load("/content/my_claude/checkpoints/best.pt", map_location="cpu", weights_only=False)
    orig = FableLM(ModelConfig(**ck["cfg"]))
    orig.load_state_dict(ck["model"])
    orig.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, 2048, (2, 64))
    with torch.no_grad():
        lo, _ = orig(ids)
        lh = model(input_ids=ids).logits
    print("max |dlogits|:", (lo - lh).abs().max().item(), "(expected 0.0)")
except Exception as e:
    print("skipped:", type(e).__name__, str(e)[:120])"""))

nb.cells = cells
out_path = REPO / "notebooks" / "fable_mini_colab.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, out_path)
print("wrote", out_path)

# validate + execute the code cells cumulatively (as a notebook kernel would),
# skipping repo-dependent and pip cells
import subprocess, sys

cumulative = ""
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source
    if "my_claude" in src or "%pip" in src:
        print(f"cell {i}: skipped (repo-dependent / pip)")
        continue
    # point the Hub repo id at the local converted folder for offline execution
    local = src.replace('"dev-sandakelum/fable-mini"', '"hf_model"')
    cumulative += "\n" + local
    r = subprocess.run([sys.executable, "-c", cumulative], capture_output=True, text=True, cwd=REPO)
    status = "OK" if r.returncode == 0 else f"FAIL: {r.stderr[-300:]}"
    print(f"cell {i}: {status}")
    assert r.returncode == 0, f"cell {i} failed"
print("NOTEBOOK CODE CELLS VERIFIED")
