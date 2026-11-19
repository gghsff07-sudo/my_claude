---
title: fable-mini
emoji: 🧸
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: A 3.3M-param educational miniature (incoherent by design)
---

# fable-mini Space

Gradio chat demo for [dev-sandakelum/fable-mini](https://huggingface.co/dev-sandakelum/fable-mini),
a 3,285,504-parameter decoder-only transformer trained from scratch as an
educational miniature of the "Fable" assistant style.

**This is an educational demo.** The model does not produce coherent answers —
output is register-imitating word soup, which is the honest, documented result
at this scale and corpus (see the
[model card](https://huggingface.co/dev-sandakelum/fable-mini) for eval numbers).
The demo exists to show the model loads and generates through plain
`transformers`, with streaming.

## Space layout

- `app.py` — Gradio Blocks app: loads via `AutoModelForCausalLM`/
  `AutoTokenizer` from a Hub repo id or local path, applies the chat template,
  streams tokens with `TextIteratorStreamer`.
- `requirements.txt` — pinned dependencies.

## Deploying from the source repo

The Space repo content is three files from
[gghsff07-sudo/my_claude](https://github.com/gghsff07-sudo/my_claude):

```
app.py                (repo root)
requirements.txt      (repo root)
space/README.md   ->  README.md (this file, renamed at the Space root)
```

```bash
python - <<'EOF'
from huggingface_hub import HfApi
import os, shutil, tempfile

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("dev-sandakelum/fable-mini-demo", repo_type="space",
                space_sdk="gradio", exist_ok=True, private=False)
with tempfile.TemporaryDirectory() as d:
    shutil.copy("app.py", f"{d}/app.py")
    shutil.copy("requirements.txt", f"{d}/requirements.txt")
    shutil.copy("space/README.md", f"{d}/README.md")
    api.upload_folder(repo_id="dev-sandakelum/fable-mini-demo",
                      repo_type="space", folder_path=d)
EOF
```
