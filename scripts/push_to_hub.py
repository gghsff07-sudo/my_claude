"""Upload the HF-format model folder to the Hugging Face Hub.

Usage:
    python scripts/push_to_hub.py --repo-id dev-sandakelum/fable-mini          # upload
    python scripts/push_to_hub.py --repo-id dev-sandakelum/fable-mini --dry-run
    python scripts/push_to_hub.py --repo-id dev-sandakelum/fable-mini --private

Reads HF_TOKEN from the environment (or a .env file next to the repo root).
The token is never printed, logged, or written anywhere.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FOLDER = REPO / "hf_model"

# files that make up the hub model repo
ALLOWED = {
    "README.md",
    "config.json",
    "configuration_fable.py",
    "modeling_fable.py",
    "model.safetensors",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
}


def load_env_file() -> None:
    """Populate os.environ from .env if HF_TOKEN is not already set."""
    if os.environ.get("HF_TOKEN"):
        return
    env_file = REPO / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload hf_model/ to the HF Hub")
    ap.add_argument("--repo-id", required=True,
                    help="e.g. dev-sandakelum/fable-mini")
    ap.add_argument("--private", action="store_true",
                    help="create the repo as private")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be uploaded, do nothing")
    ap.add_argument("--folder", default=str(FOLDER),
                    help="model folder to upload (default: hf_model/)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"model folder not found: {folder}")

    files = sorted(p for p in folder.iterdir() if p.is_file())
    unexpected = [p.name for p in files if p.name not in ALLOWED]
    if unexpected:
        raise SystemExit(f"unexpected files in {folder} (not in upload allowlist): {unexpected}")
    if not (folder / "model.safetensors").is_file():
        raise SystemExit(f"no model.safetensors in {folder} — run scripts/convert_to_hf.py first")

    total = sum(p.stat().st_size for p in files)
    print(f"repo:   {args.repo_id} ({'private' if args.private else 'public'})")
    print(f"folder: {folder}")
    for p in files:
        print(f"  {p.name:28} {p.stat().st_size:>12,} B")
    print(f"total:  {total:,} B ({total / 1e6:.2f} MB), {len(files)} files")

    if args.dry_run:
        print("\n[dry-run] no upload performed. To upload:")
        print(f"  HF_TOKEN=<your-token> python scripts/push_to_hub.py "
              f"--repo-id {args.repo_id}{' --private' if args.private else ''}")
        return

    load_env_file()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN not set. Export it (or put it in .env, see .env.example) and retry."
        )

    from huggingface_hub import HfApi, whoami

    me = whoami(token=token)
    user = me["name"]
    print(f"authenticated as: {user}")
    owner = args.repo_id.split("/")[0]
    if owner != user and owner not in [org.get("name") for org in me.get("orgs", [])]:
        raise SystemExit(f"token belongs to '{user}', but repo-id owner is '{owner}'")

    api = HfApi(token=token)
    url = api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    print(f"repo ready: {url}")

    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(folder),
    )
    print(f"uploaded: {commit}")
    print(f"model page: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
