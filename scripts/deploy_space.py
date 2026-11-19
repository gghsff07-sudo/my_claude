"""Deploy the Gradio Space for fable-mini (HF Space repo type).

Usage:
    HF_TOKEN=... python scripts/deploy_space.py --repo-id dev-sandakelum/fable-mini-demo
    HF_TOKEN=... python scripts/deploy_space.py --repo-id dev-sandakelum/fable-mini-demo --dry-run

Uploads app.py, requirements.txt and space/README.md (as README.md) to a Space
repo. Reads HF_TOKEN from the environment or .env; never prints it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO))
from scripts.push_to_hub import load_env_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy fable-mini Gradio Space")
    ap.add_argument("--repo-id", required=True, help="e.g. dev-sandakelum/fable-mini-demo")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = {
        "app.py": REPO / "app.py",
        "requirements.txt": REPO / "requirements.txt",
        "README.md": REPO / "space" / "README.md",  # Space metadata frontmatter
    }
    for name, p in files.items():
        if not p.is_file():
            raise SystemExit(f"missing {name} ({p})")
        print(f"  {name:18} {p.stat().st_size:>8,} B  <- {p.relative_to(REPO)}")

    if args.dry_run:
        print("\n[dry-run] nothing created. To deploy:")
        print(f"  HF_TOKEN=<token> python scripts/deploy_space.py --repo-id {args.repo_id}")
        return

    load_env_file()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (export it or put it in .env)")

    from huggingface_hub import HfApi, whoami

    user = whoami(token=token)["name"]
    owner = args.repo_id.split("/")[0]
    if owner != user:
        raise SystemExit(f"token belongs to '{user}', but repo-id owner is '{owner}'")

    api = HfApi(token=token)
    url = api.create_repo(
        repo_id=args.repo_id, repo_type="space", space_sdk="gradio",
        private=args.private, exist_ok=True,
    )
    with tempfile.TemporaryDirectory() as d:
        for name, p in files.items():
            shutil.copy(p, Path(d) / name)
        commit = api.upload_folder(
            repo_id=args.repo_id, repo_type="space", folder_path=d
        )
    print(f"space ready: {url}")
    print(f"commit: {commit}")
    print(f"space url: https://{args.repo_id.split('/')[0]}-{args.repo_id.split('/')[1]}.hf.space")


if __name__ == "__main__":
    main()
