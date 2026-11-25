"""Restore checkpoints/best.pt (fp32) from the shipped fp16 model.

The training checkpoints are gitignored (regenerable); a fresh clone only has
models/fable-mini-final.pt. This upcasts it back to fp32 so the full test
suite (including the HF-artifact tests) and scripts/convert_to_hf.py can run
out of the box.

Usage:
    python scripts/restore_checkpoint.py [src] [dst]
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]

src = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "models" / "fable-mini-final.pt"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "checkpoints" / "best.pt"

if not src.is_file():
    raise SystemExit(f"no checkpoint at {src}")

ck = torch.load(src, map_location="cpu", weights_only=False)
ck["model"] = {k: v.to(torch.float32) for k, v in ck["model"].items()}
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(ck, dst)
print(f"restored {dst} (step {ck['step']}, best_val {ck.get('best_val', float('nan')):.4f})")
