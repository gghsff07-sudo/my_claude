"""Convert a training checkpoint to a compact fp16 copy for shipping."""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]

src = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "checkpoints" / "best.pt"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "models" / "fable-mini-final.pt"

ck = torch.load(src, map_location="cpu")
ck["model"] = {k: v.to(torch.float16) for k, v in ck["model"].items()}
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(ck, dst)
print(f"{src.name} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB, step {ck['step']}, "
      f"best_val {ck.get('best_val', float('nan')):.4f})")
