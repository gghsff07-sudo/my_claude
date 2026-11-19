"""Convert a fable_lm .pt checkpoint into a Hugging Face model folder.

Usage:
    python scripts/convert_to_hf.py                       # checkpoints/best.pt -> hf_model/
    python scripts/convert_to_hf.py --checkpoint models/fable-mini-final.pt --fp16
    python scripts/convert_to_hf.py --out hf_model

Writes: config.json (with auto_map for trust_remote_code), model.safetensors,
generation_config.json, and self-contained remote-code files
(configuration_fable.py, modeling_fable.py, __init__.py) so the folder can be
uploaded to the Hub as-is.

Then verifies logits parity: the ORIGINAL fable_lm.FableLM on the checkpoint
weights vs the converted model RELOADED via AutoModelForCausalLM.from_pretrained
(the full transformers path: safetensors round trip + remote-code import).
Max |delta logits| must be < 1e-4 (fp32 is expected to be exactly 0.0).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fable_lm.model import FableLM, ModelConfig  # noqa: E402
from hf_fable import FableConfig, FableLMHeadModel  # noqa: E402


def build_config(ck_cfg: dict) -> FableConfig:
    return FableConfig(
        vocab_size=ck_cfg["vocab_size"],
        max_position_embeddings=ck_cfg["context_len"],
        num_hidden_layers=ck_cfg["n_layers"],
        hidden_size=ck_cfg["d_model"],
        num_attention_heads=ck_cfg["n_heads"],
        intermediate_size=ck_cfg["d_ff"],
        dropout=ck_cfg.get("dropout", 0.0),
        # auto_map -> works on the Hub with trust_remote_code=True
        auto_map={
            "AutoConfig": "configuration_fable.FableConfig",
            "AutoModelForCausalLM": "modeling_fable.FableLMHeadModel",
        },
    )


def convert(checkpoint: Path, out_dir: Path, fp16: bool, tokenizer_dir: Path | None) -> None:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_d = ck["cfg"]
    print(f"checkpoint: {checkpoint} (step {ck.get('step')}, best_val {ck.get('best_val'):.4f})")
    print(f"cfg: {cfg_d}")

    model = FableLMHeadModel(build_config(cfg_d))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert unexpected == [], f"unexpected keys: {unexpected}"
    assert set(missing) <= {"head.weight"}, f"missing keys: {missing}"  # tied
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) config.json (auto_map included via config attributes)
    model.config.save_pretrained(out_dir)

    # 2) model.safetensors — drop the tied head.weight (it shares storage
    #    with tok_emb.weight; safetensors forbids shared tensors and
    #    from_pretrained re-ties on load)
    sd = {k: v for k, v in model.state_dict().items() if k != "head.weight"}
    if fp16:
        sd = {k: v.to(torch.float16) for k, v in sd.items()}
    save_file(sd, out_dir / "model.safetensors", metadata={"format": "pt"})
    size_mb = (out_dir / "model.safetensors").stat().st_size / 1e6
    print(f"model.safetensors: {len(sd)} tensors, {size_mb:.2f} MB, dtype fp{16 if fp16 else 32}")

    # 3) generation_config.json — defaults mirroring the chat CLI (t=0.8, p=0.9)
    from transformers import GenerationConfig

    gen = GenerationConfig(
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        max_new_tokens=128,
    )
    gen.save_pretrained(out_dir)

    # 4) self-contained remote-code files (verified import style: relative
    #    import at snapshot root works with spec_from_file_location)
    for f in ("configuration_fable.py", "modeling_fable.py"):
        shutil.copy(REPO / "hf_fable" / f, out_dir / f)

    print(f"converted -> {out_dir}/")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name:28} {p.stat().st_size:>10,} B")


@torch.no_grad()
def verify_parity(checkpoint: Path, out_dir: Path, n_samples: int = 4) -> float:
    """Original FableLM vs AutoModelForCausalLM.from_pretrained(hf_model)."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    orig = FableLM(ModelConfig(**ck["cfg"]))
    orig.load_state_dict(ck["model"])
    orig.eval()

    # full transformers path: local dir + trust_remote_code (as on the Hub)
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True)
    hf.eval()

    torch.manual_seed(123)
    max_diff = 0.0
    V = orig.cfg.vocab_size
    for i in range(n_samples):
        T = [7, 33, 128, 512][i % 4]
        ids = torch.randint(0, V, (2, T))
        lo, _ = orig(ids)
        lh = hf(input_ids=ids).logits
        d = (lo - lh).abs().max().item()
        max_diff = max(max_diff, d)
        print(f"  parity sample {i + 1}/{n_samples} (B=2, T={T}): max |dlogits| = {d:.3e}")
    print(f"MAX |dlogits| = {max_diff:.3e}  (require < 1e-4: {'PASS' if max_diff < 1e-4 else 'FAIL'})")
    if max_diff >= 1e-4:
        raise SystemExit("parity check FAILED")
    return max_diff


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert fable_lm checkpoint to HF format")
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints" / "best.pt"))
    ap.add_argument("--out", default=str(REPO / "hf_model"))
    ap.add_argument("--fp16", action="store_true", help="store fp16 weights (default fp32)")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"no checkpoint at {ckpt}")
    out_dir = Path(args.out)
    convert(ckpt, out_dir, fp16=args.fp16, tokenizer_dir=None)
    if not args.skip_parity:
        print("\nverifying logits parity (original vs from_pretrained) ...")
        verify_parity(ckpt, out_dir)


if __name__ == "__main__":
    main()
