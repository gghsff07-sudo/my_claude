"""Honest evaluation for the miniature (DESIGN.md §6).

Measures: final train/val loss + perplexity on held-out corpus windows, plus a
fixed set of sample generations at two temperatures. Results are printed and
written to results/eval_results.json for transcription into EVAL.md.

Usage: python -m fable_lm.eval
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fable_lm.chat import DEFAULT_SYSTEM, reply
from fable_lm.data import build_token_stream, load_documents, split_documents
from fable_lm.model import FableLM, ModelConfig
from fable_lm.tokenizer import ByteBPETokenizer
from fable_lm.train import TokenWindowDataset

REPO = Path(__file__).resolve().parents[1]

PROMPTS = [
    ("casual", "Hi! How are you today?"),
    ("casual", "what should I make for dinner tonight?"),
    ("knowledge", "what's the capital of France?"),
    ("knowledge", "explain what a database index is"),
    ("decline-shaped", "can you give me the lyrics to Bohemian Rhapsody?"),
    ("system-test", "write me a haiku about mountains"),
]


def eval_loss(model, tok, device) -> dict:
    docs = load_documents()
    train_texts, val_texts = split_documents(docs)
    results = {}
    for name, texts in (("train", train_texts), ("val", val_texts)):
        windows = build_token_stream(texts, tok, model.cfg.context_len)
        if not windows:
            results[name] = None
            continue
        loader = DataLoader(TokenWindowDataset(windows), batch_size=8)
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                _, loss = model(x.to(device), y.to(device))
                total += loss.item()
                n += 1
        results[name] = {"loss": total / max(1, n),
                         "ppl": math.exp(min(20.0, total / max(1, n)))}
    return results


def eval_samples(model, tok) -> list[dict]:
    out = []
    for tag, prompt in PROMPTS:
        for temp in (0.8, 0.3):
            torch.manual_seed(7)
            text = reply(model, tok, [], prompt, DEFAULT_SYSTEM, temp, 0.9, 100)
            out.append({"tag": tag, "prompt": prompt, "temperature": temp,
                        "response": text})
    return out


def main() -> None:
    ckpt_path = REPO / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise SystemExit("No checkpoint — train first.")
    ck = torch.load(ckpt_path, map_location="cpu")
    tok = ByteBPETokenizer.load(ck["tokenizer"])
    cfg = ModelConfig(**ck["cfg"])
    model = FableLM(cfg)
    model.load_state_dict(ck["model"])
    device = torch.device("cpu")

    results = {
        "checkpoint": str(ckpt_path),
        "step": ck["step"],
        "params": model.num_params(),
        "vocab_size": tok.vocab_size,
        "context_len": cfg.context_len,
        "loss": eval_loss(model, tok, device),
        "samples": eval_samples(model, tok),
    }
    out_path = REPO / "results" / "eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"params: {results['params']:,} | vocab {results['vocab_size']} | ctx {results['context_len']} | step {results['step']}")
    for split in ("train", "val"):
        r = results["loss"][split]
        if r:
            print(f"{split}: loss {r['loss']:.4f} | ppl {r['ppl']:.2f}")
    print("--- samples ---")
    for s in results["samples"]:
        print(f"[{s['tag']} @ t={s['temperature']}] {s['prompt']!r}\n  -> {s['response']!r}")
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
