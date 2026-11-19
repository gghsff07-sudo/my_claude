"""Chat CLI for the Fable miniature (DESIGN.md §5).

REPL:   python -m fable_lm.chat
Single: python -m fable_lm.chat --prompt "hello"

Sampling controls: /temp X, /topp X, /system TEXT, /reset, /quit, /save FILE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fable_lm.generate import build_prompt_tokens, cue_tokens, extract_assistant_text, generate
from fable_lm.model import FableLM, ModelConfig
from fable_lm.tokenizer import SPECIAL_IDS, ByteBPETokenizer

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM = (
    "You are a concise, warm assistant. Answer in a few short sentences, "
    "decline briefly with an alternative when you can't help, and say when "
    "you don't know."
)


def load_model(ckpt_path: Path | str | None = None):
    ckpt_path = Path(ckpt_path) if ckpt_path else REPO / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(
            f"No checkpoint at {ckpt_path} — train first: python -m fable_lm.train"
        )
    ck = torch.load(ckpt_path, map_location="cpu")
    tok_path = ck.get("tokenizer")
    tok = ByteBPETokenizer.load(tok_path) if tok_path else None
    if tok is None:
        tok = ByteBPETokenizer.load(REPO / "data" / "tokenizer.json")
    cfg = ModelConfig(**ck["cfg"])
    model = FableLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ckpt_path.name}: {model.num_params():,} params, "
          f"step {ck.get('step', '?')}, val loss {ck.get('best_val', float('nan')):.4f}")
    return model, tok


def reply(model, tok, history, user_text, system, temperature, top_p, max_new):
    prompt = build_prompt_tokens(tok, system, history, user_text)
    prompt += cue_tokens(tok)
    # crop to context
    prompt = prompt[-(model.cfg.context_len - max_new - 1) :]
    idx = torch.tensor([prompt], dtype=torch.long)
    out = generate(
        model,
        idx,
        max_new_tokens=max_new,
        temperature=temperature,
        top_p=top_p,
        eos_ids=(SPECIAL_IDS["<|eos|>"],),
    )
    new_ids = out[0, len(prompt) :].tolist()
    return extract_assistant_text(tok, new_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with the Fable miniature")
    ap.add_argument("--prompt", type=str, default="", help="single-shot mode")
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--checkpoint", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(42)
    model, tok = load_model(args.checkpoint or None)
    history: list[tuple[str, str]] = []
    system = args.system

    if args.prompt:
        print(reply(model, tok, history, args.prompt, system,
                    args.temperature, args.top_p, args.max_new))
        return

    print("Fable miniature chat — /temp X, /topp X, /system TEXT, /reset, /save FILE, /quit")
    while True:
        try:
            user = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user.startswith("/temp "):
            args.temperature = float(user.split()[1]); print(f"temperature = {args.temperature}")
            continue
        if user.startswith("/topp "):
            args.top_p = float(user.split()[1]); print(f"top_p = {args.top_p}")
            continue
        if user.startswith("/system "):
            system = user[len("/system "):]; print(f"system prompt set ({len(system)} chars)")
            continue
        if user == "/reset":
            history.clear(); print("history cleared")
            continue
        if user.startswith("/save "):
            p = Path(user.split(" ", 1)[1])
            p.write_text("\n\n".join(f"{r.capitalize()}: {t}" for r, t in history))
            print(f"saved to {p}")
            continue

        print("Assistant:", end=" ", flush=True)
        text = reply(model, tok, history, user, system,
                     args.temperature, args.top_p, args.max_new)
        print(text)
        history.append(("user", user))
        history.append(("assistant", text))


if __name__ == "__main__":
    main()
