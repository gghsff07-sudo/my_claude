"""Chat CLI for the Fable miniature (DESIGN.md §5).

REPL:   python -m fable_lm.chat
Single: python -m fable_lm.chat --prompt "hello"

Tokens stream to the terminal as they are sampled (--no-stream to disable).
Commands: /help, /temp X, /topp X, /seed N, /system TEXT, /reset, /save FILE,
/quit. Malformed commands print a usage hint instead of crashing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from fable_lm.generate import (
    build_prompt_tokens,
    cue_tokens,
    extract_assistant_text,
    generate_stream,
)
from fable_lm.model import FableLM, ModelConfig
from fable_lm.tokenizer import SPECIAL_IDS, ByteBPETokenizer

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM = (
    "You are a concise, warm assistant. Answer in a few short sentences, "
    "decline briefly with an alternative when you can't help, and say when "
    "you don't know."
)

USAGE = {
    "temp": "/temp X      set temperature (e.g. 0.8)",
    "topp": "/topp X      set top-p (e.g. 0.9)",
    "seed": "/seed N      set the sampling seed",
    "system": "/system TEXT set the system prompt",
    "save": "/save FILE   save the conversation so far",
    "reset": "/reset      clear the conversation history",
    "quit": "/quit        leave the chat",
}

USAGE_TEXT = "commands: " + ", ".join(sorted(USAGE)) + " (/help for details)"


def _tokenizer_candidates(ck: dict, ckpt_path: Path) -> list[Path]:
    """Where to look for the tokenizer, most specific first.

    The checkpoint stores the tokenizer path as it was at training time — an
    absolute path on the training machine. When the repo (or the checkpoint)
    moves, that path no longer exists, so we fall back to the repo-relative
    location and next to the checkpoint itself.
    """
    cands: list[Path] = []
    stored = ck.get("tokenizer")
    if stored:
        p = Path(stored)
        if p.is_absolute():
            cands.append(p)
            # same relative position, resolved against this repo instead
            cands.append(REPO / "data" / p.name)
        else:
            cands.append(ckpt_path.parent / p)
            cands.append(REPO / p)
    cands.append(REPO / "data" / "tokenizer.json")
    return cands


def load_tokenizer_for(ck: dict, ckpt_path: Path) -> ByteBPETokenizer:
    """Load the tokenizer a checkpoint was trained with.

    The checkpoint stores the tokenizer path as it was at training time — an
    absolute path on the training machine. When the repo (or the checkpoint)
    moves, that path no longer exists, so we fall back to the repo-relative
    location and next to the checkpoint itself. Raises SystemExit with a
    clear message when nothing is found.
    """
    for cand in _tokenizer_candidates(ck, ckpt_path):
        if cand.is_file():
            try:
                return ByteBPETokenizer.load(cand)
            except (ValueError, KeyError, OSError):
                continue  # try the next candidate
    raise SystemExit(
        "tokenizer not found — expected data/tokenizer.json in the repo "
        f"(checkpoint points at {ck.get('tokenizer')!r})"
    )


def load_model(ckpt_path: Path | str | None = None):
    ckpt_path = Path(ckpt_path) if ckpt_path else REPO / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(
            f"No checkpoint at {ckpt_path} — train first: python -m fable_lm.train"
        )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    tok = load_tokenizer_for(ck, ckpt_path)
    cfg = ModelConfig(**ck["cfg"])
    model = FableLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ckpt_path.name}: {model.num_params():,} params, "
          f"step {ck.get('step', '?')}, val loss {ck.get('best_val', float('nan')):.4f}")
    return model, tok


def _prompt_ids(model, tok, history, user_text, system, max_new):
    prompt = build_prompt_tokens(tok, system, history, user_text)
    prompt += cue_tokens(tok)
    # crop to context (leave room for the reply)
    prompt = prompt[-(model.cfg.context_len - max_new - 1):]
    return torch.tensor([prompt], dtype=torch.long)


def reply(model, tok, history, user_text, system, temperature, top_p, max_new):
    idx = _prompt_ids(model, tok, history, user_text, system, max_new)
    new_ids: list[int] = []
    for nxt in generate_stream(
        model,
        idx,
        max_new_tokens=max_new,
        temperature=temperature,
        top_p=top_p,
        eos_ids=(SPECIAL_IDS["<|eos|>"],),
    ):
        new_ids.append(int(nxt[0, 0]))
    return extract_assistant_text(tok, new_ids)


def reply_stream(model, tok, history, user_text, system, temperature, top_p, max_new):
    """Generator yielding text deltas as tokens are sampled; the return value
    (StopIteration.value) is the final cleaned assistant text, as reply().

    Decoding is done on the full id prefix each step; a trailing partial
    multibyte character can still change once the next byte arrives, so the
    delta is computed against the longest stable common prefix.
    """
    idx = _prompt_ids(model, tok, history, user_text, system, max_new)
    new_ids: list[int] = []
    prev = ""
    for nxt in generate_stream(
        model,
        idx,
        max_new_tokens=max_new,
        temperature=temperature,
        top_p=top_p,
        eos_ids=(SPECIAL_IDS["<|eos|>"],),
    ):
        new_ids.append(int(nxt[0, 0]))
        text = extract_assistant_text(tok, new_ids)
        if not text.startswith(prev):
            # the previously shown tail was a partial character that got
            # completed: re-emit from the common prefix
            common = os.path.commonprefix([prev, text])
            text = common + text[len(common):]
            prev = common
        delta = text[len(prev):]
        if delta:
            yield delta
        prev = text
    return extract_assistant_text(tok, new_ids)


def parse_command(line: str) -> tuple[str, str] | None:
    """Parse a REPL slash command.

    Returns ("error", "usage...") for malformed commands, (name, arg) for
    valid ones, and None for ordinary chat input. Never raises.
    """
    if not line.startswith("/"):
        return None
    parts = line.split(" ", 1)
    name = parts[0][1:].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    if name in ("help", "reset", "quit"):
        return name, ""
    if name in ("temp", "topp", "seed", "system", "save"):
        if not arg:
            return "error", f"usage: {USAGE[name]}"
        if name in ("temp", "topp") and _to_float(arg) is None:
            return "error", f"usage: {USAGE[name]} (a number, e.g. 0.8)"
        if name == "seed" and _to_int(arg) is None:
            return "error", f"usage: {USAGE[name]} (an integer)"
        return name, arg
    return "error", f"unknown command {line!r} — {USAGE_TEXT}"


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


def _run_repl(model, tok, args, system, history) -> None:
    print(f"Fable miniature chat — {USAGE_TEXT} (tokens stream as they are sampled)")
    while True:
        try:
            user = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue

        cmd = parse_command(user)
        if cmd is not None:
            name, arg = cmd
            if name == "error":
                print(arg)
                continue
            if name == "quit":
                break
            if name == "help":
                for line in USAGE.values():
                    print(f"  {line}")
                continue
            if name == "reset":
                history.clear()
                print("history cleared")
                continue
            if name == "temp":
                args.temperature = _to_float(arg)
                print(f"temperature = {args.temperature}")
                continue
            if name == "topp":
                args.top_p = _to_float(arg)
                print(f"top_p = {args.top_p}")
                continue
            if name == "seed":
                torch.manual_seed(_to_int(arg))
                print(f"seed = {arg}")
                continue
            if name == "system":
                system = arg
                print(f"system prompt set ({len(system)} chars)")
                continue
            if name == "save":
                p = Path(arg).expanduser()
                if not p.is_absolute() and not p.parent.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("\n\n".join(f"{r.capitalize()}: {t}" for r, t in history))
                print(f"saved to {p}")
                continue

        print("Assistant:", end=" ", flush=True)
        if args.no_stream:
            text = reply(model, tok, history, user, system,
                         args.temperature, args.top_p, args.max_new)
            print(text)
        else:
            gen = reply_stream(model, tok, history, user, system,
                               args.temperature, args.top_p, args.max_new)
            while True:
                try:
                    delta = next(gen)
                except StopIteration as stop:
                    text = stop.value
                    break
                print(delta, end="", flush=True)
            print()
        history.append(("user", user))
        history.append(("assistant", text))


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with the Fable miniature")
    ap.add_argument("--prompt", type=str, default="", help="single-shot mode")
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--no-stream", action="store_true", help="print the reply only when finished")
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

    _run_repl(model, tok, args, system, history)


if __name__ == "__main__":
    main()
