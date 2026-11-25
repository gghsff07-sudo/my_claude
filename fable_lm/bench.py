"""Reproducible generation-speed benchmark for the fable_lm library.

Measures the wall-clock time of the chat reply path — greedy, sampled, and
multi-turn — with the shipped checkpoint, 3 trials each, median reported.

Usage:
    python -m fable_lm.bench [checkpoint]
    fable-bench [checkpoint]        (installed console entry point)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def bench(fn, n: int = 3):
    times, out = [], None
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2], out


def main() -> None:
    from fable_lm.chat import DEFAULT_SYSTEM, load_model, reply

    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "models" / "fable-mini-final.pt"
    torch.set_num_threads(2)
    model, tok = load_model(ckpt)

    def greedy_100():
        return reply(model, tok, [], "Hello, who are you?", DEFAULT_SYSTEM,
                     temperature=0.0, top_p=1.0, max_new=100)

    def sample_100():
        return reply(model, tok, [], "what should I make for dinner tonight?",
                     DEFAULT_SYSTEM, temperature=0.8, top_p=0.9, max_new=100)

    def multi_turn_60():
        history = [("user", "hi"), ("assistant", "Hello! How can I help?")]
        return reply(model, tok, history, "tell me more", DEFAULT_SYSTEM,
                     temperature=0.8, top_p=0.9, max_new=60)

    for label, fn in (("greedy 100 tok", greedy_100),
                      ("sample 100 tok", sample_100),
                      ("multi-turn 60 tok", multi_turn_60)):
        med, out = bench(fn)
        toks = len(tok.encode(out))
        print(f"{label:20s} median {med:6.2f}s  ({toks / med:6.1f} tok/s out)")


if __name__ == "__main__":
    main()
