"""Corpus building and dataset for the miniature (DESIGN.md §3).

Corpus = repo documents (doc/, research/, STUDY.md, DESIGN.md) + synthetic
Fable-style dialogues (data/dialogues/*.txt), joined with <|doc|> separators.
Documents are chunked to context_len; the split is at the document level,
deterministic (seed 42), 90/10 train/val.
"""

from __future__ import annotations

import random
from pathlib import Path

from fable_lm.tokenizer import SPECIAL_IDS

REPO = Path(__file__).resolve().parents[1]

CORPUS_FILES = [
    REPO / "doc" / "Claude-Fable-5.1.md",
    REPO / "research" / "README.md",
    REPO / "research" / "fable-5.1-document-analysis.md",
    REPO / "research" / "fable-5.2-delta-assumptions.md",
    REPO / "research" / "system-card.md",
    REPO / "STUDY.md",
    REPO / "DESIGN.md",
]


def load_documents() -> list[tuple[str, str]]:
    """Return [(name, text)] for every corpus file that exists."""
    docs: list[tuple[str, str]] = []
    for path in CORPUS_FILES:
        if path.is_file():
            docs.append((path.name, path.read_text(encoding="utf-8")))
    dialogues_dir = REPO / "data" / "dialogues"
    if dialogues_dir.is_dir():
        for path in sorted(dialogues_dir.glob("*.txt")):
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def split_documents(
    docs: list[tuple[str, str]], val_frac: float = 0.1, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Document-level split. Dialogues are small and numerous, so a seeded
    shuffle at the document level gives both sides a fair mix of genres."""
    names = [n for n, _ in docs]
    rng = random.Random(seed)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    n_val = max(1, int(len(docs) * val_frac)) if len(docs) > 1 else 0
    val_idx = set(idx[:n_val])
    train = [t for i, (_, t) in enumerate(docs) if i not in val_idx]
    val = [t for i, (_, t) in enumerate(docs) if i in val_idx]
    return train, val


def build_token_stream(
    texts: list[str], tokenizer, context_len: int
) -> list[list[int]]:
    """Encode each text, append <|doc|>, and chunk to context_len sequences.

    Returns a list of fixed-length token windows (all exactly context_len long;
    the final partial window is padded with <|pad|> and its pad targets are
    masked to -100 by the Dataset).
    """
    ids: list[int] = []
    for t in texts:
        ids.extend(tokenizer.encode(t))
        ids.append(SPECIAL_IDS["<|doc|>"])
    pad = SPECIAL_IDS["<|pad|>"]
    windows: list[list[int]] = []
    for start in range(0, len(ids), context_len):
        window = ids[start : start + context_len]
        if len(window) < context_len:
            window = window + [pad] * (context_len - len(window))
        windows.append(window)
    return windows
