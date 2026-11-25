"""Byte-level BPE tokenizer, implemented from scratch.

Design (DESIGN.md §1, assumption A6):
- Bytes 0-255 keep token ids 0-255; special tokens follow; learned merges after
  that. Byte-level means encode/decode is lossless on ANY text.
- Pre-tokenization splits text into words so merges never cross word boundaries
  (GPT-2 style, but with a stdlib-`re` pattern — no `regex` dependency).
- Specials: <|pad|>, <|eos|>, <|doc|>. They are recognized in input text and
  always encode to their single ids, and are excluded from merge learning.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# stdlib re: \w covers [A-Za-z0-9_] plus unicode letters/digits — enough here,
# and deterministic. Leading-space capture keeps " word" merges natural.
_PRETOK = re.compile(r" ?\w+| ?[^\w\s]+|\s+")
_SPECIAL_SPLIT = re.compile(
    "(" + "|".join(re.escape(s) for s in ["<|pad|>", "<|eos|>", "<|doc|>"]) + ")"
)

SPECIALS = ["<|pad|>", "<|eos|>", "<|doc|>"]
BYTE_IDS = 256  # ids 0..255 are the raw bytes
SPECIAL_IDS = {tok: BYTE_IDS + i for i, tok in enumerate(SPECIALS)}


def _pretokenize(text: str) -> list[str]:
    return _PRETOK.findall(text)


class ByteBPETokenizer:
    def __init__(self, merges: list[tuple[str, str]] | None = None):
        self.merges = list(merges or [])
        # id -> token string (a token is a string of byte-chars, or a special)
        self.vocab: list[str] = [self._byte_char(i) for i in range(BYTE_IDS)]
        self.vocab += SPECIALS
        for a, b in self.merges:
            self.vocab.append(a + b)
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._id_cache: dict[str, int] = {}
        self._cache_len = -1

    # ---------- byte <-> printable char helpers (GPT-2 style) ----------
    # bijection: bytes 33-126 -> chars 33-126, bytes 161-255 -> chars 161-255,
    # bytes 0-32 -> chars 256-288, bytes 127-160 -> chars 383-416

    @staticmethod
    def _byte_char(b: int) -> str:
        if 33 <= b <= 126 or b >= 161:
            return chr(b)
        return chr(b + 256)

    @staticmethod
    def _char_byte(c: str) -> int:
        o = ord(c)
        if 33 <= o <= 126 or 161 <= o <= 255:
            return o
        return o - 256

    # ---------- training ----------

    def train(self, texts: list[str], vocab_size: int, verbose: bool = False) -> int:
        """Learn merges until vocab reaches vocab_size. Returns number of merges.

        Incremental BPE: pair counts are maintained across merges and only the
        words containing the chosen pair are re-scanned, so cost is proportional
        to affected words, not the whole vocabulary, per merge.
        """
        if vocab_size < BYTE_IDS + len(SPECIALS) + 1:
            raise ValueError("vocab_size too small for bytes+specials+1 merge")

        # word (tuple of byte-chars) -> frequency; specials never enter BPE
        words: dict[tuple[str, ...], int] = defaultdict(int)
        for text in texts:
            for chunk in _SPECIAL_SPLIT.split(text):
                if not chunk or chunk in SPECIAL_IDS:
                    continue
                for w in _pretokenize(chunk):
                    key = tuple(self._byte_char(b) for b in w.encode("utf-8"))
                    words[key] += 1

        # pair stats + reverse index of which words contain each pair
        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_words: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
        for w, f in words.items():
            for i in range(len(w) - 1):
                p = (w[i], w[i + 1])
                pair_counts[p] += f
                pair_words[p].add(w)

        target_merges = vocab_size - BYTE_IDS - len(SPECIALS)
        merges: list[tuple[str, str]] = []
        while len(merges) < target_merges:
            # best = most frequent pair; deterministic tie-break on the pair
            # itself (greatest pair wins, as before, but via one C-level max)
            if not pair_counts:
                break
            best, best_cnt = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if best_cnt < 2:
                break  # no pair occurs twice — no signal left
            merges.append(best)

            # apply the merge to every word containing it, updating stats
            for w in list(pair_words[best]):
                f = words.pop(w)
                new_w = self._merge_word(w, best)
                for i in range(len(w) - 1):
                    p = (w[i], w[i + 1])
                    pair_counts[p] -= f
                    pair_words[p].discard(w)
                    if pair_counts[p] <= 0:
                        del pair_counts[p]  # keep dict tidy; 0-entries are noise
                if len(new_w) > 1:
                    for i in range(len(new_w) - 1):
                        p = (new_w[i], new_w[i + 1])
                        pair_counts[p] += f
                        pair_words[p].add(new_w)
                words[new_w] = words.get(new_w, 0) + f
            if verbose and len(merges) % 200 == 0:
                print(f"  merge {len(merges)}/{target_merges}: {best!r} (freq {best_cnt})")

        self.__init__(merges)  # recompute vocab/ranks/cache
        return len(merges)

    @staticmethod
    def _merge_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    # ---------- encode / decode ----------

    def _token_id(self, tok: str) -> int:
        if self._cache_len != len(self.vocab):
            self._id_cache = {t: i for i, t in enumerate(self.vocab)}
            self._cache_len = len(self.vocab)
        return self._id_cache[tok]

    def _encode_word(self, word: str) -> list[int]:
        parts = [self._byte_char(b) for b in word.encode("utf-8")]
        while len(parts) > 1:
            # find the lowest-rank (earliest learned) adjacent pair
            best_i, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self._ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_i, best_rank = i, r
            if best_i is None:
                break
            parts[best_i: best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        ids = []
        for p in parts:
            try:
                ids.append(self._token_id(p))
            except KeyError:
                # byte-level tokens are all in vocab; this is unreachable safety
                for ch in p:
                    ids.append(self._token_id(ch))
        return ids

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        # split out specials first so their ids are exact and unmergeable
        for chunk in _SPECIAL_SPLIT.split(text):
            if not chunk:
                continue
            if chunk in SPECIAL_IDS:
                ids.append(SPECIAL_IDS[chunk])
            else:
                for w in _pretokenize(chunk):
                    ids.extend(self._encode_word(w))
        if add_eos:
            ids.append(SPECIAL_IDS["<|eos|>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        buf: list[int] = []  # raw bytes of ordinary tokens, decoded as UTF-8

        def flush() -> None:
            if buf:
                out.append(bytes(buf).decode("utf-8", errors="replace"))
                buf.clear()

        for i in ids:
            tok = self.vocab[i]
            if tok in SPECIAL_IDS:
                flush()
                out.append(tok)
            else:
                buf.extend(self._char_byte(c) for c in tok)
        flush()
        return "".join(out)

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "type": "byte_bpe",
                    "specials": SPECIALS,
                    "merges": [list(p) for p in self.merges],
                },
                ensure_ascii=False,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        d = json.loads(Path(path).read_text())
        assert d["type"] == "byte_bpe" and d["specials"] == SPECIALS
        return cls(merges=[tuple(p) for p in d["merges"]])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
