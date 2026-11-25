"""Sampling / generation logic for the miniature (DESIGN.md §5).

Two layers:
- generate_stream(): the core loop. KV-cached by default (use_cache=True), so
  each new token costs one single-token forward instead of a full re-run of
  the context. Yields every sampled token as a (1, 1) tensor — including the
  eos token when generation stops on one.
- generate(): collects the stream into the familiar (1, T + n) tensor.

Sampling: temperature, top-p, top-k, repetition penalty. The penalty follows
the HF convention (positive logits divided, negative logits multiplied), so
fable_lm.chat and the HF wrapper's model.generate() behave the same way.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from fable_lm.model import FableLM
from fable_lm.tokenizer import SPECIAL_IDS


def _apply_repetition_penalty(
    logits: torch.Tensor, seen_ids: set[int], penalty: float
) -> torch.Tensor:
    """HF GenerationMixin semantics: positive logits are divided by the
    penalty, negative logits multiplied (dividing a negative logit would
    *raise* its probability). Modifies logits in place; returns them."""
    if seen_ids:
        idx = torch.tensor(sorted(seen_ids), dtype=torch.long, device=logits.device)
        vals = logits[0, idx]
        logits[0, idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def _sample_next(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """Pick the next token id from (1, V) logits. Returns (1, 1)."""
    if temperature <= 0:  # greedy path
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / max(1e-5, temperature)
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum - probs > top_p  # keep tokens up to and incl. first crossing
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_stream(
    model: FableLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    eos_ids: tuple[int, ...] = (),
    use_cache: bool = True,
) -> Iterator[torch.Tensor]:
    """Yield each newly sampled token as a (1, 1) int64 tensor.

    Stops at any eos_id (the eos token is yielded, then the stream ends) or
    when max_new_tokens is reached. When the context fills up, the cache is
    restarted from the last context_len tokens (same crop semantics as the
    uncached path). idx: (1, T) int64; the caller sees T + n tokens total.
    """
    model.eval()
    eos_set = set(eos_ids) or {SPECIAL_IDS["<|eos|>"]}
    ctx = model.cfg.context_len
    idx = idx[:, -ctx:] if idx.size(1) > ctx else idx  # over-long prompt: keep the tail

    # prefill: one forward over the whole prompt (or a single uncached forward)
    if use_cache:
        logits, cache = model.forward_with_cache(idx)
    else:
        logits, _ = model(idx)
        cache = None
    logits = logits[:, -1, :].float()  # (1, V)

    for _ in range(max_new_tokens):
        if repetition_penalty != 1.0:
            logits = _apply_repetition_penalty(logits, set(idx[0].tolist()), repetition_penalty)
        nxt = _sample_next(logits, temperature, top_k, top_p)
        idx = torch.cat([idx, nxt], dim=1)
        yield nxt
        if int(nxt[0, 0]) in eos_set:
            return
        # next-token logits: cached single-token step, or full recompute
        if use_cache:
            if cache.seq_len() + 1 > ctx:  # context full — restart from the tail
                logits, cache = model.forward_with_cache(idx[:, -ctx:])
            else:
                logits, cache = model.forward_with_cache(nxt, cache)
        else:
            logits, _ = model(idx[:, -ctx:])
        logits = logits[:, -1, :].float()


@torch.no_grad()
def generate(
    model: FableLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    eos_ids: tuple[int, ...] = (),
    use_cache: bool = True,
) -> torch.Tensor:
    """Autoregressive sampling with temperature, top-p, top-k and repetition
    penalty. Stops at any eos_id (which is kept in the output) or when the
    context is full. idx: (1, T) int64; returns (1, T + n)."""
    out = idx
    for nxt in generate_stream(
        model,
        idx,
        max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        eos_ids=eos_ids,
        use_cache=use_cache,
    ):
        out = torch.cat([out, nxt], dim=1)
    return out


def build_prompt_tokens(
    tokenizer,
    system: str,
    history: list[tuple[str, str]],
    new_user: str,
) -> list[int]:
    """Assemble the chat prompt: <system>...</system> + User:/Assistant: turns.

    The trailing 'Assistant:' cue is NOT added here; callers append it via
    cue_tokens() so generation starts from the cue itself.
    """
    ids: list[int] = []
    if system:
        ids.extend(tokenizer.encode(f"<system>{system}</system>\n"))
    for role, text in history:
        ids.extend(tokenizer.encode(f"{role.capitalize()}: {text}\n"))
    if new_user is not None:
        ids.extend(tokenizer.encode(f"User: {new_user}\n"))
    return ids


def cue_tokens(tokenizer) -> list[int]:
    return tokenizer.encode("Assistant:")


def extract_assistant_text(tokenizer, out_ids: list[int]) -> str:
    """Take generated ids, strip specials, and cut at a new 'User:' marker if
    the model hallucinated one."""
    text = tokenizer.decode(out_ids)
    for marker in ("<|eos|>", "<|doc|>", "<|pad|>"):
        text = text.replace(marker, "")
    if "User:" in text:
        text = text.split("User:")[0]
    return text.strip()
