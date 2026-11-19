"""Sampling / generation logic for the miniature (DESIGN.md §5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fable_lm.model import FableLM
from fable_lm.tokenizer import SPECIAL_IDS


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
) -> torch.Tensor:
    """Autoregressive sampling with temperature, top-p, top-k and repetition
    penalty. Stops at any eos_id (which is kept in the output) or when the
    context is full. idx: (1, T) int64; returns (1, T + n)."""
    model.eval()
    device = idx.device
    eos_set = set(eos_ids) or {SPECIAL_IDS["<|eos|>"]}
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.context_len :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :].float()  # (1, V)

        if repetition_penalty != 1.0:
            for tok in set(idx[0].tolist()):
                logits[0, tok] /= repetition_penalty

        if temperature <= 0:  # greedy path
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
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
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, sorted_idx, sorted_logits
                )
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, nxt], dim=1)
        if int(nxt[0, 0]) in eos_set:
            break
    return idx


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
