"""Pure helpers for the Gradio demo (app.py) — no gradio import needed.

Kept import-light and side-effect-free so they can be unit-tested directly:
history→messages conversion (gradio 6 sends the chat log as dicts) and the
output cleaning shared with fable_lm.chat.
"""

from __future__ import annotations

Message = dict  # {"role": str, "content": str}


def messages_from_history(system: str, history, message: str) -> list[Message]:
    """Assemble the transformers-style message list for apply_chat_template.

    history entries may be {"role": ..., "content": ...} dicts (gradio 6
    messages format) or (role, content) tuples; the pending user turn is
    `message` and must NOT also appear in history.
    """
    msgs: list[Message] = []
    if system:
        msgs.append({"role": "system", "content": system})
    for m in history:
        role, content = (m["role"], m["content"]) if isinstance(m, dict) else (m[0], m[1])
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": message})
    return msgs


def clean(text: str) -> str:
    """Strip specials and any hallucinated next turn, as in fable_lm.chat."""
    for marker in ("<|eos|>", "<|doc|>", "<|pad|>"):
        text = text.replace(marker, "")
    return text.split("User:")[0].strip()
