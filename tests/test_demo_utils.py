"""Tests for the pure Gradio-demo helpers (fable_lm.demo_utils).

Regression tests for the v0.1.0 app.py bug: history dicts were unpacked as
(user, assistant) PAIRS, garbling the message list sent to the chat template.
"""

from fable_lm.demo_utils import clean, messages_from_history


def test_messages_from_history_dicts():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "tell me more"},
        {"role": "assistant", "content": "Sure."},
    ]
    msgs = messages_from_history("be brief", history, "and now?")
    assert msgs == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "tell me more"},
        {"role": "assistant", "content": "Sure."},
        {"role": "user", "content": "and now?"},
    ]


def test_messages_from_history_tuples():
    history = [("user", "hi"), ("assistant", "hello")]
    msgs = messages_from_history("", history, "next?")
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "next?"},
    ]


def test_pending_turn_not_duplicated():
    """The v0.1.0 bug produced {'role': 'user', 'content': 'user'} entries."""
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "hello"}]
    msgs = messages_from_history("sys", history, "question")
    assert not any(m["content"] in ("user", "assistant") for m in msgs)
    assert sum(1 for m in msgs if m["role"] == "user") == 2  # history + pending


def test_clean_strips_specials_and_hallucinated_turn():
    assert clean("Hi there.<|eos|>") == "Hi there."
    assert clean("Sure.<|doc|> bye") == "Sure. bye"
    assert clean("Answer.\nUser: next question") == "Answer."
    assert clean("  padded  ") == "padded"
