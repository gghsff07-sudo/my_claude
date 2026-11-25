"""Gradio chat demo for fable-mini (HF Space / local).

Loads the model through plain transformers — from the Hub repo id or a local
path — using the chat template, and streams tokens with TextIteratorStreamer.

Usage:
    python app.py                                  # Hub model (dev-sandakelum/fable-mini)
    python app.py --model hf_model                 # local converted folder
    MODEL=dev-sandakelum/fable-mini python app.py  # env override

--model is parsed before the model loads (the original parsed it after, so
the flag never reached the loader); unknown extra arguments, e.g. the ones a
Space injects, are ignored.
"""

from __future__ import annotations

import argparse
import os
import threading

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from fable_lm.demo_utils import clean, messages_from_history

TRUST_REMOTE_CODE = True  # custom architecture (configuration_fable/modeling_fable)

# early, tolerant parse: only what the load needs; everything else (e.g. -h)
# falls through to the full parser in __main__
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--model", default=None)
_early_args, _ = _early.parse_known_args()

DEFAULT_MODEL = _early_args.model or os.environ.get("MODEL", "dev-sandakelum/fable-mini")

print(f"loading {DEFAULT_MODEL} ...")
tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
model.eval()
print(f"loaded: {type(model).__name__}, {model.num_params():,} params")

SYSTEM_DEFAULT = (
    "You are a concise, warm assistant. Answer in a few short sentences, "
    "decline briefly with an alternative when you can't help, and say when "
    "you don't know."
)


def respond(message, history, system, temperature, top_p, max_new_tokens):
    # history: gradio 6 messages format (dicts); the pending user turn is
    # `message` and is appended once — multi-turn context now survives
    messages = messages_from_history(system, history, message)

    enc = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    streamer = TextIteratorStreamer(tok, skip_prompt=True, timeout=60.0)
    gen_kwargs = dict(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=float(temperature) > 0,
        temperature=max(float(temperature), 1e-4),
        top_p=float(top_p),
        repetition_penalty=1.2,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for chunk in streamer:
        partial = clean(partial + chunk)
        yield partial


DEMO_TITLE = "fable-mini — a 3.3M-param educational miniature"
DEMO_DESCRIPTION = (
    "A decoder-only transformer (4L · d256 · 4h · ctx512) trained from scratch "
    "on a tiny in-repo corpus to study the full LLM pipeline. **It does not "
    "produce coherent answers** — expect register-imitating word soup; that is "
    "the honest, documented behavior at this scale ([model card]"
    "(https://huggingface.co/dev-sandakelum/fable-mini))."
)

with gr.Blocks(title=DEMO_TITLE) as demo:
    gr.Markdown(f"# {DEMO_TITLE}\n\n{DEMO_DESCRIPTION}")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=420)  # gradio 6: messages format is the only format
            msg = gr.Textbox(placeholder="Say something...", label="Message")
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=1):
            system = gr.Textbox(value=SYSTEM_DEFAULT, label="System prompt")
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            max_new = gr.Slider(16, 200, value=80, step=8, label="Max new tokens")
            gr.Markdown(
                "*Educational demo. Output is incoherent by design; see the "
                "model card for honest eval numbers.*"
            )

    def user_turn(message, chat):
        return "", chat + [{"role": "user", "content": message}]

    def bot_turn(chat, system, temperature, top_p, max_new_tokens):
        # gradio sends the full log; entries alternate user/assistant, the
        # last entry is the pending user turn
        history = [dict(m) for m in chat[:-1]]
        reply = respond(chat[-1]["content"], history, system, temperature, top_p, max_new_tokens)
        chat = chat + [{"role": "assistant", "content": ""}]
        for chunk in reply:
            chat[-1]["content"] = chunk
            yield chat

    msg.submit(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot, system, temperature, top_p, max_new], chatbot
    )
    send.click(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot, system, temperature, top_p, max_new], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Hub repo id or local path (default: %(default)s)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    args = parser.parse_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port)
