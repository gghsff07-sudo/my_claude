"""Convert the repo byte-BPE tokenizer into a PreTrainedTokenizerFast.

Usage:
    python scripts/convert_tokenizer_to_hf.py            # data/tokenizer.json -> hf_model/
    python scripts/convert_tokenizer_to_hf.py --verify-only

The repo tokenizer maps bytes to printable chars with its own bijection
(bytes 0-32 -> U+0100.., bytes 127-160 -> U+017F..), which differs from GPT-2's
byte-level mapping in the 127-160 and 173 range. Since both are bijections on
bytes, the vocab and merges are RE-EXPRESSED in GPT-2 char space (same bytes,
same merge order, same token ids) so the standard `tokenizers` ByteLevel
pre-tokenizer/decoder can be used:

    Split(regex " ?\\w+| ?[^\\w\\s]+|\\s+") -> ByteLevel(add_prefix_space=False, use_regex=False)

The regex reproduces fable_lm's stdlib-re pre-tokenizer (verified by corpus-wide
id parity below). Specials keep their ids: <|pad|>=256, <|eos|>=257, <|doc|>=258
(also placed in the BPE vocab, GPT-2-style, so vocab_size == 2048).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fable_lm.tokenizer import SPECIALS, ByteBPETokenizer  # noqa: E402


# --- GPT-2's byte -> unicode-char table (same as tokenizers' ByteLevel) ---
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


G2C = gpt2_bytes_to_unicode()

CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}\n"
    "<system>{{ messages[0]['content'] }}</system>\n"
    "{% set start = 1 %}\n"
    "{%- else %}\n"
    "{% set start = 0 %}\n"
    "{%- endif %}\n"
    "{%- for message in messages[start:] %}\n"
    "{{ message['role'] | capitalize }}: {{ message['content'] }}\n"
    "{% endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "Assistant:\n"
    "{%- endif %}"
)


def translate(token: str, repo_tok: ByteBPETokenizer) -> str:
    """Repo char space -> GPT-2 char space (per byte, bijective)."""
    return "".join(G2C[repo_tok._char_byte(c)] for c in token)


def build_fast_tokenizer(repo_tok: ByteBPETokenizer):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    # vocab: byte-chars at ids 0-255 (GPT-2 space), specials at 256-258,
    # merges at 259+ — exactly the repo's id layout.
    vocab: dict[str, int] = {G2C[b]: b for b in range(256)}
    for s in SPECIALS:
        vocab[s] = repo_tok._token_id(s)
    merges = []
    for a, b in repo_tok.merges:
        merges.append((translate(a, repo_tok), translate(b, repo_tok)))
        vocab[translate(a, repo_tok) + translate(b, repo_tok)] = repo_tok._token_id(a + b)

    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r" ?\w+| ?[^\w\s]+|\s+"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    backend.decoder = decoders.ByteLevel()

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<|pad|>",
        eos_token="<|eos|>",
        bos_token="<|doc|>",  # document separator doubles as BOS
        unk_token=None,       # byte-level: every byte has an id (GPT-2 precedent)
        model_max_length=512,
        chat_template=CHAT_TEMPLATE,
    )


def corpus_texts() -> list[str]:
    """Every text the model was trained/evaled on + adversarial edge cases."""
    texts = []
    for pat in ("data/dialogues/*.txt", "doc/*.md", "research/*.md", "*.md"):
        for p in sorted(REPO.glob(pat)):
            texts.append((p.name, p.read_text(encoding="utf-8")))
    edges = [
        "empty-adjacent", "", " ", "   ", "\n", "\t\t", "hello world",
        "   spaces   and\ttabs\n\nnewlines   ",
        "café naïve 你好世界 😀🎉 über straße",
        "cafe\u0301 decomposed accent",  # combining mark edge case
        "text with <|eos|> inside and <|pad|><|doc|> specials",
        "!!!???...—–***((()))", "3.14159 42 2026-11-19",
        "User: hello\nAssistant: hi there\n<|eos|>\nUser: again",
        "a", "z"*300, "\x00\x01\x02 control bytes \x7f\x80\xad\xff",
        "snake_case camelCase kebab-case",
    ]
    texts += [(f"edge{i}", t) for i, t in enumerate(edges)]
    return texts


def verify(repo_tok: ByteBPETokenizer, fast) -> tuple[int, int]:
    """Id parity + decode parity on the whole corpus. Returns (n_texts, n_fail)."""
    fails = 0
    total = 0
    for name, text in corpus_texts():
        total += 1
        ids_repo = repo_tok.encode(text)
        ids_fast = fast.encode(text, add_special_tokens=False)
        if ids_repo != ids_fast:
            fails += 1
            print(f"  ENCODE MISMATCH [{name}] len {len(ids_repo)} vs {len(ids_fast)}")
            for i, (r, f) in enumerate(zip(ids_repo, ids_fast)):
                if r != f:
                    print(f"    first diff at {i}: repo {r} ({repo_tok.vocab[r]!r}) "
                          f"vs fast {f}; context text around: {text[:80]!r}")
                    break
            continue
        # decode parity (specials kept, like repo decode)
        if fast.decode(ids_fast) != repo_tok.decode(ids_repo):
            fails += 1
            print(f"  DECODE MISMATCH [{name}]")
    print(f"id+decode parity: {total - fails}/{total} texts identical "
          f"({'PASS' if fails == 0 else 'FAIL'})")
    return total, fails


def verify_chat_template(fast, repo_tok) -> None:
    from fable_lm.generate import build_prompt_tokens, cue_tokens

    cases = [
        ([{"role": "system", "content": "You are concise."},
          {"role": "user", "content": "Hi!"}], True),
        ([{"role": "user", "content": "Hi!"}], True),
        ([{"role": "system", "content": "Be brief."},
          {"role": "user", "content": "Q1?"},
          {"role": "assistant", "content": "A1."},
          {"role": "user", "content": "Q2?"}], True),
        ([{"role": "user", "content": "Hi!"},
          {"role": "assistant", "content": "Hello!"}], False),
    ]
    ok = True
    for msgs, gen in cases:
        rendered = fast.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
        system = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
        rest = msgs[1:] if system else msgs
        history = [(m["role"], m["content"]) for m in rest[:-1]] if gen else \
                  [(m["role"], m["content"]) for m in rest]
        # build_prompt_tokens: system + all but the final user turn as history
        if gen:
            final_user = rest[-1]["content"]
            expected_ids = (build_prompt_tokens(repo_tok, system, history, final_user)
                            + cue_tokens(repo_tok))
        else:
            expected_ids = build_prompt_tokens(repo_tok, system, history, None)
        expected = repo_tok.decode(expected_ids)
        # rendered vs expected: compare as token ids (robust to char-mapping)
        got_ids = fast.encode(rendered, add_special_tokens=False)
        if got_ids != expected_ids:
            ok = False
            print(f"  CHAT TEMPLATE MISMATCH:\n   got:      {rendered!r}\n"
                  f"   expected: {expected!r}")
        # tokenized path must equal the text path (5.x returns a BatchEncoding
        # with input_ids/attention_mask, not a bare list)
        tok_out = fast.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)
        if hasattr(tok_out, "input_ids"):
            tok_ids = list(tok_out.input_ids)
        elif isinstance(tok_out, dict):
            tok_ids = tok_out["input_ids"]
        else:
            tok_ids = tok_out
        if tok_ids != got_ids:
            ok = False
            print(f"  CHAT TEMPLATE tokenize=True != tokenize=False for {msgs}")
    print(f"chat template == build_prompt_tokens(+cue): {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("chat template verification FAILED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(REPO / "data" / "tokenizer.json"))
    ap.add_argument("--out", default=str(REPO / "hf_model"))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    repo_tok = ByteBPETokenizer.load(args.tokenizer)
    print(f"repo tokenizer: vocab {repo_tok.vocab_size}, {len(repo_tok.merges)} merges")
    fast = build_fast_tokenizer(repo_tok)
    print(f"fast tokenizer: vocab_size={fast.vocab_size}, "
          f"specials pad/eos/bos = {fast.pad_token_id}/{fast.eos_token_id}/{fast.bos_token_id}")

    _, fails = verify(repo_tok, fast)
    if fails:
        raise SystemExit(1)
    verify_chat_template(fast, repo_tok)

    if not args.verify_only:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        fast.save_pretrained(out)
        # 5.x no longer writes special_tokens_map.json (specials live in
        # tokenizer_config.json); emit it anyway — older transformers read it,
        # and the hub-standard layout expects it.
        special_map = {"pad_token": "<|pad|>", "eos_token": "<|eos|>",
                       "bos_token": "<|doc|>", "unk_token": None}
        entries = {k: {"content": v} if v is not None else None
                   for k, v in special_map.items()}
        (out / "special_tokens_map.json").write_text(
            json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
        print(f"saved -> {out}/")
        for p in sorted(out.iterdir()):
            if p.suffix in (".json", ".jinja") or "token" in p.name:
                print(f"  {p.name:28} {p.stat().st_size:>8,} B")


if __name__ == "__main__":
    main()
