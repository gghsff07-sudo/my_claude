"""Tests for the named document split and the dialogue upweighting (v0.2.0)."""

from fable_lm.data import is_dialogue, load_documents, split_docs, split_documents


def test_split_docs_matches_split_documents():
    """split_docs is split_documents with names — same seeded assignment."""
    docs = load_documents()
    train_docs, val_docs = split_docs(docs)
    train_texts, val_texts = split_documents(docs)
    assert [t for _, t in train_docs] == train_texts
    assert [t for _, t in val_docs] == val_texts
    names = {n for n, _ in train_docs} | {n for n, _ in val_docs}
    assert names == {n for n, _ in docs}


def test_is_dialogue():
    assert is_dialogue("01_casual.txt")
    assert is_dialogue("11_practical.txt")
    assert not is_dialogue("STUDY.md")
    assert not is_dialogue("Claude-Fable-5.1.md")
    assert not is_dialogue("README.md")


def test_dialogue_upweight_does_not_leak_val_into_train():
    """v0.1.0 bug: --dialogue-mult repeated ALL dialogue docs (train AND val
    side) into train, so val was no longer held out. The fixed recipe draws
    extra copies from the train split only."""
    docs = load_documents()
    train_docs, val_docs = split_docs(docs)

    dialogue_mult = 6
    extra = [t for n, t in train_docs if is_dialogue(n)]
    upweighted_train = [t for _, t in train_docs] + extra * (dialogue_mult - 1)

    # no val document text may appear in the upweighted train set
    leaked = {t for _, t in val_docs} & set(upweighted_train)
    assert not leaked, "val docs leaked into train after upweighting"

    # and the upweighting still does its job: dialogues are repeated
    n_train_dialogues = sum(1 for n, _ in train_docs if is_dialogue(n))
    assert len(extra) == n_train_dialogues
    assert upweighted_train.count(extra[0]) == dialogue_mult
