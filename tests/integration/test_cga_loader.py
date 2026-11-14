from types import SimpleNamespace

import pytest

from artifacts.corpora import CgaCmvLoader


def chain(cid, *, derailed=False, pair_id=None, count=3):
    utterances = [
        SimpleNamespace(
            id=f"{cid}_{i}",
            text=f"context {cid} {i}" if i < count - 1 else "OUTCOME REMOVED",
            timestamp=None,
            reply_to=None if i == 0 else f"{cid}_{i - 1}",
        )
        for i in range(count)
    ]
    return SimpleNamespace(
        id=cid,
        meta={"has_removed_comment": derailed, "pair_id": pair_id},
        iter_utterances=lambda: iter(reversed(utterances)),
    )


@pytest.mark.parametrize("derailed", [False, True])
def test_short_threads_never_include_final_outcome(derailed):
    corpus = SimpleNamespace(iter_conversations=lambda: iter([chain("c", derailed=derailed)]))
    examples = list(CgaCmvLoader()._iter_examples(corpus))
    assert len(examples) == 1
    assert "OUTCOME" not in examples[0].text
    assert examples[0].text == "context c 0\n\ncontext c 1"
    assert examples[0].metadata["outcome_excluded"]
    assert examples[0].label == ("derailed" if derailed else "civil")


def test_one_comment_thread_skipped_and_limit_applied_after_outcome_exclusion():
    corpus = SimpleNamespace(
        iter_conversations=lambda: iter([chain("one", count=1), chain("long", count=8)])
    )
    examples = list(CgaCmvLoader(n_utterances=2)._iter_examples(corpus))
    assert len(examples) == 1
    assert examples[0].text == "context long 0\n\ncontext long 1"


def test_matched_pair_members_stay_in_same_hash_split(monkeypatch):
    convs = [chain("a", pair_id="b"), chain("b", pair_id="a", derailed=True)]
    corpus = SimpleNamespace(iter_conversations=lambda: iter(convs))
    loader = CgaCmvLoader()
    monkeypatch.setattr(loader, "_resolve_corpus", lambda source: corpus)
    splits = loader.load(corpus)
    assert sorted(len(getattr(splits, name)) for name in ("train", "dev", "test")) == [0, 0, 2]
    assert {
        ex.metadata["pair_group_id"]
        for name in ("train", "dev", "test")
        for ex in getattr(splits, name)
    } == {"a"}


def test_malformed_chain_fails_closed():
    conv = chain("c")
    utterances = list(conv.iter_utterances())
    utterances[0].reply_to = None
    conv.iter_utterances = lambda: iter(utterances)
    corpus = SimpleNamespace(iter_conversations=lambda: iter([conv]))
    with pytest.raises(ValueError, match="linear reply chain"):
        list(CgaCmvLoader()._iter_examples(corpus))


def test_invalid_context_limit():
    with pytest.raises(ValueError):
        CgaCmvLoader(n_utterances=0)
