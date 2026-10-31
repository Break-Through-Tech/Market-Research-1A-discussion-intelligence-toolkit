"""Quality-scoring dataset, scorers and metrics (issue #16). No downloads needed."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from artifacts.corpora import ClassificationSplits
from utils.quality.cga_cmv import (
    check_pair_disjoint,
    context_comments,
    load_quality_dataset,
    reply_chain,
    run_quality_leakage_checks,
)
from utils.quality.embeddings import SentenceTransformerEmbedder, text_key
from utils.quality.metrics import bootstrap_ci, quality_metrics
from utils.quality.scorers import ConstantScorer, EmbeddingScorer, LengthScorer, TfidfScorer

CIVIL_WORDS = "thanks good point fair evidence source agree consider"
RUDE_WORDS = "idiot stupid ridiculous moron pathetic clueless nonsense"


def utt(id, conv, reply_to, text, ts):
    return {"id": id, "conversation_id": conv, "reply-to": reply_to, "speaker": "s",
            "text": text, "timestamp": ts, "meta": {"score": 1, "parsed": [{"toks": []}]}}


def write_corpus(tmp_path, n_pairs=4, splits=("train", "train", "val", "test")):
    """Pairs of conversations; derailed ones end with the removed comment."""
    conversations, lines = {}, []
    for p in range(n_pairs):
        split = splits[p % len(splits)]
        for derailed in (False, True):
            cid = f"c{p}{'d' if derailed else 'k'}"
            twin = f"c{p}{'k' if derailed else 'd'}"
            conversations[cid] = {"meta": {"pair_id": twin, "has_removed_comment": derailed,
                                           "split": split, "summary_meta": []}, "vectors": []}
            words = RUDE_WORDS if derailed else CIVIL_WORDS
            texts = [f"opening {p}", f"{words} reply", "[deleted]", f"{words} again",
                     "FINAL removed comment" if derailed else "FINAL civil comment"]
            prev = None
            for i, t in enumerate(texts):
                uid = cid if i == 0 else f"{cid}_{i}"
                lines.append(utt(uid, cid, prev, t, 100 + i))
                prev = uid
    (tmp_path / "conversations.json").write_text(json.dumps(conversations))
    # Shuffle on disk to check that order comes from the reply chain.
    (tmp_path / "utterances.jsonl").write_text("\n".join(json.dumps(u) for u in reversed(lines)))
    return tmp_path


def test_final_comment_is_never_in_context(tmp_path):
    data = load_quality_dataset(write_corpus(tmp_path))
    for split in ("train", "dev", "test"):
        for ex in getattr(data.splits, split):
            assert not any("FINAL" in c for c in ex.metadata["comments"])
            assert "FINAL" not in ex.text


def test_context_is_ordered_and_filtered(tmp_path):
    data = load_quality_dataset(write_corpus(tmp_path))
    ex = next(e for e in data.splits.train if e.id == "c0k")
    assert ex.metadata["comments"] == ["opening 0", f"{CIVIL_WORDS} reply", f"{CIVIL_WORDS} again"]
    assert data.exclusions["deleted_text"] == 8
    assert data.exclusions["final_comment_dropped"] == 8


def test_official_split_is_kept_and_val_becomes_dev(tmp_path):
    data = load_quality_dataset(write_corpus(tmp_path))
    assert {e.id for e in data.splits.dev} == {"c2k", "c2d"}
    assert {e.id for e in data.splits.test} == {"c3k", "c3d"}
    assert data.civil("train").count(1) == data.civil("train").count(0) == 2


def test_max_comments_truncates_from_the_start(tmp_path):
    data = load_quality_dataset(write_corpus(tmp_path), max_comments=1)
    assert all(e.metadata["comments"] == [f"opening {e.id[1]}"] for e in data.splits.train)


def test_moderator_notice_is_dropped():
    chain = [{"id": "a", "text": "real point"},
             {"id": "b", "text": "Sorry u/x, your comment has been removed: Comment Rule 2"},
             {"id": "c", "text": "last"}]
    ex = Counter()
    assert context_comments(chain, ex) == ["real point"]
    assert ex["moderator_notice"] == 1


def test_reply_chain_ignores_file_order_and_bad_timestamps():
    utts = [utt("c", "r", "b", "3", 1), utt("r", "r", None, "1", 3), utt("b", "r", "r", "2", 2)]
    assert [u["id"] for u in reply_chain(utts)] == ["r", "b", "c"]


def test_conversation_with_no_context_is_skipped(tmp_path):
    conversations = {"x": {"meta": {"pair_id": "y", "has_removed_comment": True, "split": "train"}}}
    lines = [utt("x", "x", None, "[deleted]", 1), utt("x1", "x", "x", "removed one", 2)]
    (tmp_path / "conversations.json").write_text(json.dumps(conversations))
    (tmp_path / "utterances.jsonl").write_text("\n".join(json.dumps(u) for u in lines))
    data = load_quality_dataset(tmp_path)
    assert len(data.splits.train) == 0
    assert data.exclusions["conversation_without_context"] == 1


def test_leakage_checks_pass_and_catch_a_split_pair(tmp_path):
    data = load_quality_dataset(write_corpus(tmp_path))
    assert run_quality_leakage_checks(data.splits).ok
    moved = data.splits.dev[0]
    broken = ClassificationSplits(train=data.splits.train + [moved],
                                  dev=data.splits.dev[1:], test=data.splits.test)
    result = check_pair_disjoint(broken)
    assert result.status == "fail"


def test_pair_groups_match_for_both_members(tmp_path):
    data = load_quality_dataset(write_corpus(tmp_path))
    groups = dict(zip((e.id for e in data.splits.train), data.groups("train")))
    assert groups["c0k"] == groups["c0d"]
    assert groups["c0k"] != groups["c1k"]


# ---------------------------------------------------------------- scorers


def toy_conversations(n=40, seed=0):
    rng = np.random.default_rng(seed)
    convs, civil = [], []
    for i in range(n):
        is_civil = i % 2
        pool = (CIVIL_WORDS if is_civil else RUDE_WORDS).split()
        convs.append([" ".join(rng.choice(pool, 5)) for _ in range(1 + i % 3)])
        civil.append(is_civil)
    return convs, civil


class FakeEmbedder:
    """Deterministic 2-d vectors: [rude-word share, civil-word share]."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        rows = []
        for t in texts:
            words = t.split()
            rows.append([sum(w in RUDE_WORDS for w in words) / len(words),
                         sum(w in CIVIL_WORDS for w in words) / len(words)])
        return np.asarray(rows)


@pytest.mark.parametrize("make", [lambda: TfidfScorer(C=10), lambda: EmbeddingScorer(FakeEmbedder(), C=10)])
def test_text_scorers_separate_toy_classes(make):
    convs, civil = toy_conversations()
    scores = make().fit(convs, civil).predict(convs)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert quality_metrics(civil, scores).roc_auc == 1.0


def test_constant_scorer_has_no_ranking_signal():
    convs, civil = toy_conversations()
    scores = ConstantScorer().fit(convs, civil).predict(convs)
    assert set(scores) == {0.5}
    assert quality_metrics(civil, scores).roc_auc == 0.5


def test_length_scorer_handles_single_comment_threads():
    scores = LengthScorer().fit([["a"], ["a b c", "d"]], [0, 1]).predict([["x y"]])
    assert len(scores) == 1


def test_embedding_scorer_embeds_each_comment_once_per_call():
    emb = FakeEmbedder()
    convs, civil = toy_conversations(10)
    x = EmbeddingScorer(emb).features(convs)
    assert x.shape == (10, 4)  # mean + last comment
    assert emb.calls == 1


# ---------------------------------------------------------------- metrics and cache


def test_quality_metrics_on_known_values():
    r = quality_metrics([1, 0, 1, 0], [0.9, 0.1, 0.6, 0.4])
    assert r.roc_auc == 1.0
    assert r.accuracy == 1.0
    assert r.mae == pytest.approx(0.25)
    assert r.brier == pytest.approx((0.01 + 0.01 + 0.16 + 0.16) / 4)


def test_bootstrap_resamples_whole_pairs():
    civil = [1, 0] * 20
    scores = [0.9, 0.1] * 20
    groups = [f"p{i // 2}" for i in range(40)]
    assert bootstrap_ci(civil, scores, groups, n_resamples=50) == (1.0, 1.0)


def test_embedder_cache_round_trip(tmp_path):
    emb = SentenceTransformerEmbedder("fake/model", cache_dir=tmp_path)
    emb._model = type("M", (), {"encode": lambda self, texts, **kw: np.ones((len(texts), 3))})()
    first = emb.embed(["a", "b", "a"])
    assert first.shape == (3, 3)

    reloaded = SentenceTransformerEmbedder("fake/model", cache_dir=tmp_path)
    assert set(reloaded._cache) == {text_key("a"), text_key("b")}
    np.testing.assert_array_equal(reloaded.embed(["b"]), first[1:2])  # no model needed
