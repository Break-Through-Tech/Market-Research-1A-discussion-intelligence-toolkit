from collections import Counter

from conftest import raw_utt
from utils.modeling_data.convokit_reader import _raw_from_json
from utils.modeling_data.examples import iter_examples
from utils.modeling_data.normalize import (
    MISSING_TIMESTAMP,
    NormalizationStats,
    normalize_conversation,
)

ROOT = "t3_abc"


def _thread():
    rows = [
        raw_utt(ROOT, ROOT, None, "Which GPU should I buy?", "question", "op", 0),
        raw_utt("t1_a", ROOT, ROOT, "Get the 4070.", "answer", "alice"),
        raw_utt("t1_b", ROOT, "t1_a", "Agreed!", "agreement", "bob", 2),
        raw_utt("t1_c", ROOT, ROOT, "[removed]", "answer", "[deleted]"),
        raw_utt("t1_d", ROOT, "t1_gone", "orphaned reply", "negativereaction", "carol", 3),
        raw_utt("t1_e", ROOT, ROOT, None, None, "dave", annotations=["humor", "answer", "other"]),
    ]
    return [_raw_from_json(r) for r in rows]


def _normalize(**kw):
    stats = NormalizationStats()
    conv = normalize_conversation(ROOT, _thread(), {"subreddit": "buildapc", "title": "GPU?"},
                                  stats=stats, **kw)
    return conv, stats


def test_tree_invariants_hold():
    conv, _ = _normalize()
    assert conv.id == "reddit:t3_abc"
    assert conv.root().id == "reddit:t3_abc"
    ids = {u.id for u in conv.utterances}
    assert all(u.parent_id in ids for u in conv.utterances if u.parent_id is not None)
    assert len(list(conv.traverse())) == len(conv)


def test_orphan_is_reparented_to_root_with_provenance():
    conv, stats = _normalize()
    orphan = next(u for u in conv.utterances if u.source_id == "t1_d")
    assert orphan.parent_id == "reddit:t3_abc"
    assert orphan.metadata["source_parent_id"] == "t1_gone"
    assert orphan.metadata["orphan_reparented"] is True
    assert orphan.depth == 3
    assert stats.orphans_reparented == 1


def test_missing_timestamps_use_flagged_sentinel():
    conv, stats = _normalize()
    assert conv.created_at == MISSING_TIMESTAMP
    assert all(u.metadata["timestamp_missing"] for u in conv.utterances)
    assert stats.missing_timestamps == len(conv)


def test_gold_label_stays_out_of_prediction_slot():
    conv, _ = _normalize()
    assert all(u.discourse_act is None for u in conv.utterances)
    gold = {u.source_id: u.metadata["discourse_act_gold"] for u in conv.utterances}
    assert gold["t1_d"] == "negative_reaction"
    assert gold["t1_e"] is None


def test_deleted_text_and_author_flags():
    conv, stats = _normalize()
    removed = next(u for u in conv.utterances if u.source_id == "t1_c")
    assert removed.is_deleted
    assert conv.speaker_of(removed).is_deleted
    assert stats.deleted_text == 1 and stats.empty_text == 1 and stats.deleted_authors == 1


def test_usernames_are_pseudonymized_by_default():
    conv, _ = _normalize()
    dumped = conv.model_dump_json()
    for name in ("alice", "bob", "carol", "dave"):
        assert f'"{name}"' not in dumped
    alice_ids = {u.speaker_id for u in conv.utterances if u.source_id == "t1_a"}
    assert all(s.startswith("reddit:u/anon_") for s in alice_ids)


def test_usernames_can_be_kept():
    conv, _ = _normalize(pseudonymize_speakers=False)
    assert "reddit:u/alice" in conv.speakers


def test_thread_without_root_is_skipped():
    stats = NormalizationStats()
    rows = [_raw_from_json(raw_utt("t1_x", ROOT, "t1_y", "hi"))]
    assert normalize_conversation(ROOT, rows, stats=stats) is None
    assert stats.skipped_conversations_without_root == 1


def test_examples_apply_exclusion_rules_and_carry_context():
    conv, _ = _normalize()
    reasons: Counter = Counter()
    examples = {ex.id: ex for ex in iter_examples(conv, reasons)}
    assert set(examples) == {"reddit:t3_abc", "reddit:t1_a", "reddit:t1_b", "reddit:t1_d"}
    assert reasons == Counter({"deleted_text": 1, "no_majority_label": 1})
    reply = examples["reddit:t1_b"]
    assert reply.label == "agreement"
    assert reply.metadata["parent_text"] == "Get the 4070."
    assert reply.metadata["conversation_source_id"] == ROOT
    assert examples["reddit:t3_abc"].metadata["is_root"] is True
