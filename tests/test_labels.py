import pytest

from utils.modeling_data.labels import (
    CANONICAL_LABELS,
    LABEL_MAP,
    MODELING_DROP_LABELS,
    annotator_agreement,
    map_label,
)


def test_raw_labels_map_to_snake_case():
    assert map_label("negativereaction") == "negative_reaction"
    assert map_label("question") == "question"


def test_missing_majority_maps_to_none():
    assert map_label(None) is None


def test_unknown_label_fails_loudly():
    with pytest.raises(ValueError, match="LABEL_MAP"):
        map_label("sarcasm")


def test_modeling_labels_exclude_dropped():
    assert "other" in MODELING_DROP_LABELS
    assert "other" not in CANONICAL_LABELS
    assert set(CANONICAL_LABELS) == set(LABEL_MAP.values()) - MODELING_DROP_LABELS
    assert len(CANONICAL_LABELS) == 9


def test_annotator_agreement():
    assert annotator_agreement(["answer", "answer", "question"], "answer") == pytest.approx(2 / 3)
    assert annotator_agreement(["answer"] * 3, "answer") == 1.0
    assert annotator_agreement([], "answer") is None
    assert annotator_agreement(["a", "b", "c"], None) is None
