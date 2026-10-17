"""
Discourse-act label mapping for the Coarse Discourse corpus.

The raw corpus stores one `majority_type` per utterance (the label at least two
of the three annotators agreed on, or None when they all disagreed). This module
maps those raw strings to the toolkit's canonical label names and declares which
labels are excluded from the modeling task.

Bump LABEL_MAP_VERSION whenever the mapping or the drop list changes, so saved
datasets and reported metrics can be traced back to the label set they used.
"""

from __future__ import annotations

LABEL_MAP_VERSION = "1.0"

# Raw corpus label -> canonical label. Canonical names are snake_case, matching
# the label names documented in artifacts/corpora.py.
LABEL_MAP: dict[str, str] = {
    "question": "question",
    "answer": "answer",
    "announcement": "announcement",
    "agreement": "agreement",
    "disagreement": "disagreement",
    "appreciation": "appreciation",
    "elaboration": "elaboration",
    "humor": "humor",
    "negativereaction": "negative_reaction",
    "other": "other",
}

# 'other' is the annotators' catch-all bucket. It has no consistent meaning to
# learn and inflates/deflates macro F1 unpredictably, so it stays in the
# canonical conversations (as gold metadata) but is not a modeling target.
# This matches CoarseDiscourseLoader.DROP_LABELS.
MODELING_DROP_LABELS: frozenset[str] = frozenset({"other"})

CANONICAL_LABELS: tuple[str, ...] = tuple(
    sorted(set(LABEL_MAP.values()) - MODELING_DROP_LABELS)
)


def map_label(raw: str | None) -> str | None:
    """Map a raw `majority_type` to its canonical label.

    Returns None when there is no majority label. Raises on labels the mapping
    does not know about, so a new corpus version cannot silently introduce an
    unmapped class.
    """
    if raw is None:
        return None
    try:
        return LABEL_MAP[raw]
    except KeyError as e:
        raise ValueError(
            f"Unknown discourse label {raw!r}; add it to LABEL_MAP and bump LABEL_MAP_VERSION"
        ) from e


def annotator_agreement(annotations: list[str] | None, majority: str | None) -> float | None:
    """Fraction of annotators who chose the majority label, in [0, 1].

    Useful as a label-confidence signal: e.g. report test metrics separately on
    the subset where all three annotators agreed.
    """
    if not annotations or majority is None:
        return None
    return sum(1 for a in annotations if a == majority) / len(annotations)
