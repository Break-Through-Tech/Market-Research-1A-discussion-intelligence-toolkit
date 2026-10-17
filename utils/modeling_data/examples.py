"""
Turn canonical Conversations into discourse-act ClassificationExamples.

One example per labeled utterance. Exclusion rules, applied in this order and
counted by reason so the dataset card can account for every utterance:

    no_majority_label   annotators did not agree on a label
    dropped_label       label is in MODELING_DROP_LABELS ('other')
    deleted_text        text is [deleted] / [removed]
    empty_text          text is empty or whitespace

Examples use the same `ClassificationExample` type as artifacts/corpora.py, so
the written JSONL files load with `load_jsonl_classification` and plug straight
into `evaluate_classifier`.

Each example carries thread context in metadata (parent text, depth, thread
title, subreddit) because a discourse act is defined relative to what it
replies to; context-aware models need it, bag-of-words baselines can ignore it.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterator

from artifacts.corpora import ClassificationExample
from artifacts.schema import Conversation
from utils.modeling_data.labels import MODELING_DROP_LABELS


def iter_examples(
    conversation: Conversation, exclusions: Counter[str] | None = None
) -> Iterator[ClassificationExample]:
    exclusions = exclusions if exclusions is not None else Counter()
    by_id = {u.id: u for u in conversation.utterances}
    subreddit = conversation.metadata.get("subreddit")

    for utt in conversation.utterances:
        label = utt.metadata.get("discourse_act_gold")
        text = utt.text.strip()
        if label is None:
            exclusions["no_majority_label"] += 1
            continue
        if label in MODELING_DROP_LABELS:
            exclusions["dropped_label"] += 1
            continue
        if utt.is_deleted:
            exclusions["deleted_text"] += 1
            continue
        if not text:
            exclusions["empty_text"] += 1
            continue

        parent = by_id.get(utt.parent_id) if utt.parent_id else None
        yield ClassificationExample(
            id=utt.id,
            text=text,
            label=label,
            metadata={
                "conversation_id": conversation.id,
                "conversation_source_id": conversation.source_id,
                "subreddit": subreddit,
                "title": conversation.title,
                "depth": utt.depth,
                "is_root": utt.parent_id is None,
                "parent_id": utt.parent_id,
                "parent_text": parent.text.strip() if parent and not parent.is_deleted else None,
                "score": utt.score,
                "annotator_agreement": utt.metadata.get("annotator_agreement"),
            },
        )
