"""Adapt the tracked reference JSONL through the established dataset normalizer."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from artifacts.schema import Conversation
from utils.modeling_data.convokit_reader import RawUtterance
from utils.modeling_data.normalize import normalize_conversation

from .validation import validate_conversation


class ReferenceUtterance(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    id: str = Field(pattern=r"^t[13]_[A-Za-z0-9_]+$")
    parent_id: str | None
    text: str | None
    depth: int = Field(ge=0)
    discourse_act: str | None = None
    score: int | None = None
    speaker: str | None = None
    timestamp: int | float | None = None
    url: str | None = None


class ReferenceConversation(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    source: str
    conversation_id: str = Field(pattern=r"^t3_[A-Za-z0-9_]+$")
    title: str | None = None
    url: str | None = None
    subreddit: str | None = None
    utterances: list[ReferenceUtterance]


def adapt_reference(record: dict) -> Conversation:
    """Preserve gold labels as metadata; mark absent time and author information.

    Uses the dataset pipeline's UTC epoch sentinel and orphan-repair policy.
    Missing speakers use a shared, explicitly flagged unknown placeholder,
    without claiming that the author deleted an account or that each comment
    has a distinct author. No per-comment Reddit permalink is invented.
    """
    source = ReferenceConversation.model_validate(record)
    if source.source != "convokit:reddit-coarse-discourse-corpus":
        raise ValueError(
            "This adapter only accepts the documented Coarse Discourse reference format"
        )
    ids = [u.id for u in source.utterances]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source utterance IDs")
    if sum(u.parent_id is None for u in source.utterances) != 1 or not any(
        u.id == source.conversation_id and u.parent_id is None for u in source.utterances
    ):
        raise ValueError("Reference conversation must have its submission as the single root")
    rows = [
        RawUtterance(
            id=u.id,
            conversation_id=source.conversation_id,
            reply_to=u.parent_id,
            speaker=u.speaker or "__source_unknown__",
            text=u.text,
            timestamp=u.timestamp,
            meta={"post_depth": u.depth, "majority_type": u.discourse_act, "ups": u.score},
        )
        for u in source.utterances
    ]
    conv = normalize_conversation(
        source.conversation_id,
        rows,
        {"title": source.title, "url": source.url, "subreddit": source.subreddit},
    )
    if conv is None:
        raise ValueError("Reference conversation could not be normalized")
    originals = {u.id: u for u in source.utterances}
    for u in conv.utterances:
        original = originals[u.source_id]
        u.metadata["speaker_missing"] = not bool(original.speaker)
        if original.url is not None:
            u.metadata["source_url"] = original.url
    conv.metadata.update(
        {
            "reference_source": source.source,
            "timestamp_missing": conv.root().metadata.get("timestamp_missing", False),
            "speaker_identity_policy": "Shared unknown placeholder when source omits speaker; no identity inferred.",
        }
    )
    return validate_conversation(conv)


def load_reference_conversations(path: str | Path) -> list[Conversation]:
    conversations = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                conversations.append(adapt_reference(json.loads(line)))
            except (ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
                raise ValueError(
                    f"Invalid reference conversation at {path}:{line_number}: {exc}"
                ) from exc
    return conversations
