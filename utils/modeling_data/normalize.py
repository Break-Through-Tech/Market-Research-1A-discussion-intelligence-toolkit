"""
Normalize raw Coarse Discourse threads into canonical `Conversation` objects.

Source quirks and how each is handled (all are counted in NormalizationStats):

  * No timestamps. The corpus ships `timestamp: null` for every utterance, but
    the canonical schema requires `created_at`. We write the Unix epoch as a
    sentinel and set `metadata["timestamp_missing"] = True`. Never sort or
    split on `created_at` for this corpus.

  * Orphaned replies. ~200 utterances reply to a comment that was not included
    in the annotated release. The schema requires every parent to resolve, so
    orphans are re-attached to the thread root; the original parent ID is kept
    in `metadata["source_parent_id"]` and `depth` keeps the source value.

  * Deleted content/authors. `[deleted]`/`[removed]` text sets
    `Utterance.is_deleted`; a `[deleted]` author becomes a Speaker with
    `is_deleted=True`. Structure is preserved; the modeling examples skip them.

  * Usernames. Speaker handles are pseudonymized (stable SHA-256 prefix) by
    default, so normalized files never carry Reddit usernames.

  * Gold labels. The annotated discourse act goes to
    `metadata["discourse_act_gold"]`, NOT to `Utterance.discourse_act`. That
    field is the enrichment slot the classifier writes predictions into; keeping
    gold out of it means a pipeline bug can never "predict" using the answer key.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from artifacts.schema import Conversation, Speaker, Utterance
from utils.modeling_data.convokit_reader import RawUtterance
from utils.modeling_data.labels import annotator_agreement, map_label

SOURCE = "reddit"
MISSING_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc)
DELETED_MARKERS = frozenset({"[deleted]", "[removed]"})
DELETED_AUTHOR = "[deleted]"


@dataclass
class NormalizationStats:
    conversations: int = 0
    utterances: int = 0
    orphans_reparented: int = 0
    deleted_text: int = 0
    empty_text: int = 0
    deleted_authors: int = 0
    missing_timestamps: int = 0
    skipped_conversations_without_root: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def canonical_id(source_id: str) -> str:
    return f"{SOURCE}:{source_id}"


def pseudonymize(handle: str) -> str:
    return "anon_" + hashlib.sha256(handle.encode("utf-8")).hexdigest()[:12]


def _speaker_for(raw_handle: str | None, pseudonymize_speakers: bool) -> Speaker:
    handle = raw_handle or DELETED_AUTHOR
    if handle == DELETED_AUTHOR:
        return Speaker(
            id=f"{SOURCE}:u/{DELETED_AUTHOR}", handle=DELETED_AUTHOR, source=SOURCE, is_deleted=True
        )
    shown = pseudonymize(handle) if pseudonymize_speakers else handle
    return Speaker(id=f"{SOURCE}:u/{shown}", handle=shown, source=SOURCE)


def _timestamp(raw: RawUtterance) -> datetime | None:
    if raw.timestamp is None:
        return None
    return datetime.fromtimestamp(float(raw.timestamp), tz=timezone.utc)


def normalize_conversation(
    conversation_id: str,
    raw_utterances: list[RawUtterance],
    conversation_meta: dict[str, Any] | None = None,
    *,
    pseudonymize_speakers: bool = True,
    stats: NormalizationStats | None = None,
) -> Conversation | None:
    """Build one canonical Conversation from its raw utterances.

    Returns None (and counts it) if the thread's root post is missing, since a
    conversation without a root cannot satisfy the schema invariant.
    """
    stats = stats if stats is not None else NormalizationStats()
    conversation_meta = conversation_meta or {}

    present = {u.id for u in raw_utterances}
    root_raw = next(
        (u for u in raw_utterances if u.reply_to is None and u.id == conversation_id), None
    ) or next((u for u in raw_utterances if u.reply_to is None), None)
    if root_raw is None:
        stats.skipped_conversations_without_root += 1
        return None
    root_id = canonical_id(root_raw.id)

    speakers: dict[str, Speaker] = {}
    utterances: list[Utterance] = []
    for raw in raw_utterances:
        speaker = _speaker_for(raw.speaker, pseudonymize_speakers)
        speakers.setdefault(speaker.id, speaker)
        if speaker.is_deleted:
            stats.deleted_authors += 1

        meta: dict[str, Any] = {}
        if raw is root_raw:
            parent_id = None
        elif raw.reply_to is None or raw.reply_to not in present:
            # A second parentless post, or a reply whose parent is absent from
            # the release: hang it off the root to keep the tree valid.
            parent_id = root_id
            meta["source_parent_id"] = raw.reply_to
            meta["orphan_reparented"] = True
            stats.orphans_reparented += 1
        else:
            parent_id = canonical_id(raw.reply_to)

        text = raw.text or ""
        stripped = text.strip()
        is_deleted = stripped.lower() in DELETED_MARKERS
        if is_deleted:
            stats.deleted_text += 1
        elif not stripped:
            stats.empty_text += 1

        created_at = _timestamp(raw)
        if created_at is None:
            created_at = MISSING_TIMESTAMP
            meta["timestamp_missing"] = True
            stats.missing_timestamps += 1

        raw_label = raw.meta.get("majority_type")
        annotations = raw.meta.get("annotation-types")
        meta.update(
            {
                "discourse_act_gold": map_label(raw_label),
                "annotation_types": annotations,
                "annotator_agreement": annotator_agreement(annotations, raw_label),
                "majority_link": raw.meta.get("majority_link"),
            }
        )

        utterances.append(
            Utterance(
                id=canonical_id(raw.id),
                source_id=raw.id,
                conversation_id=canonical_id(conversation_id),
                parent_id=parent_id,
                speaker_id=speaker.id,
                text=text,
                created_at=created_at,
                depth=int(raw.meta.get("post_depth") or 0),
                score=raw.meta.get("ups"),
                is_deleted=is_deleted,
                metadata=meta,
            )
        )

    stats.conversations += 1
    stats.utterances += len(utterances)
    root_created = next(u.created_at for u in utterances if u.id == root_id)
    return Conversation(
        id=canonical_id(conversation_id),
        source_id=conversation_id,
        source=SOURCE,
        title=conversation_meta.get("title"),
        url=conversation_meta.get("url"),
        created_at=root_created,
        utterances=utterances,
        speakers=speakers,
        metadata={
            "corpus": "reddit-coarse-discourse-corpus",
            "subreddit": conversation_meta.get("subreddit"),
        },
    )
