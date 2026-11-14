"""
Reddit Pushshift dump connector.

Reads pre-filtered Pushshift submissions and comments parquet files and emits
canonical Conversation objects. The connector itself is intentionally NOT
responsible for handling the full 89GB Pushshift dump — that's a one-time
prep step done in the starter notebook.

Expected upstream prep (starter notebook handles this):
  1. Download `fddemarco/pushshift-reddit` + `fddemarco/pushshift-reddit-comments`
  2. Filter to chosen subreddits and date range
  3. Save filtered subsets as new parquet files
  4. Point the connector at those filtered files

Pushshift schema notes:
  * Submission `id` is the raw post id (e.g. 'abc123'). We treat the canonical
    form as `t3_<id>` (Reddit's submission prefix convention) and source-prefix
    that as `reddit:t3_<id>`.
  * Comment `id` is the raw comment id (e.g. 'def456'). Canonical: `reddit:t1_<id>`.
  * Comment `link_id` is the parent submission id, prefixed as `t3_<id>`.
  * Comment `parent_id` is either `t3_<sub_id>` (direct reply to submission)
    or `t1_<comment_id>` (reply to another comment).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from pydantic import Field

if __package__:
    from .connector import (
        Connector,
        ConnectorConfig,
        ConversationNotFoundError,
        MalformedConversationError,
    )
    from .schema import Conversation, Speaker, Utterance
else:  # Preserve the original starter-notebook import path.
    from connector import (
        Connector,
        ConnectorConfig,
        ConversationNotFoundError,
        MalformedConversationError,
    )
    from schema import Conversation, Speaker, Utterance

log = logging.getLogger(__name__)


SOURCE = "reddit"
DELETED_MARKERS = {"[deleted]", "[removed]", "None", ""}


# ---------------------------------------------------------------------------
# Module-level helpers (pure, no connector state)
# ---------------------------------------------------------------------------


def _canonical_id(raw_id: str) -> str:
    """Return the source-prefixed canonical form of a Pushshift raw ID."""
    return f"{SOURCE}:{raw_id}"


def _ensure_submission_prefix(raw_id: str) -> str:
    """Pushshift submission rows sometimes have bare ids ('abc'), sometimes 't3_abc'.

    Normalize to the t3_ form so canonical IDs are consistent.
    """
    return raw_id if raw_id.startswith("t3_") else f"t3_{raw_id}"


def _ensure_comment_prefix(raw_id: str) -> str:
    """Same idea as submissions, for comments (`t1_` prefix)."""
    return raw_id if raw_id.startswith("t1_") else f"t1_{raw_id}"


def _to_utc(created_utc: Any) -> datetime:
    """Convert Pushshift unix timestamp (int/float/str) to UTC datetime."""
    return datetime.fromtimestamp(float(created_utc), tz=timezone.utc)


def _is_deleted(value: Any) -> bool:
    """True if a Pushshift author/body field indicates deletion."""
    return str(value) in DELETED_MARKERS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class RedditDumpConfig(ConnectorConfig):
    """Configuration for the Reddit Pushshift dump connector."""

    submissions_path: Path = Field(
        description="Path to the filtered submissions parquet file"
    )
    comments_path: Path = Field(
        description="Path to the filtered comments parquet file"
    )
    min_comments: int = Field(
        default=3,
        ge=0,
        description="Skip submissions with fewer than this many comments in bulk iteration",
    )


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class RedditDumpConnector(Connector):
    """Connector for Pushshift Reddit dumps in parquet format.

    Loads both parquet files lazily on first access, indexes comments by `link_id`,
    and emits Conversations in submission-row order (typically chronological).
    """

    source = SOURCE

    def __init__(self, config: RedditDumpConfig) -> None:
        super().__init__(config)
        self.config: RedditDumpConfig = config
        self._submissions: pd.DataFrame | None = None
        self._comments_by_link: dict[str, pd.DataFrame] = {}

    # --- Loading -----------------------------------------------------------

    def _load(self) -> None:
        """Read both parquet files and index comments by `link_id`. Idempotent."""
        if self._submissions is not None:
            return
        log.info("Loading submissions from %s", self.config.submissions_path)
        self._submissions = pd.read_parquet(self.config.submissions_path)
        log.info("Loading comments from %s", self.config.comments_path)
        comments = pd.read_parquet(self.config.comments_path)
        self._comments_by_link = {
            link_id: group for link_id, group in comments.groupby("link_id")
        }
        log.info(
            "Indexed %d submissions across %d threads of comments",
            len(self._submissions),
            len(self._comments_by_link),
        )

    # --- Public interface --------------------------------------------------

    def fetch_conversation(self, conversation_id: str) -> Conversation:
        """Look up a submission by canonical ID and assemble its Conversation."""
        self._load()
        prefix = f"{SOURCE}:"
        if not conversation_id.startswith(prefix):
            raise ConversationNotFoundError(
                f"ID {conversation_id!r} does not match source {SOURCE!r}"
            )
        link_id = conversation_id.removeprefix(prefix)  # e.g. 't3_abc'
        raw_id = link_id.removeprefix("t3_")            # e.g. 'abc'

        rows = self._submissions[self._submissions["id"] == raw_id]
        if rows.empty:
            # Try the prefixed form in case the dump stores 't3_<id>' in `id`
            rows = self._submissions[self._submissions["id"] == link_id]
        if rows.empty:
            raise ConversationNotFoundError(f"No submission with id={raw_id}")

        return self._build_conversation(rows.iloc[0])

    def list_conversations(self, limit: int | None = None) -> Iterator[Conversation]:
        """Iterate submissions; yield assembled Conversations meeting min_comments."""
        self._load()
        yielded = 0
        for _, sub_row in self._submissions.iterrows():
            if limit is not None and yielded >= limit:
                return
            try:
                conv = self._build_conversation(sub_row)
            except MalformedConversationError as e:
                if self.config.strict:
                    raise
                log.warning("Skipping malformed submission %s: %s", sub_row.get("id"), e)
                continue

            # -1 because the root utterance is the submission itself
            if (len(conv) - 1) < self.config.min_comments:
                continue

            yielded += 1
            yield conv

    # --- Internal: assembly ------------------------------------------------

    def _build_conversation(self, sub_row: pd.Series) -> Conversation:
        """Convert one submission row + its matching comments into a Conversation."""
        sub_link_id = _ensure_submission_prefix(str(sub_row["id"]))
        sub_canonical = _canonical_id(sub_link_id)

        comment_rows = self._comments_by_link.get(sub_link_id, pd.DataFrame())

        speakers: dict[str, Speaker] = {}
        utterances: list[Utterance] = []

        # --- Root utterance from the submission --------------------------
        root_speaker = self._speaker_from_handle(sub_row.get("author"))
        speakers[root_speaker.id] = root_speaker

        title = str(sub_row.get("title") or "").strip()
        selftext = str(sub_row.get("selftext") or "").strip()
        root_text = f"{title}\n\n{selftext}".strip() if selftext else title

        root = Utterance(
            id=sub_canonical,
            source_id=sub_link_id,
            conversation_id=sub_canonical,
            parent_id=None,
            speaker_id=root_speaker.id,
            text=root_text,
            created_at=_to_utc(sub_row["created_utc"]),
            depth=0,
            score=int(sub_row.get("score", 0)) if pd.notna(sub_row.get("score")) else None,
            is_deleted=_is_deleted(sub_row.get("author")),
            metadata={
                "title": title,
                "subreddit": sub_row.get("subreddit"),
                "num_comments": int(sub_row.get("num_comments", 0)) if pd.notna(sub_row.get("num_comments")) else 0,
                "permalink": sub_row.get("permalink"),
            },
        )
        utterances.append(root)

        # --- Comment utterances ------------------------------------------
        for _, c_row in comment_rows.iterrows():
            speaker = self._speaker_from_handle(c_row.get("author"))
            speakers.setdefault(speaker.id, speaker)

            c_canonical = _canonical_id(_ensure_comment_prefix(str(c_row["id"])))
            raw_parent = str(c_row.get("parent_id") or "")
            parent_canonical = (
                sub_canonical if raw_parent.startswith("t3_") else _canonical_id(raw_parent)
            )

            score_val = c_row.get("score")
            utterances.append(
                Utterance(
                    id=c_canonical,
                    source_id=str(c_row["id"]),
                    conversation_id=sub_canonical,
                    parent_id=parent_canonical,
                    speaker_id=speaker.id,
                    text=str(c_row.get("body") or ""),
                    created_at=_to_utc(c_row["created_utc"]),
                    depth=0,  # filled in below
                    score=int(score_val) if pd.notna(score_val) else None,
                    is_deleted=_is_deleted(c_row.get("body")),
                )
            )

        # --- Repair orphan parents ---------------------------------------
        # Comments whose parent was deleted out of the dump get re-parented to root.
        known_ids = {u.id for u in utterances}
        for u in utterances[1:]:
            if u.parent_id not in known_ids:
                u.parent_id = sub_canonical

        # --- Compute depths via fixed-point iteration --------------------
        depth_by_id: dict[str, int] = {sub_canonical: 0}
        progress = True
        while progress:
            progress = False
            for u in utterances:
                if u.id in depth_by_id or u.parent_id is None:
                    continue
                if u.parent_id in depth_by_id:
                    u.depth = depth_by_id[u.parent_id] + 1
                    depth_by_id[u.id] = u.depth
                    progress = True

        if len(utterances) == 0:
            raise MalformedConversationError("No utterances assembled")

        permalink = sub_row.get("permalink")
        return Conversation(
            id=sub_canonical,
            source_id=sub_link_id,
            source=SOURCE,
            title=title or None,
            url=f"https://www.reddit.com{permalink}" if permalink else None,
            created_at=_to_utc(sub_row["created_utc"]),
            utterances=utterances,
            speakers=speakers,
            metadata={"subreddit": sub_row.get("subreddit")},
        )

    # --- Internal: speaker helpers ----------------------------------------

    def _speaker_from_handle(self, handle: Any) -> Speaker:
        """Construct a Speaker from a Pushshift author field."""
        handle_str = str(handle) if handle is not None else "[deleted]"
        return Speaker(
            id=_canonical_id(f"u/{handle_str}"),
            handle=handle_str,
            source=SOURCE,
            is_deleted=_is_deleted(handle_str),
        )
