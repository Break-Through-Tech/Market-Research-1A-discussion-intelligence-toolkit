"""
Dependency-free reader for an on-disk ConvoKit corpus directory.

`convokit` is a heavy install (spaCy, torch-adjacent pins) and lags new Python
releases. Dataset preparation only needs the two files below, so we read them
directly instead:

    <corpus_dir>/utterances.jsonl    one JSON object per utterance
    <corpus_dir>/conversations.json  {conversation_id: {subreddit, url, title}}

The Coarse Discourse download uses the ConvoKit v1 on-disk layout, where the
speaker field is called `user` and the parent field `reply-to`. Newer corpora
use `speaker` / `reply_to`; both spellings are accepted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

CORPUS_NAME = "reddit-coarse-discourse-corpus"


@dataclass(frozen=True)
class RawUtterance:
    id: str
    conversation_id: str
    reply_to: str | None
    speaker: str | None
    text: str | None
    timestamp: int | float | None
    meta: dict[str, Any] = field(default_factory=dict)


def default_corpus_dir() -> Path:
    """Where `convokit.download()` puts the corpus, falling back to a repo-local copy."""
    home = Path.home() / ".convokit" / "saved-corpora" / CORPUS_NAME
    if home.exists():
        return home
    return Path(".convokit") / "saved-corpora" / CORPUS_NAME


def _raw_from_json(d: dict[str, Any]) -> RawUtterance:
    return RawUtterance(
        id=d["id"],
        conversation_id=d.get("root") or d.get("conversation_id"),
        reply_to=d.get("reply-to", d.get("reply_to")),
        speaker=d.get("user", d.get("speaker")),
        text=d.get("text"),
        timestamp=d.get("timestamp"),
        meta=d.get("meta") or {},
    )


def iter_raw_utterances(corpus_dir: str | Path) -> Iterator[RawUtterance]:
    path = Path(corpus_dir) / "utterances.jsonl"
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield _raw_from_json(json.loads(line))
            except (KeyError, json.JSONDecodeError) as e:
                raise ValueError(f"Malformed utterance at {path}:{line_num}: {e}") from e


def load_conversation_meta(corpus_dir: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(corpus_dir) / "conversations.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_by_conversation(utterances: Iterator[RawUtterance]) -> dict[str, list[RawUtterance]]:
    """Group utterances by conversation, preserving source order within each."""
    grouped: dict[str, list[RawUtterance]] = {}
    for u in utterances:
        grouped.setdefault(u.conversation_id, []).append(u)
    return grouped


def file_sha256(path: str | Path) -> str:
    """Content hash recorded in the dataset card so outputs trace to exact inputs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
