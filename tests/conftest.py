"""Shared fixtures. Puts the repo root on sys.path so `artifacts.*` and `utils.*` import."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def raw_utt(id, root, reply_to, text, label="answer", user="alice", depth=1, annotations=None, ups=1):
    """One utterance in ConvoKit v1 on-disk format (as shipped for Coarse Discourse)."""
    return {
        "id": id,
        "root": root,
        "reply-to": reply_to,
        "user": user,
        "text": text,
        "timestamp": None,
        "meta": {
            "post_depth": depth,
            "majority_type": label,
            "majority_link": "none",
            "annotation-types": annotations if annotations is not None else [label] * 3,
            "annotation-links": ["none"] * 3,
            "ups": ups,
        },
    }


def tiny_thread(root: str, n_replies: int = 3) -> list[dict]:
    utts = [raw_utt(root, root, None, f"How do I fix thing {root}?", "question", "op", 0)]
    for i in range(n_replies):
        utts.append(raw_utt(f"t1_{root}_{i}", root, root, f"Try option {i} for {root}", "answer", f"u{i}"))
    return utts


def write_corpus(directory: Path, utterances: list[dict], conversations: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "utterances.jsonl", "w") as f:
        for u in utterances:
            f.write(json.dumps(u) + "\n")
    with open(directory / "conversations.json", "w") as f:
        json.dump(conversations, f)
    return directory


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """60 small threads across two subreddits, plus one thread with every edge case."""
    utts: list[dict] = []
    convs: dict[str, dict] = {}
    for i in range(60):
        root = f"t3_c{i:03d}"
        utts += tiny_thread(root)
        convs[root] = {"subreddit": "buildapc" if i % 2 else "askscience", "url": None, "title": f"T{i}"}
    edge = "t3_edge"
    utts += [
        raw_utt(edge, edge, None, "Root post", "announcement", "op", 0),
        raw_utt("t1_e1", edge, edge, "[deleted]", "answer", "[deleted]"),
        raw_utt("t1_e2", edge, edge, None, "answer", "bob"),
        raw_utt("t1_e3", edge, edge, "meh", None, "bob", annotations=["humor", "answer", "other"]),
        raw_utt("t1_e4", edge, edge, "something else", "other", "bob"),
        raw_utt("t1_e5", edge, "t1_missing", "orphan reply", "negativereaction", "carol", 3),
    ]
    convs[edge] = {"subreddit": "buildapc", "url": "https://reddit.com/x", "title": "Edge"}
    return write_corpus(tmp_path / "corpus", utts, convs)
