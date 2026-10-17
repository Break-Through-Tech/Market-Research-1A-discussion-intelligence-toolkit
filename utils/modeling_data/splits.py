"""
Conversation-level train/dev/test assignment and the split manifest.

Every utterance in a thread lands in the same split: replies quote, paraphrase
and answer each other, so splitting a thread across train and test would let a
model see near-copies of test text during training.

Assignment reuses the exact hash bucketing from `artifacts.corpora.split_by_group`
(md5 of "{seed}:{conversation_id}"), keyed on the raw conversation ID. That
makes these splits identical, at the thread level, to the ones
`CoarseDiscourseLoader` produced for the first baselines (PR #18), so earlier
baseline numbers stay comparable.

Hashing makes the assignment a pure function of (conversation ID, seed,
fractions): adding, removing or re-filtering other threads never moves a thread
to a different split. The manifest CSV records the result so it can be
committed, reviewed, and reused without re-running anything.
"""

from __future__ import annotations

import csv
from pathlib import Path

from artifacts.corpora import _bucket_for

SPLITS: tuple[str, ...] = ("train", "dev", "test")
MANIFEST_FIELDS = ("conversation_id", "subreddit", "split")


def assign_split(
    conversation_id: str, seed: int = 42, dev_frac: float = 0.1, test_frac: float = 0.1
) -> str:
    if dev_frac < 0 or test_frac < 0 or dev_frac + test_frac >= 1.0:
        raise ValueError("dev_frac and test_frac must be >= 0 and sum to < 1.0")
    bucket = _bucket_for(conversation_id, seed)
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + dev_frac:
        return "dev"
    return "train"


def write_manifest(rows: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["conversation_id"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) or "" for k in MANIFEST_FIELDS})


def read_manifest(path: str | Path) -> dict[str, str]:
    """Return {raw conversation_id: split} from a manifest CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["conversation_id"]: row["split"] for row in csv.DictReader(f)}
