"""
Conversation-quality dataset from ConvoKit's Conversations Gone Awry, CMV.

Target
------
Each of the 6,842 ChangeMyView conversations is labeled by whether a moderator
later removed one of its comments for incivility (`has_removed_comment`). The
corpus is built in matched pairs: every derailed conversation is paired with a
civil one from the same original post, so the topic is held roughly constant
and the label is balanced 50/50.

The quality score we model is P(civil | comments so far), in [0, 1], matching
`Utterance.quality_score`'s range. Higher means the discussion is more likely to
stay civil.

What the model may see
----------------------
Derailed conversations end at the removed comment, so the last utterance is
the label itself. We drop the final utterance of every conversation (civil ones
too, so length does not give the label away) and only use the earlier comments.
This is the standard forecasting setup for this corpus.

Inside that context we also drop:

    moderator_notice   "your comment has been removed" style notices, which
                       state the label outright (about 1% of conversations)
    deleted_text       [deleted] / [removed]
    empty_text         empty or whitespace

and skip conversations with no comment left after filtering.

Splits
------
The corpus ships an official train / val / test assignment (4,106 / 1,368 /
1,368). It keeps both members of each pair in the same split, so a model never
sees the civil twin of a test conversation during training. We keep that
assignment and call val "dev".
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from artifacts.corpora import ClassificationExample, ClassificationSplits
from utils.modeling_data.leakage import (
    CheckResult,
    LeakageReport,
    check_conversation_disjoint,
    check_example_disjoint,
    check_label_coverage,
    check_label_distribution,
    check_text_overlap,
)

CORPUS_NAME = "conversations-gone-awry-cmv-corpus"
LABELS = ("civil", "derailed")
SPLIT_NAMES = {"train": "train", "val": "dev", "test": "test"}

_DELETED = {"[deleted]", "[removed]"}
# Narrow on purpose: only phrasings that announce a removal. Broader patterns
# ("mod", "rule") also hit ordinary CMV arguments about rules and moderation.
MODERATOR_NOTICE = re.compile(
    r"(your|this) (comment|post|submission) has been removed"
    r"|has been removed for (breaking|violating)"
    r"|comment rule \d",
    re.IGNORECASE,
)


def default_corpus_dir() -> Path:
    home = Path.home() / ".convokit" / "saved-corpora" / CORPUS_NAME
    if home.exists():
        return home
    return Path(".convokit") / "saved-corpora" / CORPUS_NAME


@dataclass
class QualityDataset:
    splits: ClassificationSplits
    exclusions: Counter[str] = field(default_factory=Counter)

    def comments(self, split: str) -> list[list[str]]:
        return [ex.metadata["comments"] for ex in getattr(self.splits, split)]

    def civil(self, split: str) -> list[int]:
        return [int(ex.label == "civil") for ex in getattr(self.splits, split)]

    def groups(self, split: str) -> list[str]:
        """One key per matched pair. Each member's `pair_id` names the other
        member, so the key is the smaller of the two IDs."""
        return [pair_key(ex) for ex in getattr(self.splits, split)]


def pair_key(ex: ClassificationExample) -> str:
    return min(ex.id, ex.metadata["pair_id"] or ex.id)


def read_corpus(corpus_dir: str | Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Return (conversation meta by id, utterances grouped by conversation id)."""
    corpus_dir = Path(corpus_dir).expanduser()
    with open(corpus_dir / "conversations.json") as f:
        conversations = {cid: c.get("meta", {}) for cid, c in json.load(f).items()}
    grouped: dict[str, list[dict]] = defaultdict(list)
    with open(corpus_dir / "utterances.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            # The `parsed` dependency trees are most of the file and unused here.
            d.get("meta", {}).pop("parsed", None)
            grouped[d.get("conversation_id") or d.get("root")].append(d)
    return conversations, grouped


def reply_chain(utterances: list[dict]) -> list[dict]:
    """Order a conversation by following replies from the root.

    CGA-CMV conversations are linear chains. Timestamps break ties if a
    conversation is not a clean chain.
    """
    by_parent: dict[Any, list[dict]] = defaultdict(list)
    for u in utterances:
        by_parent[u.get("reply-to", u.get("reply_to"))].append(u)
    for children in by_parent.values():
        children.sort(key=lambda u: (u.get("timestamp") or 0, u["id"]))
    ordered: list[dict] = []
    frontier = list(by_parent.get(None, []))
    while frontier:
        u = frontier.pop(0)
        ordered.append(u)
        frontier = by_parent.get(u["id"], []) + frontier
    if len(ordered) != len(utterances):
        return sorted(utterances, key=lambda u: (u.get("timestamp") or 0, u["id"]))
    return ordered


def context_comments(
    chain: list[dict], exclusions: Counter[str], max_comments: int | None = None
) -> list[str]:
    """Texts the model may see: every comment but the last, filtered."""
    kept: list[str] = []
    for u in chain[:-1]:
        text = (u.get("text") or "").strip()
        if not text:
            exclusions["empty_text"] += 1
        elif text in _DELETED:
            exclusions["deleted_text"] += 1
        elif MODERATOR_NOTICE.search(text):
            exclusions["moderator_notice"] += 1
        else:
            kept.append(text)
    if max_comments is not None:
        kept = kept[:max_comments]
    return kept


def iter_examples(
    conversations: dict[str, dict],
    grouped: dict[str, list[dict]],
    exclusions: Counter[str],
    max_comments: int | None = None,
) -> Iterator[ClassificationExample]:
    for cid in sorted(grouped):
        meta = conversations.get(cid, {})
        chain = reply_chain(grouped[cid])
        exclusions["final_comment_dropped"] += 1
        comments = context_comments(chain, exclusions, max_comments)
        if not comments:
            exclusions["conversation_without_context"] += 1
            continue
        yield ClassificationExample(
            id=cid,
            text="\n\n".join(comments),
            label="derailed" if meta.get("has_removed_comment") else "civil",
            metadata={
                "conversation_id": cid,
                "pair_id": meta.get("pair_id"),
                "split": SPLIT_NAMES[meta.get("split", "train")],
                "comments": comments,
                "n_comments_total": len(chain),
            },
        )


def split_examples(examples: Iterable[ClassificationExample]) -> ClassificationSplits:
    buckets: dict[str, list[ClassificationExample]] = {"train": [], "dev": [], "test": []}
    for ex in examples:
        buckets[ex.metadata["split"]].append(ex)
    return ClassificationSplits(**buckets)


def load_quality_dataset(
    corpus_dir: str | Path | None = None, max_comments: int | None = None
) -> QualityDataset:
    conversations, grouped = read_corpus(corpus_dir or default_corpus_dir())
    exclusions: Counter[str] = Counter()
    splits = split_examples(iter_examples(conversations, grouped, exclusions, max_comments))
    return QualityDataset(splits=splits, exclusions=exclusions)


def check_pair_disjoint(splits: ClassificationSplits) -> CheckResult:
    seen: dict[str, set[str]] = defaultdict(set)
    for name in ("train", "dev", "test"):
        for ex in getattr(splits, name):
            seen[pair_key(ex)].add(name)
    shared = sorted(k for k, names in seen.items() if len(names) > 1)
    if shared:
        return CheckResult("pair_disjoint", "fail",
                           f"{len(shared)} matched pairs are split across train/dev/test",
                           {"examples": shared[:10]})
    return CheckResult("pair_disjoint", "pass", f"{len(seen)} matched pairs, each in one split")


def run_quality_leakage_checks(splits: ClassificationSplits) -> LeakageReport:
    return LeakageReport([
        check_conversation_disjoint(splits),
        check_pair_disjoint(splits),
        check_example_disjoint(splits),
        check_label_coverage(splits),
        check_label_distribution(splits),
        check_text_overlap(splits),
    ])
