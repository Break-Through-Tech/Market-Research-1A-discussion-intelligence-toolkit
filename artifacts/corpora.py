"""
Dataset loaders for the labeled NLP corpora used in evaluation.

Supports:

  * ConvoKit Coarse Discourse Corpus — discourse-act labels per utterance
  * ConvoKit CGA-CMV — derailment labels per ChangeMyView thread
  * Generic JSONL loaders for custom labeled data

Each loader returns ClassificationExample / RegressionExample objects with
text + label + metadata. Splits are produced by hashing a grouping key
(typically `conversation_id`) so that all examples from the same source
conversation land in the same split. This prevents thread-level leakage.

ConvoKit is a lazy runtime dependency — imported only inside loader methods
that actually need it. The harness works without ConvoKit installed if you
only use the JSONL loaders.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Example types
# ---------------------------------------------------------------------------


class ClassificationExample(BaseModel):
    """A single labeled example for classification tasks."""

    id: str
    text: str
    label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegressionExample(BaseModel):
    """A single labeled example for regression tasks."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Split containers
# ---------------------------------------------------------------------------


@dataclass
class ClassificationSplits:
    """Train/dev/test splits for a classification dataset."""

    train: list[ClassificationExample]
    dev: list[ClassificationExample]
    test: list[ClassificationExample]

    @property
    def labels(self) -> list[str]:
        """Sorted label inventory observed in the train split."""
        return sorted({ex.label for ex in self.train})

    def summary(self) -> str:
        return f"train={len(self.train)} dev={len(self.dev)} test={len(self.test)}"


@dataclass
class RegressionSplits:
    """Train/dev/test splits for a regression dataset."""

    train: list[RegressionExample]
    dev: list[RegressionExample]
    test: list[RegressionExample]

    def summary(self) -> str:
        return f"train={len(self.train)} dev={len(self.dev)} test={len(self.test)}"


# ---------------------------------------------------------------------------
# Group-stratified splitting (the leakage-prevention primitive)
# ---------------------------------------------------------------------------


def _bucket_for(key: str, seed: int) -> float:
    """Deterministic float in [0, 1) from a group key + seed."""
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return (int(h, 16) % 1000) / 1000.0


def split_by_group(
    examples: list[ClassificationExample],
    group_key: Callable[[ClassificationExample], str],
    dev_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> ClassificationSplits:
    """Split examples into train/dev/test buckets, keeping each group whole.

    All examples whose `group_key(ex)` hashes into the same bucket end up in
    the same split. This is how we prevent leakage when one source thread
    contributes many labeled utterances.

    The split is deterministic for a given (seed, group_key) pair.
    """
    if dev_frac + test_frac >= 1.0:
        raise ValueError("dev_frac + test_frac must be < 1.0")

    train: list[ClassificationExample] = []
    dev: list[ClassificationExample] = []
    test: list[ClassificationExample] = []
    test_cutoff = test_frac
    dev_cutoff = test_frac + dev_frac

    for ex in examples:
        bucket = _bucket_for(group_key(ex), seed)
        if bucket < test_cutoff:
            test.append(ex)
        elif bucket < dev_cutoff:
            dev.append(ex)
        else:
            train.append(ex)

    return ClassificationSplits(train=train, dev=dev, test=test)


# ---------------------------------------------------------------------------
# ConvoKit Coarse Discourse loader
# ---------------------------------------------------------------------------


class CoarseDiscourseLoader:
    """Loader for ConvoKit's Coarse Discourse Corpus.

    ~115k Reddit utterances with majority-vote discourse act labels:
    question, answer, agreement, disagreement, humor, announcement,
    appreciation, negative_reaction, elaboration, other.

    The 'other' bucket is dropped — it's a catch-all that pollutes F1 numbers.
    Splits are produced by hashing `conversation_id` to keep threads intact.
    """

    NAME = "reddit-coarse-discourse-corpus"
    DROP_LABELS: frozenset[str] = frozenset({"other"})

    def __init__(
        self,
        dev_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.dev_frac = dev_frac
        self.test_frac = test_frac
        self.seed = seed

    def load(self, corpus_or_path: Any = None) -> ClassificationSplits:
        """Load the corpus and produce conversation-stratified splits.

        Args:
            corpus_or_path: One of:
                * None — download via `convokit.download(NAME)`
                * str / Path — path to an already-downloaded corpus
                * convokit.Corpus instance — use directly
        """
        corpus = self._resolve_corpus(corpus_or_path)
        examples = list(self._iter_examples(corpus))
        splits = split_by_group(
            examples,
            group_key=lambda ex: ex.metadata["conversation_id"],
            dev_frac=self.dev_frac,
            test_frac=self.test_frac,
            seed=self.seed,
        )
        log.info("Coarse Discourse: %s", splits.summary())
        return splits

    def _resolve_corpus(self, corpus_or_path: Any) -> Any:
        try:
            from convokit import Corpus, download
        except ImportError as e:
            raise ImportError(
                "ConvoKit is required for CoarseDiscourseLoader. "
                "Install with: pip install convokit"
            ) from e

        if corpus_or_path is None:
            log.info("Downloading ConvoKit corpus %s", self.NAME)
            return Corpus(filename=download(self.NAME))
        if isinstance(corpus_or_path, (str, Path)):
            return Corpus(filename=str(corpus_or_path))
        return corpus_or_path

    def _iter_examples(self, corpus: Any) -> Iterator[ClassificationExample]:
        for utt in corpus.iter_utterances():
            label = utt.meta.get("majority_type")
            if not label or label in self.DROP_LABELS:
                continue
            text = (utt.text or "").strip()
            if not text:
                continue
            yield ClassificationExample(
                id=utt.id,
                text=text,
                label=label,
                metadata={
                    "conversation_id": utt.conversation_id,
                    "subreddit": utt.meta.get("subreddit"),
                },
            )


# ---------------------------------------------------------------------------
# ConvoKit CGA-CMV loader
# ---------------------------------------------------------------------------


class CgaCmvLoader:
    """Loader for ConvoKit's Conversations Gone Awry — ChangeMyView corpus.

    6,842 CMV threads with binary derailment labels at the conversation level.
    Each example is one thread, with text aggregated from the first `n_utterances`
    comments (default 4 — enough context, before the derailment usually fires).
    Label is 'derailed' or 'civil'.

    Splits are produced by hashing `conversation_id` (same as Coarse Discourse).
    Note: the corpus is balanced by design (pairs of awry/non-awry threads
    matched on opener), so train/dev/test stay roughly balanced too.
    """

    NAME = "conversations-gone-awry-cmv-corpus"

    def __init__(
        self,
        n_utterances: int = 4,
        dev_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_utterances = n_utterances
        self.dev_frac = dev_frac
        self.test_frac = test_frac
        self.seed = seed

    def load(self, corpus_or_path: Any = None) -> ClassificationSplits:
        corpus = self._resolve_corpus(corpus_or_path)
        examples = list(self._iter_examples(corpus))
        splits = split_by_group(
            examples,
            group_key=lambda ex: ex.metadata["conversation_id"],
            dev_frac=self.dev_frac,
            test_frac=self.test_frac,
            seed=self.seed,
        )
        log.info("CGA-CMV: %s", splits.summary())
        return splits

    def _resolve_corpus(self, corpus_or_path: Any) -> Any:
        try:
            from convokit import Corpus, download
        except ImportError as e:
            raise ImportError(
                "ConvoKit is required for CgaCmvLoader. "
                "Install with: pip install convokit"
            ) from e

        if corpus_or_path is None:
            log.info("Downloading ConvoKit corpus %s", self.NAME)
            return Corpus(filename=download(self.NAME))
        if isinstance(corpus_or_path, (str, Path)):
            return Corpus(filename=str(corpus_or_path))
        return corpus_or_path

    def _iter_examples(self, corpus: Any) -> Iterator[ClassificationExample]:
        for conv in corpus.iter_conversations():
            # Take the first n_utterances in chronological order
            utts = sorted(conv.iter_utterances(), key=lambda u: u.timestamp or 0)[: self.n_utterances]
            if not utts:
                continue
            text = "\n\n".join((u.text or "").strip() for u in utts if u.text)
            if not text.strip():
                continue
            derailed = bool(conv.meta.get("has_removed_comment", False))
            yield ClassificationExample(
                id=conv.id,
                text=text,
                label="derailed" if derailed else "civil",
                metadata={
                    "conversation_id": conv.id,
                    "n_utterances_used": len(utts),
                },
            )


# ---------------------------------------------------------------------------
# JSONL loaders for custom data
# ---------------------------------------------------------------------------


def load_jsonl_classification(path: str | Path) -> list[ClassificationExample]:
    """Load classification examples from a JSONL file (one JSON object per line)."""
    examples: list[ClassificationExample] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(ClassificationExample.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"Failed to parse {path}:{line_num}: {e}") from e
    return examples


def load_jsonl_regression(path: str | Path) -> list[RegressionExample]:
    """Load regression examples from a JSONL file (one JSON object per line)."""
    examples: list[RegressionExample] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(RegressionExample.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"Failed to parse {path}:{line_num}: {e}") from e
    return examples
