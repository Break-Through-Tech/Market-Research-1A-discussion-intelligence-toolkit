"""
Leakage and sanity checks over prepared train/dev/test splits.

Each check returns a CheckResult with a status:

    fail   the splits are invalid for evaluation; the build exits non-zero
    warn   valid, but something a reader of the metrics should know about
    pass   nothing to report

Checks:

  conversation_disjoint   no thread contributes examples to two splits  (fail)
  example_disjoint        no example ID appears in two splits            (fail)
  manifest_consistency    every example sits in its thread's manifest split (fail)
  label_coverage          every train label also appears in dev and test (warn)
  label_distribution      max |p_train(label) - p_eval(label)|           (warn > threshold)
  text_overlap            dev/test texts that also appear verbatim in train (warn if long)

Text overlap is reported, not removed. Most overlaps are short generic replies
("Thanks!", "yes") that genuinely recur across unrelated threads; deleting them
from dev/test would make evaluation easier than reality, not more honest. Long
duplicated texts (copy-pasted posts, bots) are the real leakage risk, so the
check warns on those and lists a few for inspection.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from artifacts.corpora import ClassificationSplits

LONG_TEXT_TOKENS = 8
DISTRIBUTION_WARN_THRESHOLD = 0.02


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakageReport:
    checks: list[CheckResult]

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}

    def format(self) -> str:
        return "\n".join(f"[{c.status.upper():4}] {c.name}: {c.detail}" for c in self.checks)


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    return re.sub(r"\s+", " ", text.lower()).strip(" .!?,;:")


def _by_split(splits: ClassificationSplits) -> dict[str, list]:
    return {"train": splits.train, "dev": splits.dev, "test": splits.test}


def check_conversation_disjoint(splits: ClassificationSplits) -> CheckResult:
    seen: dict[str, set[str]] = {}
    for name, examples in _by_split(splits).items():
        for ex in examples:
            seen.setdefault(ex.metadata["conversation_id"], set()).add(name)
    shared = sorted(cid for cid, names in seen.items() if len(names) > 1)
    if shared:
        return CheckResult("conversation_disjoint", "fail",
                           f"{len(shared)} conversations appear in more than one split",
                           {"examples": shared[:10]})
    return CheckResult("conversation_disjoint", "pass",
                       f"{len(seen)} conversations, each in exactly one split")


def check_example_disjoint(splits: ClassificationSplits) -> CheckResult:
    counts = Counter(ex.id for examples in _by_split(splits).values() for ex in examples)
    dupes = sorted(i for i, c in counts.items() if c > 1)
    if dupes:
        return CheckResult("example_disjoint", "fail",
                           f"{len(dupes)} example IDs appear more than once", {"examples": dupes[:10]})
    return CheckResult("example_disjoint", "pass", f"{len(counts)} unique example IDs")


def check_manifest_consistency(splits: ClassificationSplits, manifest: dict[str, str]) -> CheckResult:
    wrong = [
        ex.id
        for name, examples in _by_split(splits).items()
        for ex in examples
        if manifest.get(ex.metadata["conversation_source_id"]) != name
    ]
    if wrong:
        return CheckResult("manifest_consistency", "fail",
                           f"{len(wrong)} examples disagree with the split manifest",
                           {"examples": wrong[:10]})
    return CheckResult("manifest_consistency", "pass", "all examples match the split manifest")


def check_label_coverage(splits: ClassificationSplits) -> CheckResult:
    train_labels = set(splits.labels)
    missing = {
        name: sorted(train_labels - {ex.label for ex in examples})
        for name, examples in (("dev", splits.dev), ("test", splits.test))
    }
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        return CheckResult("label_coverage", "warn", f"labels missing from eval splits: {missing}", missing)
    return CheckResult("label_coverage", "pass", f"all {len(train_labels)} labels present in every split")


def label_proportions(examples: list) -> dict[str, float]:
    counts = Counter(ex.label for ex in examples)
    total = sum(counts.values()) or 1
    return {label: n / total for label, n in sorted(counts.items())}


def check_label_distribution(
    splits: ClassificationSplits, threshold: float = DISTRIBUTION_WARN_THRESHOLD
) -> CheckResult:
    train_p = label_proportions(splits.train)
    gaps: dict[str, float] = {}
    for name, examples in (("dev", splits.dev), ("test", splits.test)):
        p = label_proportions(examples)
        labels = set(train_p) | set(p)
        gaps[name] = round(max(abs(train_p.get(l, 0.0) - p.get(l, 0.0)) for l in labels), 4)
    worst = max(gaps.values())
    status = "warn" if worst > threshold else "pass"
    return CheckResult("label_distribution", status,
                       f"max label-proportion gap vs train: dev={gaps['dev']:.4f}, test={gaps['test']:.4f}"
                       f" (warn above {threshold})", {"max_gap": gaps})


def check_text_overlap(splits: ClassificationSplits, long_tokens: int = LONG_TEXT_TOKENS) -> CheckResult:
    train_texts = {normalize_text(ex.text) for ex in splits.train}
    data: dict[str, Any] = {}
    long_total = 0
    for name, examples in (("dev", splits.dev), ("test", splits.test)):
        overlaps = [ex for ex in examples if normalize_text(ex.text) in train_texts]
        long = [ex for ex in overlaps if len(ex.text.split()) >= long_tokens]
        long_total += len(long)
        data[name] = {
            "overlapping_examples": len(overlaps),
            "overlap_rate": round(len(overlaps) / max(len(examples), 1), 4),
            "long_overlaps": len(long),
            "long_overlap_samples": [ex.text[:120] for ex in long[:5]],
        }
    status = "warn" if long_total else "pass"
    detail = ", ".join(
        f"{n}: {d['overlapping_examples']} ({d['overlap_rate']:.2%}) verbatim in train, "
        f"{d['long_overlaps']} with >= {long_tokens} tokens"
        for n, d in data.items()
    )
    return CheckResult("text_overlap", status, detail, data)


def run_leakage_checks(splits: ClassificationSplits, manifest: dict[str, str]) -> LeakageReport:
    return LeakageReport([
        check_conversation_disjoint(splits),
        check_example_disjoint(splits),
        check_manifest_consistency(splits, manifest),
        check_label_coverage(splits),
        check_label_distribution(splits),
        check_text_overlap(splits),
    ])
