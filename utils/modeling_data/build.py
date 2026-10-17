"""
Build the Coarse Discourse modeling datasets end to end.

    uv run --no-project --with pydantic python -m utils.modeling_data.build \\
        --corpus-dir ~/.convokit/saved-corpora/reddit-coarse-discourse-corpus

Outputs (under --out-dir, default data/processed/coarse_discourse/, gitignored):

    conversations.jsonl    canonical Conversation objects, one per line
    train.jsonl            ClassificationExample rows (load_jsonl_classification)
    dev.jsonl
    test.jsonl
    split_manifest.csv     conversation_id, subreddit, split
    dataset_card.json      config, input hashes, counts, exclusions, leakage report

With --publish-dir (default data/splits/), the manifest and dataset card are also
copied to a small tracked location so the team can review and reuse the exact
splits without sharing the corpus itself.

Exit code is 1 if any leakage check fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifacts.corpora import ClassificationSplits, load_jsonl_classification
from artifacts.schema import SCHEMA_VERSION
from utils.modeling_data.convokit_reader import (
    CORPUS_NAME,
    default_corpus_dir,
    file_sha256,
    group_by_conversation,
    iter_raw_utterances,
    load_conversation_meta,
)
from utils.modeling_data.examples import iter_examples
from utils.modeling_data.labels import (
    CANONICAL_LABELS,
    LABEL_MAP,
    LABEL_MAP_VERSION,
    MODELING_DROP_LABELS,
)
from utils.modeling_data.leakage import LeakageReport, run_leakage_checks
from utils.modeling_data.normalize import NormalizationStats, normalize_conversation
from utils.modeling_data.splits import SPLITS, assign_split, read_manifest, write_manifest

log = logging.getLogger(__name__)

DATASET_NAME = "coarse_discourse"
DATASET_VERSION = "1.0"
DEFAULT_OUT_DIR = Path("data/processed") / DATASET_NAME
DEFAULT_PUBLISH_DIR = Path("data/splits")


@dataclass
class BuildConfig:
    corpus_dir: Path
    out_dir: Path = DEFAULT_OUT_DIR
    publish_dir: Path | None = DEFAULT_PUBLISH_DIR
    seed: int = 42
    dev_frac: float = 0.1
    test_frac: float = 0.1
    subreddits: frozenset[str] | None = None
    pseudonymize_speakers: bool = True


@dataclass
class BuildResult:
    splits: ClassificationSplits
    manifest: dict[str, str]
    report: LeakageReport
    card: dict[str, Any]


def _label_counts(examples: list) -> dict[str, int]:
    return dict(sorted(Counter(ex.label for ex in examples).items()))


def build(config: BuildConfig) -> BuildResult:
    corpus_dir = Path(config.corpus_dir).expanduser()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conv_meta = load_conversation_meta(corpus_dir)
    grouped = group_by_conversation(iter_raw_utterances(corpus_dir))
    log.info("Read %d conversations from %s", len(grouped), corpus_dir)

    stats = NormalizationStats()
    exclusions: Counter[str] = Counter()
    by_split: dict[str, list] = {s: [] for s in SPLITS}
    manifest_rows: list[dict[str, str]] = []
    filtered_out = 0

    with open(out_dir / "conversations.jsonl", "w", encoding="utf-8") as conv_file:
        for conv_id in sorted(grouped):
            meta = conv_meta.get(conv_id, {})
            if config.subreddits is not None and meta.get("subreddit") not in config.subreddits:
                filtered_out += 1
                continue
            conv = normalize_conversation(
                conv_id, grouped[conv_id], meta,
                pseudonymize_speakers=config.pseudonymize_speakers, stats=stats,
            )
            if conv is None:
                continue
            split = assign_split(conv_id, config.seed, config.dev_frac, config.test_frac)
            conv.metadata["split"] = split
            conv_file.write(conv.model_dump_json() + "\n")
            manifest_rows.append(
                {"conversation_id": conv_id, "subreddit": meta.get("subreddit"), "split": split}
            )
            by_split[split].extend(iter_examples(conv, exclusions))

    for split, examples in by_split.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(ex.model_dump_json() + "\n")

    manifest_path = out_dir / "split_manifest.csv"
    write_manifest(manifest_rows, manifest_path)
    manifest = read_manifest(manifest_path)

    splits = ClassificationSplits(train=by_split["train"], dev=by_split["dev"], test=by_split["test"])
    report = run_leakage_checks(splits, manifest)

    conv_per_split = Counter(manifest.values())
    n_conv = sum(conv_per_split.values()) or 1
    card: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "task": "utterance-level discourse-act classification",
        "source": {
            "corpus": CORPUS_NAME,
            "url": "https://convokit.cornell.edu/documentation/coarseDiscourse.html",
            "utterances_sha256": file_sha256(corpus_dir / "utterances.jsonl"),
        },
        "config": {
            "seed": config.seed,
            "dev_frac": config.dev_frac,
            "test_frac": config.test_frac,
            "group_key": "conversation_id (raw Reddit root ID)",
            "subreddits": sorted(config.subreddits) if config.subreddits else "all",
            "pseudonymize_speakers": config.pseudonymize_speakers,
            "schema_version": SCHEMA_VERSION,
            "label_map_version": LABEL_MAP_VERSION,
        },
        "label_map": LABEL_MAP,
        "modeling_labels": list(CANONICAL_LABELS),
        "dropped_labels": sorted(MODELING_DROP_LABELS),
        "normalization": {**stats.as_dict(), "conversations_filtered_by_subreddit": filtered_out},
        "exclusions": dict(sorted(exclusions.items())),
        "counts": {
            split: {
                "conversations": conv_per_split.get(split, 0),
                "conversation_share": round(conv_per_split.get(split, 0) / n_conv, 4),
                "examples": len(by_split[split]),
                "labels": _label_counts(by_split[split]),
            }
            for split in SPLITS
        },
        "total_examples": sum(len(v) for v in by_split.values()),
        "leakage": report.as_dict(),
    }
    with open(out_dir / "dataset_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
        f.write("\n")

    if config.publish_dir is not None:
        publish = Path(config.publish_dir)
        publish.mkdir(parents=True, exist_ok=True)
        suffix = DATASET_NAME if config.subreddits is None else f"{DATASET_NAME}_subset"
        shutil.copyfile(manifest_path, publish / f"{suffix}_split_manifest.csv")
        shutil.copyfile(out_dir / "dataset_card.json", publish / f"{suffix}_dataset_card.json")

    return BuildResult(splits=splits, manifest=manifest, report=report, card=card)


def load_prepared_splits(out_dir: str | Path = DEFAULT_OUT_DIR) -> ClassificationSplits:
    """Load previously built train/dev/test files as ClassificationSplits."""
    out_dir = Path(out_dir)
    return ClassificationSplits(
        **{s: load_jsonl_classification(out_dir / f"{s}.jsonl") for s in SPLITS}
    )


def _parse_args(argv: list[str] | None) -> BuildConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", type=Path, default=None,
                   help="Downloaded ConvoKit corpus directory (default: ~/.convokit/... or ./.convokit/...)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--publish-dir", type=Path, default=DEFAULT_PUBLISH_DIR,
                   help="Where to copy the manifest and dataset card for version control")
    p.add_argument("--no-publish", action="store_true", help="Skip copying to --publish-dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--subreddits", default=None,
                   help="Comma-separated subreddit allowlist, e.g. buildapc,techsupport")
    p.add_argument("--keep-usernames", action="store_true",
                   help="Keep raw Reddit handles instead of pseudonymizing them")
    a = p.parse_args(argv)
    return BuildConfig(
        corpus_dir=a.corpus_dir or default_corpus_dir(),
        out_dir=a.out_dir,
        publish_dir=None if a.no_publish else a.publish_dir,
        seed=a.seed,
        dev_frac=a.dev_frac,
        test_frac=a.test_frac,
        subreddits=frozenset(s.strip() for s in a.subreddits.split(",")) if a.subreddits else None,
        pseudonymize_speakers=not a.keep_usernames,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = _parse_args(argv)
    if not (Path(config.corpus_dir).expanduser() / "utterances.jsonl").exists():
        log.error("No utterances.jsonl under %s. Download the corpus first (see data/README.md).",
                  config.corpus_dir)
        return 2

    result = build(config)
    for split in SPLITS:
        c = result.card["counts"][split]
        log.info("%-5s conversations=%5d examples=%6d", split, c["conversations"], c["examples"])
    log.info("Exclusions: %s", result.card["exclusions"])
    log.info("Leakage checks:\n%s", result.report.format())
    log.info("Wrote %s", config.out_dir)
    return 0 if result.report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
