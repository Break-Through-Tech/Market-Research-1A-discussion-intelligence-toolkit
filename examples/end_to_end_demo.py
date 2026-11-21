"""Analyze both real reference threads offline; optional trusted owner models."""

import argparse
import tempfile
from pathlib import Path

from discussion_intelligence import analyze
from discussion_intelligence.bundles import load_trusted_bundle
from discussion_intelligence.sample import load_reference_conversations

ROOT = Path(__file__).resolve().parents[1]
# Curated demonstration questions, not generated themes or relevance judgments.
QUERIES = {
    "reddit:t3_10r2f8": "RAID data drives",
    "reddit:t3_169c79": "comment karma easier",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/coarse-discourse-reference-sample.jsonl"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="New directory; never overwrites existing files"
    )
    parser.add_argument(
        "--trusted-bundle",
        type=Path,
        help="Optional fitted owner objects. Pickle/joblib loading can execute code; trusted files only.",
    )
    args = parser.parse_args()
    try:
        if args.output_dir is not None and args.output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {args.output_dir}")
        conversations = load_reference_conversations(args.input)
        if not conversations:
            raise ValueError("Demo requires at least one conversation")
        components = load_trusted_bundle(args.trusted_bundle) if args.trusted_bundle else {}
        # Analyze and validate all results before creating output files.
        reports = [
            analyze(conv, query=QUERIES.get(conv.id), top_k=3, **components)
            for conv in conversations
        ]
        rendered = [(report.to_json(), report.to_markdown()) for report in reports]
        output = args.output_dir
        if output is None:
            output = Path(tempfile.mkdtemp(prefix="discussion-intelligence-demo-"))
        else:
            output.mkdir(parents=True, exist_ok=False)
        for index, (report, (json_text, markdown)) in enumerate(
            zip(reports, rendered, strict=True)
        ):
            (output / f"thread-{index}.json").write_text(json_text + "\n", encoding="utf-8")
            (output / f"thread-{index}.md").write_text(markdown + "\n", encoding="utf-8")
            first = report.summary[0].evidence_ids if report.summary else []
            print(
                f"{report.conversation_id}: {len(report.evidence)} source utterances; first citation {first}"
            )
            print(
                "  "
                + "; ".join(f"{name}={status.state}" for name, status in report.components.items())
            )
        print(f"Reports: {output.resolve()}")
        print(
            "Source excerpts are participant statements. Model status is availability, not accuracy."
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
