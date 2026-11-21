"""Evaluate the inspected reference threads; NOT a generalization benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

from discussion_intelligence.retrieval_evaluation import RetrievalJudgments, evaluate_retrieval
from discussion_intelligence.sample import load_reference_conversations

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/coarse-discourse-reference-sample.jsonl"
    )
    parser.add_argument(
        "--judgments", type=Path, default=ROOT / "examples/retrieval-judgments.json"
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        judgments = RetrievalJudgments.model_validate_json(
            args.judgments.read_text(encoding="utf-8")
        )
        if hashlib.sha256(args.input.read_bytes()).hexdigest() != judgments.source_sha256:
            raise ValueError("Input SHA-256 differs from the judged corpus; review judgments first")
        result = evaluate_retrieval(load_reference_conversations(args.input), judgments, ks=args.ks)
        if args.format == "json":
            rendered = json.dumps(result, indent=2, allow_nan=False)
        else:
            lines = [
                f"# {result['evaluation']}",
                "",
                result["scope"],
                "",
                result["curation_note"],
                "",
                f"Positive queries: {result['positive_queries']}; no-answer queries: {result['no_answer_queries']}.",
                "",
                "| k | Macro Recall@k (positive queries) | No-answer abstention rate |",
                "|---|---:|---:|",
            ]

            def display(value):
                return "n/a" if value is None else f"{value:.4f}"

            for k, metrics in result["metrics"].items():
                lines.append(
                    f"| {k} | {display(metrics['macro_recall'])} | "
                    f"{display(metrics['no_answer_abstention_rate'])} |"
                )
            lines.extend(
                [
                    "",
                    "Per-query details and original source links are available with `--format json`.",
                ]
            )
            rendered = "\n".join(lines)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
