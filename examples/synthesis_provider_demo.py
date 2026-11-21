"""Deterministic mock-provider integration demo. No LLM or external service runs."""

import argparse
from pathlib import Path

from discussion_intelligence import analyze
from discussion_intelligence.sample import load_reference_conversations

ROOT = Path(__file__).resolve().parents[1]


class MockQuoteProvider:
    """Only receives retrieved evidence; never creates a model or loads one."""

    def __init__(self, mode="quote"):
        self.mode = mode

    def synthesize(self, evidence, query):
        if self.mode == "error":
            raise RuntimeError("Deliberate mock-provider failure")
        text = (
            "All participants agree that this is the best product."
            if self.mode == "fabricated"
            else evidence[0].text_clean
        )
        return {"points": [{"text": text, "evidence_ids": [evidence[0].id]}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/coarse-discourse-reference-sample.jsonl"
    )
    parser.add_argument("--mode", choices=["quote", "fabricated", "error"], default="quote")
    parser.add_argument("--query", default="RAID data drives")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()
    try:
        conversations = load_reference_conversations(args.input)
        if not conversations:
            raise ValueError("Demo requires a reference conversation")
        report = analyze(
            conversations[0],
            query=args.query,
            top_k=1,
            synthesis_provider=MockQuoteProvider(args.mode),
        )
        report.limitations.append("Deterministic mock-provider demonstration; no LLM was executed.")
        print(report.to_json() if args.format == "json" else report.to_markdown())
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
