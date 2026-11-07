"""Small offline CLI: reference JSONL or canonical Conversation JSON input."""

import argparse
from pathlib import Path

from artifacts.schema import Conversation

from .pipeline import analyze
from .sample import load_reference_conversations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Reference JSONL path")
    parser.add_argument("--canonical", action="store_true", help="Read one canonical JSON object")
    parser.add_argument("--conversation-index", type=int, default=0, help="Zero-based JSONL record")
    parser.add_argument(
        "--query", help="Retrieve excerpts for this query; otherwise use centroid ranking"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path, help="Output file; default is stdout")
    args = parser.parse_args(argv)
    try:
        if args.conversation_index < 0:
            raise ValueError("conversation-index must be nonnegative")
        conversations = (
            [Conversation.model_validate_json(args.input.read_text(encoding="utf-8"))]
            if args.canonical
            else load_reference_conversations(args.input)
        )
        if args.conversation_index >= len(conversations):
            raise ValueError(
                f"conversation-index outside input containing {len(conversations)} conversations"
            )
        report = analyze(conversations[args.conversation_index], query=args.query, top_k=args.top_k)
        result = report.to_json() if args.format == "json" else report.to_markdown()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result + "\n", encoding="utf-8")
        else:
            print(result)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0
