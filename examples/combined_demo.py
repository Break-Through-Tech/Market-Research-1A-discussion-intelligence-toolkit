"""Run the integrated toolkit with trained topic clustering and optional Gemini."""

import argparse
import json
import tempfile
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

from artifacts.schema import Conversation
from discussion_intelligence import analyze
from discussion_intelligence.llm_synthesis import create_gemini_provider
from discussion_intelligence.sample import load_reference_conversations
from discussion_intelligence.topic_clustering import SemanticTopicClusterer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/coarse-discourse-reference-sample.jsonl",
    )
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--gemini", action="store_true")
    args = parser.parse_args()

    folder = ROOT / "artifacts/topic_clustering/semantic/models"
    metadata = json.loads((folder / "model_metadata.json").read_text())

    # Load the trusted model contributed by Franceli.
    model = joblib.load(folder / "semantic_kmeans.joblib")
    encoder = SentenceTransformer(metadata["embedding_model"], device="cpu")

    if model.n_clusters != metadata["selected_k"]:
        raise ValueError("Cluster count mismatch")

    if model.n_features_in_ != metadata["embedding_dimensions"]:
        raise ValueError("Embedding dimension mismatch")

    clusterer = SemanticTopicClusterer(
        encoder,
        model,
        normalize_embeddings=metadata["normalize_embeddings"],
    )

    provider = create_gemini_provider() if args.gemini else None

    conversations = (
        [Conversation.model_validate_json(args.input.read_text())]
        if args.canonical
        else load_reference_conversations(args.input)
    )

    if not conversations:
        raise ValueError("No conversations found")

    output = Path(tempfile.mkdtemp(prefix="combined-discussion-demo-"))

    for index, conversation in enumerate(conversations):
        report = analyze(
            conversation,
            query=args.query,
            clusterer=clusterer,
            synthesis_provider=provider,
        )

        (output / f"thread-{index}.json").write_text(report.to_json(), encoding="utf-8")
        (output / f"thread-{index}.md").write_text(report.to_markdown(), encoding="utf-8")

        print(f"\nThread: {conversation.id}")
        print("Topic assignments:", report.topic_assignments)

        for name, status in report.components.items():
            print(f"  {name}: {status.state} - {status.detail}")

        if report.components["clustering"].state != "available":
            raise RuntimeError("Trained clustering did not run successfully")

    print(f"\nReports saved to: {output}")


if __name__ == "__main__":
    main()
