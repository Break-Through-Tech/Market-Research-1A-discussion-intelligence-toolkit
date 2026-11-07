"""Connect trusted, already fitted artifacts; never import training experiments.

Run from the repository root with --bundle pointing to a trusted joblib mapping
exported by the model owner. Supported keys: discourse_model (text-only), OR
context_model + target_vectorizer + parent_vectorizer; civility_model;
topic_clusterer (assign protocol); embedder (embed protocol).

Only load trusted artifacts: joblib/pickle loading can execute Python code.
This file does not include trained artifacts or claim the team's model is shipped.
"""

import argparse
from pathlib import Path

import joblib

from discussion_intelligence import analyze
from discussion_intelligence.adapters import (
    ContextClassifierAdapter,
    ConversationCivilityAdapter,
    TextClassifierAdapter,
)
from discussion_intelligence.sample import load_reference_conversations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Trusted fitted joblib bundle")
    parser.add_argument("--input", type=Path, required=True, help="Reference JSONL")
    args = parser.parse_args()
    bundle = joblib.load(args.bundle)
    classifier = None
    if "context_model" in bundle:
        classifier = ContextClassifierAdapter(
            bundle["context_model"], bundle["target_vectorizer"], bundle["parent_vectorizer"]
        )
    elif "discourse_model" in bundle:
        classifier = TextClassifierAdapter(bundle["discourse_model"])
    quality = None
    if "civility_model" in bundle:
        quality = ConversationCivilityAdapter(
            bundle["civility_model"], model_name=bundle.get("civility_model_name", "owner-supplied")
        )
    report = analyze(
        load_reference_conversations(args.input)[0],
        classifier=classifier,
        quality_scorer=quality,
        clusterer=bundle.get("topic_clusterer"),
        embedder=bundle.get("embedder"),
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
