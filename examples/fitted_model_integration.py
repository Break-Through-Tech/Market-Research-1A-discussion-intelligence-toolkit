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

from discussion_intelligence import analyze
from discussion_intelligence.bundles import load_trusted_bundle
from discussion_intelligence.sample import load_reference_conversations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Trusted fitted joblib bundle")
    parser.add_argument("--input", type=Path, required=True, help="Reference JSONL")
    args = parser.parse_args()
    report = analyze(
        load_reference_conversations(args.input)[0],
        **load_trusted_bundle(args.bundle),
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
