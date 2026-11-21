
"""
Issue #17 - Topic Clustering Baseline
Market Research 1A - Discussion Intelligence Toolkit

Method: TF-IDF + MiniBatch K-Means
Evaluation: Cosine Silhouette Score
Author: Franceli Gonzalez
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA = Path("data/processed/coarse_discourse/train.jsonl")
OUTPUT = Path("artifacts/topic_clustering")

MAX_RECORDS = 12000
SEED = 42
CLUSTER_COUNTS = (8, 12, 16)


# --------------------------------------------------
# Step 1: Load and sample the dataset
# --------------------------------------------------

def sample_records(path, count, seed):
    """Reservoir-sample valid utterances from JSONL."""
    rng = random.Random(seed)
    sample = []
    eligible = 0

    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            text = record.get("text")

            if not isinstance(text, str):
                continue

            if len(text.strip()) < 25:
                continue

            eligible += 1

            if len(sample) < count:
                sample.append(record)
            else:
                index = rng.randrange(eligible)

                if index < count:
                    sample[index] = record

    print(f"Eligible utterances: {eligible:,}")
    print(f"Sampled utterances: {len(sample):,}")

    return sample


# --------------------------------------------------
# Step 2: Generate TF-IDF features
# --------------------------------------------------

def generate_features(texts):
    print("\nGenerating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.85,
        max_features=12000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(texts)

    print(f"Feature matrix: {X.shape}")

    return X, vectorizer


# --------------------------------------------------
# Step 3: Compare clustering configurations
# --------------------------------------------------

def evaluate_models(X):
    print("\nComparing clustering configurations...")

    rng = np.random.default_rng(SEED)

    sample_indices = rng.choice(
        X.shape[0],
        size=min(1000, X.shape[0]),
        replace=False,
    )

    comparisons = []

    best_score = float("-inf")
    best_model = None
    best_assignments = None
    best_k = None

    for k in CLUSTER_COUNTS:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=3,
            batch_size=1024,
        )

        assignments = model.fit_predict(X)
        sample_labels = assignments[sample_indices]

        if len(set(sample_labels)) < 2:
            print(
                f"K={k}: insufficient clusters "
                "for silhouette evaluation"
            )
            continue

        score = silhouette_score(
            X[sample_indices],
            sample_labels,
            metric="cosine",
        )

        print(f"K={k}: silhouette={score:.4f}")

        comparisons.append({
            "k": k,
            "silhouette_score": round(float(score), 5),
        })

        if score > best_score:
            best_score = score
            best_model = model
            best_assignments = assignments
            best_k = k

    if best_model is None:
        raise RuntimeError(
            "No valid clustering configuration found."
        )

    print(f"\nBest configuration: K={best_k}")
    print(f"Best silhouette score: {best_score:.4f}")

    return (
        best_model,
        best_assignments,
        best_k,
        comparisons,
    )


# --------------------------------------------------
# Step 4: Extract topic keywords
# --------------------------------------------------

def export_topic_summary(
    model,
    assignments,
    records,
    X,
    vectorizer,
    k,
):
    vocabulary = vectorizer.get_feature_names_out()

    output_file = OUTPUT / "topic_summary.csv"

    print("\nDiscovered topic keywords:")

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "topic_id",
            "utterance_count",
            "top_terms",
            "example_utterance_id",
            "example_text",
        ])

        for topic_id in range(k):
            indices = np.flatnonzero(
                assignments == topic_id
            )

            center = model.cluster_centers_[topic_id]

            top_indices = np.argsort(center)[-10:][::-1]

            keywords = ", ".join(
                vocabulary[top_indices]
            )

            if len(indices) > 0:
                similarities = X[indices].dot(center)

                nearest_index = int(
                    indices[np.argmax(similarities)]
                )

                example = records[nearest_index]

                example_id = example["id"]

                example_text = (
                    example["text"][:300]
                    .replace("\n", " ")
                )

            else:
                example_id = ""
                example_text = ""

            writer.writerow([
                topic_id,
                len(indices),
                keywords,
                example_id,
                example_text,
            ])

            print(
                f"Topic {topic_id}: "
                f"{len(indices)} utterances | "
                f"{keywords}"
            )


# --------------------------------------------------
# Step 5: Export utterance assignments
# --------------------------------------------------

def export_assignments(records, assignments):
    output_file = OUTPUT / "utterance_topics.csv"

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "utterance_id",
            "conversation_id",
            "subreddit",
            "topic_id",
        ])

        for record, topic_id in zip(
            records,
            assignments,
        ):

            metadata = record.get("metadata") or {}

            writer.writerow([
                record["id"],
                metadata.get("conversation_id", ""),
                metadata.get("subreddit", ""),
                int(topic_id),
            ])


# --------------------------------------------------
# Step 6: Export model comparisons
# --------------------------------------------------

def export_comparisons(comparisons):
    output_file = OUTPUT / "model_comparison.csv"

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "k",
                "silhouette_score",
            ],
        )

        writer.writeheader()
        writer.writerows(comparisons)


# --------------------------------------------------
# Step 7: Run the complete pipeline
# --------------------------------------------------

def main():

    print("=" * 55)
    print("ISSUE #17 - TOPIC CLUSTERING BASELINE")
    print("=" * 55)

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = sample_records(
        DATA,
        MAX_RECORDS,
        SEED,
    )

    texts = [
        record["text"]
        for record in records
    ]

    X, vectorizer = generate_features(texts)

    (
        best_model,
        best_assignments,
        best_k,
        comparisons,
    ) = evaluate_models(X)

    export_comparisons(comparisons)

    export_topic_summary(
        best_model,
        best_assignments,
        records,
        X,
        vectorizer,
        best_k,
    )

    export_assignments(
        records,
        best_assignments,
    )

    config = {
        "source": str(DATA),
        "sample_size": len(records),
        "seed": SEED,
        "cluster_counts": list(CLUSTER_COUNTS),
        "selected_k": best_k,
        "silhouette_sample_size": min(1000, len(records)),
        "method": "TF-IDF + MiniBatchKMeans",
    }

    with (
        OUTPUT / "run_config.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            config,
            file,
            indent=2,
        )

    print("\n" + "=" * 55)
    print("TOPIC CLUSTERING COMPLETE")
    print("=" * 55)

    print(f"Results saved to: {OUTPUT}")

    print("\nGenerated files:")
    print("- model_comparison.csv")
    print("- topic_summary.csv")
    print("- utterance_topics.csv")
    print("- run_config.json")


if __name__ == "__main__":
    main()
