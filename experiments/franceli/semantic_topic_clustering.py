
"""
Issue #17 - Semantic Topic Clustering
Market Research 1A - Discussion Intelligence Toolkit

Method: Sentence Transformers + MiniBatch K-Means
Evaluation: Cosine Silhouette Score
Author: Franceli Gonzalez
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA = Path("data/processed/coarse_discourse/train.jsonl")
OUTPUT = Path("artifacts/topic_clustering/semantic")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_RECORDS = 12000
SEED = 42
CLUSTER_COUNTS = (8, 12, 16)
BATCH_SIZE = 32
SILHOUETTE_SAMPLE = 1000


# --------------------------------------------------
# Step 1: Sample the same dataset as the baseline
# --------------------------------------------------

def sample_records(path, count, seed):
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
# Step 2: Generate semantic embeddings
# --------------------------------------------------

def generate_embeddings(texts):
    print("\nLoading sentence-transformer model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("\nGenerating semantic embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Embedding matrix: {embeddings.shape}")

    return embeddings


# --------------------------------------------------
# Step 3: Compare clustering configurations
# --------------------------------------------------

def evaluate_models(embeddings):
    print("\nComparing semantic clustering models...")

    rng = np.random.default_rng(SEED)

    sample_indices = rng.choice(
        len(embeddings),
        size=min(SILHOUETTE_SAMPLE, len(embeddings)),
        replace=False,
    )

    comparisons = []

    best_score = float("-inf")
    best_model = None
    best_assignments = None
    best_k = None

    for k in CLUSTER_COUNTS:
        print(f"\nTraining MiniBatch K-Means with K={k}...")

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=3,
            batch_size=1024,
        )

        assignments = model.fit_predict(embeddings)

        sample_labels = assignments[sample_indices]

        if len(set(sample_labels)) < 2:
            print(f"K={k}: insufficient clusters")
            continue

        score = silhouette_score(
            embeddings[sample_indices],
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
# Step 4: Extract interpretable topic keywords
# --------------------------------------------------

def export_topic_summary(
    records,
    embeddings,
    assignments,
    model,
    k,
):
    print("\nExtracting topic keywords...")

    texts = [record["text"] for record in records]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.85,
        max_features=12000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tfidf = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()

    output_file = OUTPUT / "topic_summary.csv"

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

        print("\nDiscovered semantic topics:")

        for topic_id in range(k):
            indices = np.flatnonzero(
                assignments == topic_id
            )

            if len(indices) == 0:
                continue

            # Average TF-IDF importance within the cluster
            average_terms = np.asarray(
                tfidf[indices].mean(axis=0)
            ).ravel()

            top_indices = np.argsort(
                average_terms
            )[-10:][::-1]

            keywords = ", ".join(
                vocabulary[top_indices]
            )

            # Choose representative utterance nearest
            # to its cluster center in embedding space
            center = model.cluster_centers_[topic_id]

            similarities = embeddings[indices] @ center

            nearest_index = int(
                indices[np.argmax(similarities)]
            )

            example = records[nearest_index]

            example_text = (
                example["text"][:300]
                .replace("\n", " ")
            )

            writer.writerow([
                topic_id,
                len(indices),
                keywords,
                example["id"],
                example_text,
            ])

            print(
                f"Topic {topic_id}: "
                f"{len(indices)} utterances | {keywords}"
            )


# --------------------------------------------------
# Step 5: Export assignments
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
# Step 6: Export model comparison
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
            fieldnames=["k", "silhouette_score"],
        )

        writer.writeheader()
        writer.writerows(comparisons)


# --------------------------------------------------
# Step 7: Run the pipeline
# --------------------------------------------------

def main():
    print("=" * 55)
    print("ISSUE #17 - SEMANTIC TOPIC CLUSTERING")
    print("=" * 55)

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    records = sample_records(
        DATA,
        MAX_RECORDS,
        SEED,
    )

    texts = [record["text"] for record in records]

    embeddings = generate_embeddings(texts)

    (
        best_model,
        best_assignments,
        best_k,
        comparisons,
    ) = evaluate_models(embeddings)

    export_comparisons(comparisons)

    export_topic_summary(
        records,
        embeddings,
        best_assignments,
        best_model,
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
        "embedding_model": MODEL_NAME,
        "embedding_dimensions": int(embeddings.shape[1]),
        "cluster_counts": list(CLUSTER_COUNTS),
        "selected_k": best_k,
        "silhouette_sample_size": min(
            SILHOUETTE_SAMPLE, len(records)
        ),
        "method": "Sentence Transformers + MiniBatchKMeans",
    }

    with (
        OUTPUT / "run_config.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("\n" + "=" * 55)
    print("SEMANTIC CLUSTERING COMPLETE")
    print("=" * 55)

    print(f"Results saved to: {OUTPUT}")

    print("\nGenerated files:")
    print("- model_comparison.csv")
    print("- topic_summary.csv")
    print("- utterance_topics.csv")
    print("- run_config.json")


if __name__ == "__main__":
    main()
