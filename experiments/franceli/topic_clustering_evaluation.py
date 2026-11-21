
"""
Issue #17 - Topic Clustering Evaluation
Market Research 1A - Discussion Intelligence Toolkit

Evaluates:
1. Topic coherence using NPMI
2. TF-IDF vs semantic clustering
3. Generic and URL-heavy topics
4. Sampled utterances for manual error analysis

"""

import csv
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA = Path("data/processed/coarse_discourse/train.jsonl")

BASELINE = Path("artifacts/topic_clustering")
SEMANTIC = BASELINE / "semantic"
OUTPUT = BASELINE / "evaluation"

TOP_N_TERMS = 10
EXAMPLES_PER_TOPIC = 5
SEED = 42

GENERIC_TERMS = {
    "just", "like", "really", "think", "know",
    "good", "don", "ve", "ll", "yeah",
    "yes", "thing", "things", "people",
    "time", "way", "right", "got",
}

URL_TERMS = {
    "http", "https", "www", "com",
    "jpg", "png", "imgur",
    "reddit com", "http www",
}


# --------------------------------------------------
# Step 1: Read CSV files
# --------------------------------------------------

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def load_model_results(name, folder):
    topics = read_csv(folder / "topic_summary.csv")
    assignments = read_csv(folder / "utterance_topics.csv")
    comparisons = read_csv(folder / "model_comparison.csv")

    with (folder / "run_config.json").open(
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    print(f"\nLoaded {name}:")
    print(f"  Topics: {len(topics)}")
    print(f"  Assignments: {len(assignments)}")

    return {
        "name": name,
        "topics": topics,
        "assignments": assignments,
        "comparisons": comparisons,
        "config": config,
    }


# --------------------------------------------------
# Step 2: Load sampled utterance text
# --------------------------------------------------

def load_utterance_text(models):
    wanted_ids = set()

    for model in models:
        wanted_ids.update(
            row["utterance_id"]
            for row in model["assignments"]
        )

    texts = {}

    with DATA.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            utterance_id = record.get("id")

            if utterance_id in wanted_ids:
                texts[utterance_id] = record["text"]

    missing = wanted_ids - texts.keys()

    if missing:
        raise ValueError(
            f"Could not locate {len(missing)} utterances"
        )

    print(f"\nLoaded {len(texts):,} unique utterances")

    return texts


# --------------------------------------------------
# Step 3: Prepare a document-term matrix
# --------------------------------------------------

def build_term_matrix(models, texts):
    vocabulary = set()

    for model in models:
        for topic in model["topics"]:
            terms = topic["top_terms"].split(",")

            for term in terms[:TOP_N_TERMS]:
                term = term.strip().lower()

                if term:
                    vocabulary.add(term)

    vocabulary = sorted(vocabulary)

    ordered_ids = sorted(texts)
    documents = [texts[uid] for uid in ordered_ids]

    vectorizer = CountVectorizer(
        vocabulary=vocabulary,
        lowercase=True,
        binary=True,
        ngram_range=(1, 2),
        dtype=np.int64,
    )

    matrix = vectorizer.transform(documents)

    # Number of documents containing each term
    frequencies = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    # Number of documents containing each pair
    cooccurrence = (
        matrix.T @ matrix
    ).toarray()

    term_indices = vectorizer.vocabulary_

    print(f"Coherence vocabulary: {len(vocabulary)} terms")

    return (
        len(documents),
        term_indices,
        frequencies,
        cooccurrence,
    )


# --------------------------------------------------
# Step 4: Calculate NPMI topic coherence
# --------------------------------------------------

def calculate_npmi(
    terms,
    n_docs,
    indices,
    frequencies,
    cooccurrence,
):
    scores = []

    for term_a, term_b in combinations(terms, 2):
        a = indices.get(term_a)
        b = indices.get(term_b)

        if a is None or b is None:
            continue

        count_a = frequencies[a]
        count_b = frequencies[b]
        count_ab = cooccurrence[a, b]

        if count_a == 0 or count_b == 0:
            scores.append(-1.0)
            continue

        if count_ab == 0:
            scores.append(-1.0)
            continue

        p_a = count_a / n_docs
        p_b = count_b / n_docs
        p_ab = count_ab / n_docs

        pmi = np.log(p_ab / (p_a * p_b))
        npmi = pmi / (-np.log(p_ab))

        # Identical document-presence patterns can
        # make p_ab = 1, where NPMI is undefined.
        if not np.isfinite(npmi):
            npmi = 1.0 if p_ab == 1 else -1.0

        scores.append(float(npmi))

    if not scores:
        return None

    return float(np.mean(scores))


# --------------------------------------------------
# Step 5: Analyze topic quality
# --------------------------------------------------

def evaluate_topics(models, texts, term_data):
    n_docs, indices, frequencies, cooccurrence = term_data

    results = []

    for model in models:
        name = model["name"]

        for topic in model["topics"]:
            topic_id = int(topic["topic_id"])

            terms = [
                term.strip().lower()
                for term in topic["top_terms"].split(",")
                if term.strip()
            ][:TOP_N_TERMS]

            score = calculate_npmi(
                terms,
                n_docs,
                indices,
                frequencies,
                cooccurrence,
            )

            generic_count = sum(
                term in GENERIC_TERMS
                for term in terms
            )

            url_count = sum(
                term in URL_TERMS
                or bool(
                    re.search(
                        r"https?://|www\.|\.com",
                        term,
                    )
                )
                for term in terms
            )

            flags = []

            if generic_count >= 4:
                flags.append("generic_keywords")

            if url_count >= 3:
                flags.append("url_heavy")

            if score is not None and score < 0:
                flags.append("negative_npmi")

            results.append({
                "model": name,
                "topic_id": topic_id,
                "utterance_count": int(
                    topic["utterance_count"]
                ),
                "npmi_coherence": (
                    round(score, 5)
                    if score is not None
                    else ""
                ),
                "generic_term_count": generic_count,
                "url_term_count": url_count,
                "review_flags": "; ".join(flags),
                "top_terms": ", ".join(terms),
            })

    return results


# --------------------------------------------------
# Step 6: Export topic evaluations
# --------------------------------------------------

def export_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# Step 7: Sample examples for manual review
# --------------------------------------------------

def export_review_samples(models, texts):
    rng = np.random.default_rng(SEED)
    review_rows = []

    for model in models:
        groups = {}

        for assignment in model["assignments"]:
            topic_id = int(assignment["topic_id"])

            groups.setdefault(topic_id, []).append(
                assignment
            )

        for topic_id, assignments in sorted(groups.items()):
            count = min(
                EXAMPLES_PER_TOPIC,
                len(assignments),
            )

            selected = rng.choice(
                len(assignments),
                size=count,
                replace=False,
            )

            for index in selected:
                assignment = assignments[int(index)]
                utterance_id = assignment["utterance_id"]

                review_rows.append({
                    "model": model["name"],
                    "topic_id": topic_id,
                    "utterance_id": utterance_id,
                    "conversation_id": assignment[
                        "conversation_id"
                    ],
                    "text": texts[utterance_id][:500],
                    "fits_topic": "",
                    "suggested_topic": "",
                    "error_notes": "",
                })

    export_csv(
        OUTPUT / "manual_review_samples.csv",
        review_rows,
    )

    print(
        f"Exported {len(review_rows)} "
        "manual review examples"
    )


# --------------------------------------------------
# Step 8: Compare models
# --------------------------------------------------

def compare_models(models, topic_results):
    comparison = []

    for model in models:
        name = model["name"]

        results = [
            row for row in topic_results
            if row["model"] == name
        ]

        scores = [
            float(row["npmi_coherence"])
            for row in results
            if row["npmi_coherence"] != ""
        ]

        silhouette_results = [
            float(row["silhouette_score"])
            for row in model["comparisons"]
        ]

        flagged = sum(
            bool(row["review_flags"])
            for row in results
        )

        comparison.append({
            "model": name,
            "selected_k": model["config"]["selected_k"],
            "sample_size": len(model["assignments"]),
            "best_silhouette": round(
                max(silhouette_results), 5
            ),
            "mean_npmi": (
                round(float(np.mean(scores)), 5)
                if scores else ""
            ),
            "flagged_topics": flagged,
            "total_topics": len(results),
        })

    return comparison


# --------------------------------------------------
# Step 9: Run evaluation
# --------------------------------------------------

def main():
    print("=" * 55)
    print("ISSUE #17 - TOPIC CLUSTERING EVALUATION")
    print("=" * 55)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    models = [
        load_model_results("TF-IDF", BASELINE),
        load_model_results("Semantic", SEMANTIC),
    ]

    texts = load_utterance_text(models)

    term_data = build_term_matrix(models, texts)

    topic_results = evaluate_topics(
        models,
        texts,
        term_data,
    )

    export_csv(
        OUTPUT / "topic_coherence.csv",
        topic_results,
    )

    export_review_samples(models, texts)

    comparisons = compare_models(
        models,
        topic_results,
    )

    export_csv(
        OUTPUT / "evaluation_comparison.csv",
        comparisons,
    )

    print("\nModel comparison:")
    for row in comparisons:
        print(
            f"\n{row['model']}:"
            f"\n  Silhouette: {row['best_silhouette']}"
            f"\n  Mean NPMI: {row['mean_npmi']}"
            f"\n  Flagged topics: "
            f"{row['flagged_topics']}/{row['total_topics']}"
        )

    print("\nEvaluation complete!")
    print(f"Results saved to: {OUTPUT}")

    print("\nGenerated files:")
    print("- topic_coherence.csv")
    print("- manual_review_samples.csv")
    print("- evaluation_comparison.csv")

    print(
        "\nNote: Review flags are heuristic. "
        "Manual review is still required."
    )


if __name__ == "__main__":
    main()
