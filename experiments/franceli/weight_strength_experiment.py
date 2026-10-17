
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from utils.modeling_data.build import load_prepared_splits
from artifacts.evaluation import evaluate_classifier


# STEP 1: Load standardized datasets
print("Loading dataset...")

splits = load_prepared_splits()

X_train = [example.text for example in splits.train]
y_train = [example.label for example in splits.train]

X_dev = [example.text for example in splits.dev]
y_dev = [example.label for example in splits.dev]

print("Training examples:", len(X_train))
print("Development examples:", len(X_dev))


# STEP 2: Calculate balanced class weights
classes = np.unique(y_train)

balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.array(y_train)
)

print("\nBalanced class weights:")

for label, weight in zip(classes, balanced_weights):
    print(f"{label}: {weight:.4f}")


# STEP 3: Define balancing strengths
strengths = [0.0, 0.25, 0.5, 1.0]

experiment_results = []


# STEP 4: Train and evaluate each model
for strength in strengths:

    class_weights = {
        label: 1.0 + strength * (float(weight) - 1.0)
        for label, weight in zip(classes, balanced_weights)
    }

    model = make_pipeline(
        TfidfVectorizer(max_features=10000),
        LogisticRegression(
            max_iter=1000,
            class_weight=class_weights
        )
    )

    print(
        f"\nTraining model with balancing strength {strength:.0%}...",
        flush=True
    )

    model.fit(X_train, y_train)

    print("Training complete!", flush=True)

    results = evaluate_classifier(
        model,
        texts=X_dev,
        labels=y_dev
    )

    accuracy = results.accuracy
    macro_f1 = results.macro_f1
    weighted_f1 = results.weighted_f1

    experiment_results.append({
        "strength": strength,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")


# STEP 5: Compare all model configurations
print("\n========== FINAL COMPARISON ==========")

print(
    f"{'Strength':<12}"
    f"{'Accuracy':<12}"
    f"{'Macro F1':<12}"
    f"{'Weighted F1':<12}"
)

for result in experiment_results:
    print(
        f"{result['strength']:.0%}".ljust(12)
        + f"{result['accuracy']:.4f}".ljust(12)
        + f"{result['macro_f1']:.4f}".ljust(12)
        + f"{result['weighted_f1']:.4f}".ljust(12)
    )


# STEP 6: Identify best model by macro F1
best_model = max(
    experiment_results,
    key=lambda result: result["macro_f1"]
)

print("\nBest configuration based on Macro F1:")
print(f"Balancing strength: {best_model['strength']:.0%}")
print(f"Accuracy: {best_model['accuracy']:.4f}")
print(f"Macro F1: {best_model['macro_f1']:.4f}")
print(f"Weighted F1: {best_model['weighted_f1']:.4f}")

print("\nExperiment complete!")
