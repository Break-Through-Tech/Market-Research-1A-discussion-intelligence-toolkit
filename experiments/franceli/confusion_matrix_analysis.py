
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

from utils.modeling_data.build import load_prepared_splits


# STEP 1: Load standardized dataset
print("Loading dataset...", flush=True)

splits = load_prepared_splits()

X_train = [example.text for example in splits.train]
y_train = np.array([example.label for example in splits.train])

X_dev = [example.text for example in splits.dev]
y_dev = np.array([example.label for example in splits.dev])

classes = np.unique(y_train)

print("Training examples:", len(X_train))
print("Development examples:", len(X_dev))


# STEP 2: Calculate class weights
balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)


# STEP 3: Prepare results directory
output_dir = Path("experiments/franceli/results")
output_dir.mkdir(parents=True, exist_ok=True)


# STEP 4: Train and evaluate both models
for strength in [0.25, 0.50]:

    percent = int(strength * 100)

    class_weights = {
        label: 1.0 + strength * (float(weight) - 1.0)
        for label, weight in zip(classes, balanced_weights)
    }

    print(
        f"\nTraining {percent}% balanced model...",
        flush=True
    )

    # Convert text into TF-IDF numerical features
    vectorizer = TfidfVectorizer(max_features=10000)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_dev_tfidf = vectorizer.transform(X_dev)

    # Train Logistic Regression classifier
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weights,
    )

    model.fit(X_train_tfidf, y_train)

    print("Training complete!", flush=True)

    # Generate predictions
    predictions = model.predict(X_dev_tfidf)

    # STEP 5: Calculate evaluation metrics
    accuracy = accuracy_score(y_dev, predictions)

    macro_f1 = f1_score(
        y_dev,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_dev,
        predictions,
        average="weighted"
    )

    print(f"\nResults - {percent}% Balanced")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")


    # STEP 6: Generate classification report
    report = classification_report(
        y_dev,
        predictions,
        labels=classes,
        zero_division=0,
    )

    print(f"\nClassification Report - {percent}% Balanced")
    print(report)

    report_path = (
        output_dir / f"classification_report_{percent}.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8"
    )


    # STEP 7: Identify the most common prediction errors
    errors = Counter(
        (actual, predicted)
        for actual, predicted in zip(y_dev, predictions)
        if actual != predicted
    )

    print(f"\nTop 10 Classification Errors - {percent}% Balanced")

    error_lines = []

    for (actual, predicted), count in errors.most_common(10):
        line = f"{actual} -> {predicted}: {count}"
        print(line)
        error_lines.append(line)

    # Save error analysis
    error_path = (
        output_dir / f"error_analysis_{percent}.txt"
    )

    error_path.write_text(
        "\n".join(error_lines) + "\n",
        encoding="utf-8"
    )


    # STEP 8: Generate normalized confusion matrix
    cm = confusion_matrix(
        y_dev,
        predictions,
        labels=classes,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=45,
        colorbar=True,
    )

    ax.set_title(
        f"Discourse Classifier Confusion Matrix - {percent}% Balanced",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    output_path = (
        output_dir / f"confusion_matrix_{percent}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved confusion matrix:", output_path)
    print("Saved classification report:", report_path)
    print("Saved error analysis:", error_path)


print("\nAll experiments completed successfully!")
