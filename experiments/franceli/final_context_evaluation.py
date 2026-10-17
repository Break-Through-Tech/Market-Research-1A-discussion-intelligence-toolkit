
from pathlib import Path

import numpy as np
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

from utils.modeling_data.build import load_prepared_splits


# STEP 1: Load training and development data
print("Loading dataset...", flush=True)

splits = load_prepared_splits()

train = splits.train
dev = splits.dev

y_train = np.array([item.label for item in train])
y_dev = np.array([item.label for item in dev])


def get_parent_text(item):
    value = item.metadata.get("parent_text")
    return value.strip() if isinstance(value, str) else ""


train_text = [item.text for item in train]
dev_text = [item.text for item in dev]

train_parent = [get_parent_text(item) for item in train]
dev_parent = [get_parent_text(item) for item in dev]


# STEP 2: Calculate 50% class balancing
classes = np.unique(y_train)

balanced = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

weights = {
    label: 1.0 + 0.5 * (float(weight) - 1.0)
    for label, weight in zip(classes, balanced)
}


# STEP 3: Create separate TF-IDF features
print("Creating TF-IDF features...", flush=True)

target_vectorizer = TfidfVectorizer(max_features=10000)
parent_vectorizer = TfidfVectorizer(max_features=10000)

X_train_target = target_vectorizer.fit_transform(train_text)
X_dev_target = target_vectorizer.transform(dev_text)

X_train_parent = parent_vectorizer.fit_transform(train_parent)
X_dev_parent = parent_vectorizer.transform(dev_parent)

X_train_context = hstack(
    [X_train_target, X_train_parent],
    format="csr",
)

X_dev_context = hstack(
    [X_dev_target, X_dev_parent],
    format="csr",
)


# STEP 4: Train both models
output = Path("experiments/franceli/results")
output.mkdir(parents=True, exist_ok=True)

for name, X_train, X_dev in [
    ("text_only", X_train_target, X_dev_target),
    ("context_aware", X_train_context, X_dev_context),
]:

    print(f"\nTraining {name}...", flush=True)

    model = LogisticRegression(
        max_iter=1000,
        class_weight=weights,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, predictions)
    macro_f1 = f1_score(
        y_dev, predictions, average="macro"
    )
    weighted_f1 = f1_score(
        y_dev, predictions, average="weighted"
    )

    print(f"\n{name.upper()} RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    report = classification_report(
        y_dev,
        predictions,
        labels=classes,
        zero_division=0,
    )

    print(report)

    report_path = output / f"{name}_final_report.txt"
    report_path.write_text(report, encoding="utf-8")

    matrix = confusion_matrix(
        y_dev,
        predictions,
        labels=classes,
    )

    np.savetxt(
        output / f"{name}_confusion_counts.csv",
        matrix,
        delimiter=",",
        fmt="%d",
        header=",".join(classes),
        comments="",
    )

    print("Saved report:", report_path)


print("\nDevelopment-set evaluation complete!")
