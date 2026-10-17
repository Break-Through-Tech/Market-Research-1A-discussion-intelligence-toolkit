
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

from utils.modeling_data.build import load_prepared_splits


print("Loading prepared dataset...", flush=True)

splits = load_prepared_splits()

train = splits.train
test = splits.test

y_train = np.array([item.label for item in train])
y_test = np.array([item.label for item in test])


def get_parent_text(item):
    value = item.metadata.get("parent_text")
    return value.strip() if isinstance(value, str) else ""


train_text = [item.text for item in train]
test_text = [item.text for item in test]

train_parent = [get_parent_text(item) for item in train]
test_parent = [get_parent_text(item) for item in test]


# Freeze the previously selected 50% class balancing
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


# Fit vectorizers on training data only
print("Creating TF-IDF features...", flush=True)

target_vectorizer = TfidfVectorizer(max_features=10000)
parent_vectorizer = TfidfVectorizer(max_features=10000)

X_train_target = target_vectorizer.fit_transform(train_text)
X_test_target = target_vectorizer.transform(test_text)

X_train_parent = parent_vectorizer.fit_transform(train_parent)
X_test_parent = parent_vectorizer.transform(test_parent)

X_train = hstack(
    [X_train_target, X_train_parent],
    format="csr",
)

X_test = hstack(
    [X_test_target, X_test_parent],
    format="csr",
)


# Train the frozen model on the training split
print("Training selected classifier...", flush=True)

model = LogisticRegression(
    max_iter=1000,
    class_weight=weights,
)

model.fit(X_train, y_train)


# Evaluate once on held-out test data
print("Evaluating held-out test set...", flush=True)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")
weighted_f1 = f1_score(y_test, predictions, average="weighted")

print("\nHELD-OUT TEST RESULTS")
print(f"Test examples: {len(test)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

report = classification_report(
    y_test,
    predictions,
    labels=classes,
    zero_division=0,
)

print("\nPer-class classification report:")
print(report)


# Save results
output = Path("experiments/franceli/results")
output.mkdir(parents=True, exist_ok=True)

summary = (
    "Held-Out Test Evaluation\n"
    "Model: TF-IDF target + TF-IDF parent + Logistic Regression\n"
    "Class balancing: 50%\n"
    f"Test examples: {len(test)}\n"
    f"Accuracy: {accuracy:.4f}\n"
    f"Macro F1: {macro_f1:.4f}\n"
    f"Weighted F1: {weighted_f1:.4f}\n\n"
    + report
)

path = output / "heldout_test_report.txt"
path.write_text(summary, encoding="utf-8")

print("\nSaved:", path)
print("Held-out evaluation complete.")
