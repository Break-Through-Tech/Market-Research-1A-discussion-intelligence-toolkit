
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from utils.modeling_data.build import load_prepared_splits


print("Loading dataset...", flush=True)

splits = load_prepared_splits()
train = splits.train
dev = splits.dev

y_train = np.array([item.label for item in train])
y_dev = np.array([item.label for item in dev])


def parent_text(item):
    value = item.metadata.get("parent_text")
    return value.strip() if isinstance(value, str) else ""


train_text = [item.text for item in train]
dev_text = [item.text for item in dev]

train_parent = [parent_text(item) for item in train]
dev_parent = [parent_text(item) for item in dev]


# Compute class weights at 50% strength
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


# Vectorize target text
print("Creating target text features...", flush=True)

target_vectorizer = TfidfVectorizer(max_features=10000)

X_train_target = target_vectorizer.fit_transform(train_text)
X_dev_target = target_vectorizer.transform(dev_text)


# Create parent availability indicator
print("Creating parent availability features...", flush=True)

X_train_flag = csr_matrix(
    np.array([bool(text) for text in train_parent], dtype=float).reshape(-1, 1)
)

X_dev_flag = csr_matrix(
    np.array([bool(text) for text in dev_parent], dtype=float).reshape(-1, 1)
)


# Vectorize actual parent text
print("Creating parent text features...", flush=True)

parent_vectorizer = TfidfVectorizer(max_features=10000)

X_train_parent = parent_vectorizer.fit_transform(train_parent)
X_dev_parent = parent_vectorizer.transform(dev_parent)


# Build three feature configurations
configurations = {
    "A_text_only": (
        X_train_target,
        X_dev_target,
    ),
    "B_parent_available": (
        hstack([X_train_target, X_train_flag], format="csr"),
        hstack([X_dev_target, X_dev_flag], format="csr"),
    ),
    "C_parent_text": (
        hstack([X_train_target, X_train_parent], format="csr"),
        hstack([X_dev_target, X_dev_parent], format="csr"),
    ),
}


results = []

for name, (X_train, X_dev) in configurations.items():

    print(f"\nTraining {name}...", flush=True)

    model = LogisticRegression(
        max_iter=1000,
        class_weight=weights,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, predictions)
    macro = f1_score(y_dev, predictions, average="macro")
    weighted = f1_score(y_dev, predictions, average="weighted")

    results.append((name, accuracy, macro, weighted))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro:.4f}")
    print(f"Weighted F1: {weighted:.4f}")


print("\nABLATION STUDY SUMMARY")

for name, accuracy, macro, weighted in results:
    print(
        f"{name}: accuracy={accuracy:.4f}, "
        f"macro_f1={macro:.4f}, weighted_f1={weighted:.4f}"
    )


# Save experiment results
output = Path("experiments/franceli/results")
output.mkdir(parents=True, exist_ok=True)

lines = ["model,accuracy,macro_f1,weighted_f1"]

for name, accuracy, macro, weighted in results:
    lines.append(
        f"{name},{accuracy:.4f},{macro:.4f},{weighted:.4f}"
    )

(output / "context_ablation.csv").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\nSaved: experiments/franceli/results/context_ablation.csv")
