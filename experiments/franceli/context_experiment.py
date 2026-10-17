from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from utils.modeling_data.build import load_prepared_splits


print("Loading prepared dataset...", flush=True)

splits = load_prepared_splits()

train = splits.train
dev = splits.dev

y_train = np.array([example.label for example in train])
y_dev = np.array([example.label for example in dev])


def get_parent_text(example):
    value = example.metadata.get("parent_text")
    return value if isinstance(value, str) else ""


train_text = [example.text for example in train]
dev_text = [example.text for example in dev]

train_parent = [get_parent_text(example) for example in train]
dev_parent = [get_parent_text(example) for example in dev]


# Use the same class weighting as our strongest model
classes = np.unique(y_train)

balanced = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

strength = 0.50

weights = {
    label: 1 + strength * (float(weight) - 1)
    for label, weight in zip(classes, balanced)
}


# Step 1: Vectorize target comments
print("Vectorizing target comments...", flush=True)

target_vectorizer = TfidfVectorizer(max_features=10000)

X_train_target = target_vectorizer.fit_transform(train_text)
X_dev_target = target_vectorizer.transform(dev_text)


# Step 2: Vectorize parent comments separately
print("Vectorizing parent comments...", flush=True)

parent_vectorizer = TfidfVectorizer(max_features=10000)

X_train_parent = parent_vectorizer.fit_transform(train_parent)
X_dev_parent = parent_vectorizer.transform(dev_parent)


# Step 3: Combine the features
X_train_context = hstack(
    [X_train_target, X_train_parent],
    format="csr"
)

X_dev_context = hstack(
    [X_dev_target, X_dev_parent],
    format="csr"
)


def evaluate(name, X_train, X_dev):
    print(f"\nTraining: {name}", flush=True)

    model = LogisticRegression(
        max_iter=1000,
        class_weight=weights,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, predictions)
    macro = f1_score(y_dev, predictions, average="macro")
    weighted = f1_score(y_dev, predictions, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro:.4f}")
    print(f"Weighted F1: {weighted:.4f}")

    return accuracy, macro, weighted


# Step 4: Compare models
baseline = evaluate(
    "Text-only baseline",
    X_train_target,
    X_dev_target,
)

context = evaluate(
    "Target + parent context",
    X_train_context,
    X_dev_context,
)


# Step 5: Calculate improvement
difference = context[1] - baseline[1]

print("\nCONTEXT EXPERIMENT SUMMARY")
print(f"Text-only macro F1: {baseline[1]:.4f}")
print(f"Context-aware macro F1: {context[1]:.4f}")
print(f"Macro F1 difference: {difference:+.4f}")

if difference > 0:
    print("Context-aware model improved macro F1.")
elif difference < 0:
    print("Text-only baseline performed better.")
else:
    print("Both models achieved the same macro F1.")


# Step 6: Save results
output = Path("experiments/franceli/results")
output.mkdir(parents=True, exist_ok=True)

summary = (
    "Context Experiment Results\n"
    f"Text-only accuracy: {baseline[0]:.4f}\n"
    f"Text-only macro F1: {baseline[1]:.4f}\n"
    f"Text-only weighted F1: {baseline[2]:.4f}\n"
    f"Context accuracy: {context[0]:.4f}\n"
    f"Context macro F1: {context[1]:.4f}\n"
    f"Context weighted F1: {context[2]:.4f}\n"
    f"Macro F1 difference: {difference:+.4f}\n"
)

(output / "context_experiment.txt").write_text(
    summary,
    encoding="utf-8",
)

print("\nSaved results to context_experiment.txt")
