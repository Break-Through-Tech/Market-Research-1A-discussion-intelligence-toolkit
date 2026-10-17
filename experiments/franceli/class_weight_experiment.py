from collections import Counter
from utils.modeling_data.build import load_prepared_splits

# Load the standardized data from Issue #14
splits = load_prepared_splits()

X_train = [example.text for example in splits.train]
y_train = [example.label for example in splits.train]

X_dev = [example.text for example in splits.dev]
y_dev = [example.label for example in splits.dev]

# Count examples belonging to each category
class_counts = Counter(y_train)

print("Training examples:", len(X_train))
print("Development examples:", len(X_dev))

print("\nTraining class distribution:")
for label, count in class_counts.most_common():
    print(f"{label}: {count:,}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from artifacts.evaluation import evaluate_classifier

# Create a Logistic Regression model that balances class weights
balanced_model = make_pipeline(
    TfidfVectorizer(max_features=10000),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
)

print("\nTraining class-balanced model...")

balanced_model.fit(X_train, y_train)

print("Training complete!")

# Evaluate using the development dataset
results = evaluate_classifier(
    balanced_model,
    texts=X_dev,
    labels=y_dev
)

print("\nClass-Balanced Model Results")
print("Accuracy:", round(results.accuracy, 4))
print("Macro F1:", round(results.macro_f1, 4))
print("Weighted F1:", round(results.weighted_f1, 4))

print("\nPer-Class F1 Scores")
for label, score in results.per_class_f1.items():
    print(f"{label}: {score:.4f}")

