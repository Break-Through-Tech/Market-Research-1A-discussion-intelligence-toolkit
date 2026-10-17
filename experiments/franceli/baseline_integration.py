from utils.modeling_data.build import load_prepared_splits

# Load the standardized datasets from Issue #14
splits = load_prepared_splits()

# Separate the comment text and discourse labels
X_train = [example.text for example in splits.train]
y_train = [example.label for example in splits.train]

X_dev = [example.text for example in splits.dev]
y_dev = [example.label for example in splits.dev]

print("Training examples:", len(X_train))
print("Development examples:", len(X_dev))
print("Discourse labels:", sorted(set(y_train)))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Convert Reddit comments into numerical features
# and train a classifier using those features.
model = make_pipeline(
    TfidfVectorizer(max_features=10000),
    LogisticRegression(max_iter=1000)
)

print("Training baseline model...")

model.fit(X_train, y_train)

print("Training complete!")

from artifacts.evaluation import evaluate_classifier

# Evaluate the trained model on development data
dev_results = evaluate_classifier(
    model,
    texts=X_dev,
    labels=y_dev
)

print("\nBaseline Evaluation Results")
print("Accuracy:", round(dev_results.accuracy, 4))
print("Macro F1:", round(dev_results.macro_f1, 4))
print("Weighted F1:", round(dev_results.weighted_f1, 4))

print("\nPer-Class F1 Scores")
for label, score in dev_results.per_class_f1.items():
    print(f"{label}: {score:.4f}")
