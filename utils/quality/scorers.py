"""
Conversation quality scorers.

Every scorer takes conversations as lists of comment texts (oldest first) and
returns a quality score in [0, 1]: the predicted probability that the
conversation stays civil.

    ConstantScorer    train-set civil rate for everything (floor)
    LengthScorer      logistic regression on comment count and length
    TfidfScorer       TF-IDF word 1-2 grams + logistic regression
    EmbeddingScorer   sentence embeddings + logistic regression

Scores are probabilities, so the same output feeds both the ranking metrics
(ROC-AUC, Spearman) and the level metrics (Brier, MAE) in `utils.quality.metrics`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from artifacts.evaluation import TextEmbedder

Conversations = Sequence[Sequence[str]]


def _join(conversations: Conversations) -> list[str]:
    return ["\n\n".join(c) for c in conversations]


class ConstantScorer:
    name = "constant"

    def fit(self, conversations: Conversations, civil: Sequence[int]) -> "ConstantScorer":
        self.rate_ = float(np.mean(civil))
        return self

    def predict(self, conversations: Conversations) -> list[float]:
        return [self.rate_] * len(conversations)


class LengthScorer:
    """Shape-only baseline: how many comments, how long they are.

    Not a useful quality signal on its own, but it shows how much of the
    task a model can solve without reading the text.
    """

    name = "length"

    def __init__(self, C: float = 1.0) -> None:
        self.C = C

    @staticmethod
    def features(conversations: Conversations) -> np.ndarray:
        rows = []
        for comments in conversations:
            lengths = [len(c.split()) for c in comments] or [0]
            rows.append([
                np.log1p(len(comments)),
                np.log1p(np.mean(lengths)),
                np.log1p(lengths[-1]),
            ])
        return np.asarray(rows, dtype=float)

    def fit(self, conversations: Conversations, civil: Sequence[int]) -> "LengthScorer":
        self.scaler_ = StandardScaler().fit(self.features(conversations))
        self.model_ = LogisticRegression(C=self.C, max_iter=1000)
        self.model_.fit(self.scaler_.transform(self.features(conversations)), civil)
        return self

    def predict(self, conversations: Conversations) -> list[float]:
        x = self.scaler_.transform(self.features(conversations))
        return self.model_.predict_proba(x)[:, 1].tolist()


class TfidfScorer:
    name = "tfidf_lr"

    def __init__(self, C: float = 1.0, max_features: int = 50_000) -> None:
        self.C = C
        self.max_features = max_features

    def fit(self, conversations: Conversations, civil: Sequence[int]) -> "TfidfScorer":
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=self.max_features, sublinear_tf=True
        )
        x = self.vectorizer_.fit_transform(_join(conversations))
        self.model_ = LogisticRegression(C=self.C, max_iter=2000)
        self.model_.fit(x, civil)
        return self

    def predict(self, conversations: Conversations) -> list[float]:
        x = self.vectorizer_.transform(_join(conversations))
        return self.model_.predict_proba(x)[:, 1].tolist()

    def top_terms(self, k: int = 15) -> dict[str, list[tuple[str, float]]]:
        """Highest-weight n-grams toward each label, for error analysis."""
        terms = self.vectorizer_.get_feature_names_out()
        coef = self.model_.coef_[0]
        order = np.argsort(coef)
        return {
            "civil": [(terms[i], round(float(coef[i]), 3)) for i in order[::-1][:k]],
            "derailed": [(terms[i], round(float(coef[i]), 3)) for i in order[:k]],
        }


class EmbeddingScorer:
    """Logistic regression over pooled sentence embeddings.

    Each comment is embedded on its own (long threads would otherwise be cut
    at the encoder's token limit). A conversation's features are the mean of
    its comment embeddings joined with the embedding of its latest comment,
    since the most recent turn says the most about where the thread is going.
    """

    name = "embedding_lr"

    def __init__(self, embedder: TextEmbedder, C: float = 1.0) -> None:
        self.embedder = embedder
        self.C = C

    def features(self, conversations: Conversations) -> np.ndarray:
        flat = [c for comments in conversations for c in comments]
        vectors = np.asarray(self.embedder.embed(flat), dtype=np.float32)
        rows, start = [], 0
        for comments in conversations:
            block = vectors[start:start + len(comments)]
            start += len(comments)
            rows.append(np.concatenate([block.mean(axis=0), block[-1]]))
        return np.vstack(rows)

    def fit(self, conversations: Conversations, civil: Sequence[int]) -> "EmbeddingScorer":
        x = self.features(conversations)
        self.scaler_ = StandardScaler().fit(x)
        self.model_ = LogisticRegression(C=self.C, max_iter=2000)
        self.model_.fit(self.scaler_.transform(x), civil)
        return self

    def predict(self, conversations: Conversations) -> list[float]:
        x = self.scaler_.transform(self.features(conversations))
        return self.model_.predict_proba(x)[:, 1].tolist()

