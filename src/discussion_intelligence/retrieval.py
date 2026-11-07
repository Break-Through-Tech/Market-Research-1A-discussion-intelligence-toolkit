"""Deterministic TF-IDF/cosine retrieval, with injectable sentence embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from artifacts.schema import Conversation

from .reports import Evidence, ReportModel
from .validation import usable_text, validate_conversation


class TextEmbedder(Protocol):
    """Structural match for artifacts.evaluation.TextEmbedder and Gia's embedder."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class SearchHit(ReportModel):
    evidence: Evidence
    score: float


class EvidenceIndex:
    """Index one conversation. Scores measure relevance, never argument quality.

    Ties use canonical IDs, independent of source list order. Query search
    excludes zero/negative similarity; representative() ranks centroid
    similarity when no query is supplied. No query expansion or model downloads.
    """

    def __init__(self, conversation: Conversation, *, embedder: TextEmbedder | None = None):
        conv = validate_conversation(conversation)
        self.backend = "embeddings" if embedder is not None else "tfidf"
        self._embedder = embedder
        self.evidence: dict[str, Evidence] = {}
        for u in sorted(conv.utterances, key=lambda item: item.id):
            text = usable_text(u.text, u.is_deleted)
            if not text:
                continue
            self.evidence[u.id] = Evidence(
                id=u.id,
                conversation_id=conv.id,
                source_id=u.source_id,
                parent_id=u.parent_id,
                speaker_id=u.speaker_id,
                text=u.text,
                text_clean=text,
                source_url=u.metadata.get("url") or u.metadata.get("source_url"),
                conversation_url=conv.url,
                created_at=None if u.metadata.get("timestamp_missing") else u.created_at,
                metadata=u.metadata.copy(),
                discourse_act=u.discourse_act,
                argument_quality_score=u.quality_score,
            )
        self._ids = list(self.evidence)
        texts = [e.text_clean for e in self.evidence.values()]
        self._vectors = None
        self._vectorizer = None
        if not texts:
            return
        if embedder is not None:
            self._vectors = self._validated_vectors(embedder.embed(texts), len(texts))
        else:
            # Single-character tokens are useful in short technical discussions.
            self._vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
            if not any(self._vectorizer.build_analyzer()(text) for text in texts):
                return  # punctuation-only text has no searchable vocabulary
            self._vectors = self._vectorizer.fit_transform(texts)

    @staticmethod
    def _validated_vectors(values, rows: int, dimensions: int | None = None) -> np.ndarray:
        vectors = np.asarray(values, dtype=float)
        if (
            vectors.ndim != 2
            or vectors.shape[0] != rows
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
            or (dimensions is not None and vectors.shape[1] != dimensions)
        ):
            raise ValueError("Embedder must return finite (N, D) vectors with a stable dimension")
        return vectors

    @staticmethod
    def _check_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        self._check_k(top_k)
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip() or top_k == 0 or self._vectors is None:
            return []
        if self._embedder is not None:
            query_vector = self._validated_vectors(
                self._embedder.embed([query]), 1, self._vectors.shape[1]
            )
        else:
            query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._vectors)[0]
        return self._rank(scores, top_k, positive_only=True)

    def representative(self, *, top_k: int = 5) -> list[SearchHit]:
        self._check_k(top_k)
        if top_k == 0 or self._vectors is None:
            return []
        centroid = np.asarray(self._vectors.mean(axis=0)).reshape(1, -1)
        scores = cosine_similarity(centroid, self._vectors)[0]
        return self._rank(scores, top_k, positive_only=False)

    def _rank(self, scores, top_k: int, *, positive_only: bool) -> list[SearchHit]:
        ranked = sorted(zip(self._ids, scores), key=lambda item: (-float(item[1]), item[0]))
        return [
            SearchHit(evidence=self.evidence[uid].model_copy(deep=True), score=float(score))
            for uid, score in ranked
            if not positive_only or score > 0
        ][:top_k]
