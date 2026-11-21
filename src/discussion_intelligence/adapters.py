"""Inference-only bridges for fitted models. No training, disk loading or downloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .reports import ConversationQuality, Evidence, SynthesisDraft


@dataclass(frozen=True)
class InferenceUtterance:
    """Gold labels and annotation metadata deliberately do not reach models."""

    id: str
    text: str
    parent_id: str | None
    parent_text: str
    speaker_id: str
    created_at: datetime | None = None


class DiscourseClassifier(Protocol):
    def predict(self, utterances: Sequence[InferenceUtterance]) -> Mapping[str, str]: ...


class QualityScorer(Protocol):
    def score(self, utterances: Sequence[InferenceUtterance]) -> ConversationQuality: ...


class TopicClusterer(Protocol):
    """A partial mapping is allowed; unassigned utterances remain ungrouped."""

    def assign(self, utterances: Sequence[InferenceUtterance]) -> Mapping[str, str]: ...


class SynthesisProvider(Protocol):
    def synthesize(self, evidence: Sequence[Evidence], query: str | None) -> SynthesisDraft: ...


class TextClassifierAdapter:
    """Wrap a fitted sklearn pipeline or artifacts.evaluation.TextClassifier."""

    def __init__(self, model):
        self.model = model

    def predict(self, utterances: Sequence[InferenceUtterance]) -> dict[str, str]:
        labels = list(self.model.predict([u.text for u in utterances]))
        if len(labels) != len(utterances):
            raise ValueError("Classifier prediction length mismatch")
        return dict(zip((u.id for u in utterances), labels))


class ContextClassifierAdapter(TextClassifierAdapter):
    """Franceli's target + parent sparse-feature contract, using fitted objects.

    Pass the two vectorizers and the selected LogisticRegression after training
    elsewhere. transform()/predict() only; never import the experiment scripts.
    """

    def __init__(self, model, target_vectorizer, parent_vectorizer):
        super().__init__(model)
        self.target_vectorizer = target_vectorizer
        self.parent_vectorizer = parent_vectorizer

    def predict(self, utterances: Sequence[InferenceUtterance]) -> dict[str, str]:
        from scipy.sparse import hstack

        features = hstack(
            [
                self.target_vectorizer.transform([u.text for u in utterances]),
                self.parent_vectorizer.transform([u.parent_text for u in utterances]),
            ],
            format="csr",
        )
        labels = list(self.model.predict(features))
        if len(labels) != len(utterances):
            raise ValueError("Context classifier prediction length mismatch")
        return dict(zip((u.id for u in utterances), labels))


class ConversationCivilityAdapter:
    """Wrap Gia's fitted predict([list_of_comment_texts]) -> [P(civil)] contract.

    All supplied usable utterances are the *observed context*. This adapter
    does not guess which is a future removed outcome. CGA-CMV evaluation must
    use the outcome-excluding loader, not a complete labeled conversation.
    """

    def __init__(self, model, *, model_name: str):
        self.model = model
        self.model_name = model_name

    def score(self, utterances: Sequence[InferenceUtterance]) -> ConversationQuality:
        if not utterances:
            raise ValueError("Cannot score civility without observed context")
        ordered = list(utterances)
        if all(u.created_at is not None for u in ordered):
            ordered.sort(key=lambda u: u.created_at)
        scores = list(self.model.predict([[u.text for u in ordered]]))
        if len(scores) != 1:
            raise ValueError("Quality scorer must return one conversation score")
        return ConversationQuality(
            value=scores[0], evidence_ids=[u.id for u in ordered], model_name=self.model_name
        )
