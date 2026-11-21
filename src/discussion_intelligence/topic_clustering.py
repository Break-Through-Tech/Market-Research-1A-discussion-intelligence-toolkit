"""Inference adapter for Franceli's trained semantic clustering model."""

from collections.abc import Sequence

from .adapters import InferenceUtterance


class SemanticTopicClusterer:
    """Assign existing topic IDs without retraining."""

    def __init__(self, encoder, model, *, normalize_embeddings=True):
        self.encoder = encoder
        self.model = model
        self.normalize_embeddings = normalize_embeddings

    def assign(self, utterances: Sequence[InferenceUtterance]) -> dict[str, str]:
        if not utterances:
            return {}

        embeddings = self.encoder.encode(
            [u.text for u in utterances],
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        predictions = self.model.predict(embeddings)

        if len(predictions) != len(utterances):
            raise ValueError("Topic prediction count mismatch")

        return {
            utterance.id: str(int(topic_id))
            for utterance, topic_id in zip(
                utterances, predictions, strict=True
            )
        }
