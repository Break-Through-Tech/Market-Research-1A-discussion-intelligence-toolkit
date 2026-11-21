"""Tests for the semantic clustering pipeline adapter."""

import json
from types import SimpleNamespace

import pytest

from discussion_intelligence.adapters import InferenceUtterance
from discussion_intelligence.llm_synthesis import create_gemini_provider
from discussion_intelligence.pipeline import analyze
from discussion_intelligence.retrieval import EvidenceIndex
from discussion_intelligence.sample import load_reference_conversations
from discussion_intelligence.topic_clustering import SemanticTopicClusterer


class FakeEncoder:
    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["convert_to_numpy"] is True
        return [[len(text)] for text in texts]


class FakeKMeans:
    def predict(self, embeddings):
        return [11 for _ in embeddings]


def test_clustering_and_gemini_pipeline():
    conversation = load_reference_conversations(
        "data/coarse-discourse-reference-sample.jsonl"
    )[0]
    query = "RAID data drives"
    evidence = EvidenceIndex(conversation).search(query, top_k=3)[0].evidence

    response = json.dumps({
        "points": [{
            "text": evidence.text_clean[:100],
            "evidence_ids": [evidence.id],
        }]
    })

    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(text=response)
        )
    )

    report = analyze(
        conversation,
        query=query,
        top_k=3,
        clusterer=SemanticTopicClusterer(FakeEncoder(), FakeKMeans()),
        synthesis_provider=create_gemini_provider(client=client),
    )

    assert report.components["clustering"].state == "available"
    assert report.components["synthesis"].state == "available"
    assert report.components["retrieval"].state == "available"
    assert len(report.topic_assignments) == 7
    assert set(report.topic_assignments.values()) == {"11"}
    assert report.sections[0].heading == "Topic cluster 11"
    assert report.summary[0].evidence_ids == [evidence.id]


def test_clustering_rejects_wrong_prediction_count():
    class BrokenModel:
        def predict(self, embeddings):
            return []

    utterance = InferenceUtterance(
        id="comment_1",
        text="Example discussion comment",
        parent_id=None,
        parent_text="",
        speaker_id="user_1",
    )

    clusterer = SemanticTopicClusterer(FakeEncoder(), BrokenModel())

    with pytest.raises(ValueError, match="count mismatch"):
        clusterer.assign([utterance])
