
"""Tests for evidence-grounded LLM synthesis."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from artifacts.schema import Conversation, Speaker, Utterance
from discussion_intelligence.llm_synthesis import (
    LLMSynthesisProvider,
    create_openai_provider,
)
from discussion_intelligence.pipeline import analyze
from discussion_intelligence.reports import Evidence


def make_evidence():
    """Create sample evidence for testing."""
    return Evidence(
        id="comment_101",
        conversation_id="thread_1",
        source_id="reddit_101",
        parent_id=None,
        speaker_id="user_1",
        text="AI helps me understand difficult concepts.",
        text_clean="AI helps me understand difficult concepts.",
    )


def test_valid_llm_synthesis():
    """Verify that valid evidence-backed quotations are accepted."""

    def fake_llm(prompt):
        return """
        {
            "points": [
                {
                    "text": "AI helps me understand difficult concepts.",
                    "evidence_ids": ["comment_101"]
                }
            ]
        }
        """

    provider = LLMSynthesisProvider(generate=fake_llm)

    result = provider.synthesize(
        [make_evidence()],
        query="AI in education",
    )

    assert len(result.points) == 1
    assert result.points[0].evidence_ids == ["comment_101"]


def test_rejects_hallucinated_quotation():
    """Verify that unsupported quotations are rejected."""

    def fake_llm(prompt):
        return """
        {
            "points": [
                {
                    "text": "AI guarantees better grades.",
                    "evidence_ids": ["comment_101"]
                }
            ]
        }
        """

    provider = LLMSynthesisProvider(generate=fake_llm)

    with pytest.raises(ValueError):
        provider.synthesize([make_evidence()], query=None)


def test_openai_provider_with_mock_client():
    """Verify OpenAI integration without making a real API request."""

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)

        return SimpleNamespace(
            output_text="""
            {
                "points": [
                    {
                        "text": "AI helps me understand difficult concepts.",
                        "evidence_ids": ["comment_101"]
                    }
                ]
            }
            """
        )

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=fake_create)
    )

    provider = create_openai_provider(
        model="gpt-4.1-mini",
        client=fake_client,
    )

    result = provider.synthesize(
        [make_evidence()],
        query="AI in education",
    )

    assert len(result.points) == 1
    assert result.points[0].evidence_ids == ["comment_101"]

    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-4.1-mini"
    assert "AI in education" in calls[0]["input"]


def test_llm_synthesis_pipeline_integration():
    """Verify synthesis works with the full Reddit analysis pipeline."""

    conversation = Conversation(
        id="reddit:thread_1",
        source_id="reddit_thread_1",
        source="reddit",
        title="AI in education",
        created_at=datetime(2026, 11, 14, tzinfo=timezone.utc),
        speakers={
            "user_1": Speaker(
                id="user_1",
                handle="user_1",
                source="reddit",
            )
        },
        utterances=[
            Utterance(
                id="reddit:comment_101",
                conversation_id="reddit:thread_1",
                source_id="reddit_101",
                parent_id=None,
                speaker_id="user_1",
                text="AI helps me understand difficult concepts.",
                created_at=datetime(2026, 11, 14, tzinfo=timezone.utc),
                depth=0,
            )
        ],
    )

    def fake_llm(prompt):
        return """
        {
            "points": [
                {
                    "text": "AI helps me understand difficult concepts.",
                    "evidence_ids": ["reddit:comment_101"]
                }
            ]
        }
        """

    provider = LLMSynthesisProvider(generate=fake_llm)

    report = analyze(
        conversation,
        query="AI in education",
        synthesis_provider=provider,
    )

    assert report.components["synthesis"].state == "available"
    assert report.summary[0].text == (
        "AI helps me understand difficult concepts."
    )
    assert report.summary[0].evidence_ids == ["reddit:comment_101"]


def test_gemini_provider_on_real_reference_thread():
    """Verify Gemini's interface works with real Reddit evidence."""
    import json

    from discussion_intelligence.llm_synthesis import create_gemini_provider
    from discussion_intelligence.retrieval import EvidenceIndex
    from discussion_intelligence.sample import load_reference_conversations

    conversation = load_reference_conversations(
        "data/coarse-discourse-reference-sample.jsonl"
    )[0]
    query = "RAID data drives"
    selected = EvidenceIndex(conversation).search(query, top_k=3)[0].evidence
    quote = selected.text_clean[:100]
    calls = []

    def fake_generate_content(*, model, contents, config):
        calls.append((model, contents, config))
        return SimpleNamespace(
            text=json.dumps({
                "points": [
                    {"text": quote, "evidence_ids": [selected.id]}
                ]
            })
        )

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=fake_generate_content)
    )

    report = analyze(
        conversation,
        query=query,
        top_k=3,
        synthesis_provider=create_gemini_provider(client=client),
    )

    assert report.components["synthesis"].state == "available"
    assert report.summary[0].text == quote
    assert report.summary[0].evidence_ids == [selected.id]
    assert len(calls) == 1
    assert calls[0][0] == "gemini-3.5-flash-lite"
    assert calls[0][2]["response_mime_type"] == "application/json"
