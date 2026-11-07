import json
import re
from html.parser import HTMLParser
from urllib.parse import quote, unquote

import pytest

from discussion_intelligence import analyze
from discussion_intelligence.adapters import ConversationCivilityAdapter, TextClassifierAdapter
from discussion_intelligence.reports import Claim, InsightReport, SynthesisDraft


def test_honest_fallback_and_no_gold_label_leakage(conversation):
    before = conversation.model_dump_json()
    report = analyze(conversation)
    assert report.summary
    assert (
        report.questions == report.agreements == report.disagreements == report.open_questions == []
    )
    assert report.conversation_quality is None
    for name in ("discourse", "quality", "clustering", "embeddings"):
        assert report.components[name].state == "unavailable"
    assert all(
        e.discourse_act is None and e.argument_quality_score is None
        for e in report.evidence.values()
    )
    assert conversation.model_dump_json() == before


def test_json_markdown_and_citation_integrity(conversation):
    report = analyze(conversation)
    restored = InsightReport.model_validate_json(report.to_json())
    assert restored == report
    markdown = report.to_markdown()
    assert all(line == line.rstrip() for line in markdown.splitlines())
    for point in report.summary:
        for uid in point.evidence_ids:
            assert uid in report.evidence
            assert point.text in report.evidence[uid].text_clean
            assert f"#evidence-{quote(uid, safe='')}" in markdown

    class Anchors(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = set()

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.ids.add(dict(attrs).get("id"))

    anchors = Anchors()
    anchors.feed(markdown)
    # Browsers decode URL fragments before matching an HTML element ID.
    assert all(
        unquote(fragment) in anchors.ids for fragment in re.findall(r"\]\(#([^)]*)\)", markdown)
    )
    assert "evidence-reddit:t1_storage" in anchors.ids
    assert "Component availability" in markdown
    assert json.loads(report.to_json())["conversation_id"] == conversation.id


def test_repeated_runs_byte_identical(conversation):
    assert analyze(conversation).to_json() == analyze(conversation).to_json()
    assert analyze(conversation).to_markdown() == analyze(conversation).to_markdown()


def test_query_with_no_match_does_not_pad_report(conversation):
    report = analyze(conversation, query="telescope")
    assert not report.summary and not report.sections
    assert "No supported excerpts selected" in report.to_markdown()


def test_model_enrichment_is_scoped_and_nonmutating(conversation):
    class Classifier:
        def predict(self, texts):
            return ["agreement"] * len(texts)

    class Scorer:
        def predict(self, contexts):
            assert len(contexts) == 1 and len(contexts[0]) == 3
            return [0.7]

    class Clusterer:
        def assign(self, utterances):
            assert all(not hasattr(u, "metadata") for u in utterances)
            return {u.id: "0" for u in utterances}

    report = analyze(
        conversation,
        classifier=TextClassifierAdapter(Classifier()),
        quality_scorer=ConversationCivilityAdapter(Scorer(), model_name="test-fitted"),
        clusterer=Clusterer(),
    )
    assert report.agreements and not report.open_questions
    assert report.conversation_quality.value == 0.7
    assert report.conversation_quality.scope == "conversation"
    assert report.conversation_quality.target == "civility_probability"
    assert all(e.argument_quality_score is None for e in report.evidence.values())
    assert all(u.discourse_act is None for u in conversation.utterances)
    assert report.sections[0].heading == "Topic cluster 0"
    assert "P(civil): 0.7000" in report.to_markdown()


@pytest.mark.parametrize(
    "bad", [{"reddit:unknown": "answer"}, {"reddit:t3_test": "made_up"}, ["answer"]]
)
def test_invalid_classifier_output_isolated(conversation, bad):
    class Classifier:
        def predict(self, utterances):
            return bad

    report = analyze(conversation, classifier=Classifier())
    assert report.components["discourse"].state == "failed"
    assert report.summary
    assert all(e.discourse_act is None for e in report.evidence.values())


@pytest.mark.parametrize("stage", ["discourse", "quality", "clustering"])
def test_component_runtime_failure_isolated(conversation, stage):
    class Broken:
        def predict(self, utterances):
            raise RuntimeError("secret-do-not-echo")

        score = assign = predict

    kwargs = {"discourse": "classifier", "quality": "quality_scorer", "clustering": "clusterer"}
    report = analyze(conversation, **{kwargs[stage]: Broken()})
    assert report.components[stage].state == "failed"
    assert "secret-do-not-echo" not in report.to_json()
    assert report.summary


@pytest.mark.parametrize(
    "value,ids",
    [(float("nan"), ["reddit:t3_test"]), (1.1, ["reddit:t3_test"]), (0.5, ["reddit:missing"])],
)
def test_invalid_quality_isolated(conversation, value, ids):
    class Scorer:
        def score(self, utterances):
            return {"value": value, "evidence_ids": ids, "model_name": "test"}

    report = analyze(conversation, quality_scorer=Scorer())
    assert report.conversation_quality is None
    assert report.components["quality"].state == "failed"


def test_partial_topic_assignments_preserve_ungrouped_evidence(conversation):
    class Clusterer:
        def assign(self, utterances):
            return {"reddit:t1_storage": "storage"}

    report = analyze(conversation, clusterer=Clusterer())
    assert {s.heading for s in report.sections} == {"Discussion excerpts", "Topic cluster storage"}
    assert len([p for s in report.sections for p in s.points]) == 3


class Provider:
    def synthesize(self, evidence, query):
        return SynthesisDraft(
            points=[Claim(text=evidence[0].text_clean, evidence_ids=[evidence[0].id])]
        )


def test_validated_injected_quotation_provider(conversation):
    report = analyze(conversation, query="RAID", synthesis_provider=Provider())
    assert report.components["synthesis"].state == "available"
    assert "Injected provider" in report.components["synthesis"].detail
    assert len(report.summary) == 1


@pytest.mark.parametrize(
    "kind", ["unknown", "unretrieved", "fabricated", "empty", "duplicate", "runtime"]
)
def test_unsafe_synthesis_rejected_and_falls_back(conversation, kind):
    class BadProvider:
        def synthesize(self, evidence, query):
            if kind == "runtime":
                raise RuntimeError("unavailable")
            uid = evidence[0].id
            ids = ["reddit:missing"] if kind == "unknown" else [uid]
            text = evidence[0].text_clean
            if kind == "unretrieved":
                ids, text = ["reddit:t1_gpu"], "A blue graphics card offers faster gaming."
            elif kind == "fabricated":
                text = "Everyone agrees with this product."
            elif kind == "duplicate":
                ids = [uid, uid]
            return {"points": [] if kind == "empty" else [{"text": text, "evidence_ids": ids}]}

    report = analyze(conversation, query="RAID", top_k=1, synthesis_provider=BadProvider())
    assert report.components["synthesis"].state == "failed"
    assert report.summary[0].evidence_ids == ["reddit:t1_storage"]
    assert report.summary[0].text == conversation.utterances[1].text


def test_invalid_serialized_and_mutated_report_rejected(conversation):
    report = analyze(conversation)
    payload = json.loads(report.to_json())
    payload["summary"][0]["evidence_ids"] = ["reddit:missing"]
    with pytest.raises(ValueError):
        InsightReport.model_validate(payload)
    report.summary[0].text = "An invented assertion."
    with pytest.raises(ValueError):
        report.to_markdown()
    with pytest.raises(ValueError):
        report.to_json()


def test_source_markdown_is_escaped(conversation):
    conversation.title = "<script>alert(1)</script>"
    conversation.utterances[1].text = '<img src=x onerror="alert(1)"> [go](javascript:evil)'
    report = analyze(conversation)
    markdown = report.to_markdown()
    assert "<script>" not in markdown and "<img " not in markdown
    assert "\\[go\\]" in markdown


def test_supplied_enrichments_preserved_separately(conversation):
    conversation.utterances[1].discourse_act = "answer"
    conversation.utterances[1].quality_score = 0.8
    report = analyze(conversation)
    evidence = report.evidence["reddit:t1_storage"]
    assert evidence.argument_quality_score == 0.8
    assert evidence.discourse_act == "answer"
    assert report.conversation_quality is None
    assert report.components["quality"].state == "unavailable"
    assert report.components["discourse"].state == "available"


def test_invalid_topic_assignment_isolated(conversation):
    class Clusterer:
        def assign(self, utterances):
            return {"reddit:unknown": "0"}

    report = analyze(conversation, clusterer=Clusterer())
    assert report.topic_assignments == {}
    assert report.components["clustering"].state == "failed"


def test_synthesis_cannot_mutate_original_evidence(conversation):
    class MutatingProvider:
        def synthesize(self, evidence, query):
            evidence[0].text_clean = "An invented assertion."
            return {"points": [{"text": evidence[0].text_clean, "evidence_ids": [evidence[0].id]}]}

    report = analyze(conversation, query="RAID", synthesis_provider=MutatingProvider())
    assert report.components["synthesis"].state == "failed"
    assert report.summary[0].text == conversation.utterances[1].text


def test_serialized_evidence_cannot_override_source_text(conversation):
    payload = json.loads(analyze(conversation).to_json())
    payload["evidence"]["reddit:t1_storage"]["text_clean"] = "An invented assertion."
    with pytest.raises(ValueError):
        InsightReport.model_validate(payload)
