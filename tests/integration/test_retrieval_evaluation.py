"""Metric/provenance checks; reference judgments are a curated prototype."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from discussion_intelligence import EvidenceIndex
from discussion_intelligence.retrieval import SearchHit
from discussion_intelligence.retrieval_evaluation import RetrievalJudgments, evaluate_retrieval
from discussion_intelligence.sample import load_reference_conversations

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/coarse-discourse-reference-sample.jsonl"
JUDGMENTS = ROOT / "examples/retrieval-judgments.json"


def reference_judgments():
    return RetrievalJudgments.model_validate_json(JUDGMENTS.read_text())


def test_curated_reference_quotes_ids_links_and_reproducible_measurements():
    judgments = reference_judgments()
    assert hashlib.sha256(SAMPLE.read_bytes()).hexdigest() == judgments.source_sha256
    conversations = load_reference_conversations(SAMPLE)
    before = [conv.model_dump_json() for conv in conversations]
    result = evaluate_retrieval(conversations, judgments)
    expected = json.loads((ROOT / "examples/retrieval-evaluation.json").read_text())
    # Preserve ranking/identity checks while allowing scientific-library float
    # roundoff across the supported Python/NumPy versions in the CI matrix.
    for row, recorded in zip(result["cases"], expected["cases"], strict=True):
        for hit, saved in zip(row["returned"], recorded["returned"], strict=True):
            assert hit["score"] == pytest.approx(saved["score"], abs=1e-12)
            saved["score"] = hit["score"]
    assert result == expected
    assert result["positive_queries"] == 8
    assert result["no_answer_queries"] == 2
    assert result["metrics"]["1"]["macro_recall"] == pytest.approx(35 / 48)
    assert result["metrics"]["2"]["macro_recall"] == pytest.approx(5 / 6)
    assert result["metrics"]["3"]["macro_recall"] == pytest.approx(7 / 8)
    assert result["metrics"]["3"]["no_answer_abstention_rate"] == 0.5
    assert [conv.model_dump_json() for conv in conversations] == before
    for case in result["cases"]:
        for hit in case["returned"]:
            assert hit["id"] == "reddit:" + hit["source_id"]
            assert hit["source_url"] is None  # The source supplies thread URLs only.
            assert hit["conversation_url"].startswith("https://www.reddit.com/")


def test_known_partial_recall_and_no_answer_denominators_with_injected_index(conversation):
    ids = [u.id for u in conversation.utterances]
    evidence = EvidenceIndex(conversation).evidence

    def relevant(uid):
        return {"id": uid, "quote": evidence[uid].text_clean, "rationale": "Synthetic metric check"}

    judgments = RetrievalJudgments(
        name="Synthetic metric arithmetic",
        scope="Unit test only",
        curation_note="Invented setup",
        source_sha256="0" * 64,
        cases=[
            {
                "id": "two",
                "conversation_id": conversation.id,
                "query": "two",
                "rationale": "test",
                "relevant": [relevant(ids[0]), relevant(ids[1])],
            },
            {
                "id": "one",
                "conversation_id": conversation.id,
                "query": "one",
                "rationale": "test",
                "relevant": [relevant(ids[2])],
            },
            {
                "id": "none",
                "conversation_id": conversation.id,
                "query": "none",
                "rationale": "test",
                "relevant": [],
            },
        ],
    )

    class StubIndex:
        def search(self, query, *, top_k):
            ranked = {"two": ids[:2], "one": [ids[2]], "none": []}[query]
            return [SearchHit(evidence=evidence[uid], score=1) for uid in ranked[:top_k]]

    result = evaluate_retrieval(
        [conversation], judgments, index_factory=lambda conv: StubIndex(), retriever_name="stub"
    )
    assert result["metrics"]["1"] == {"macro_recall": 0.75, "no_answer_abstention_rate": 1}
    assert result["metrics"]["2"]["macro_recall"] == 1
    assert result["cases"][2]["recall_at_k"]["1"] is None


@pytest.mark.parametrize("kind", ["unknown", "quote", "duplicate", "conversation"])
def test_bad_judgments_are_rejected(kind):
    judgments = reference_judgments()
    if kind == "unknown":
        judgments.cases[0].relevant[0].id = "reddit:missing"
    elif kind == "quote":
        judgments.cases[0].relevant[0].quote = "An invented source quotation"
    elif kind == "duplicate":
        judgments.cases[0].relevant.append(judgments.cases[0].relevant[0])
    else:
        judgments.cases[0].conversation_id = "reddit:missing"
    with pytest.raises(ValueError):
        evaluate_retrieval(load_reference_conversations(SAMPLE), judgments)


@pytest.mark.parametrize("ks", [[], [0], [-1], [True], [1.5]])
def test_invalid_cutoffs_are_rejected(ks):
    with pytest.raises(ValueError, match="positive integers"):
        evaluate_retrieval(load_reference_conversations(SAMPLE), reference_judgments(), ks=ks)


@pytest.mark.parametrize("field", ["source_id", "source_url", "text", "id"])
def test_injected_retriever_cannot_change_sources(field):
    class CorruptIndex:
        def __init__(self, conv):
            self.index = EvidenceIndex(conv)

        def search(self, query, *, top_k):
            hits = self.index.search(query, top_k=top_k)
            if hits:
                setattr(hits[0].evidence, field, "invented")
            return hits

    with pytest.raises(ValueError, match="Retriever"):
        evaluate_retrieval(
            load_reference_conversations(SAMPLE), reference_judgments(), index_factory=CorruptIndex
        )


def test_deleted_reference_mutation_is_separate_robustness_check():
    # The actual 11-utterance reference corpus has no deleted content. This
    # controlled mutation is NOT included in the reported curated measurements.
    conversations = load_reference_conversations(SAMPLE)
    target = next(u for u in conversations[0].utterances if u.id == "reddit:t1_c6fwalq")
    target.is_deleted = True
    assert target.id not in EvidenceIndex(conversations[0]).evidence
    with pytest.raises(ValueError, match="usable source support"):
        evaluate_retrieval(conversations, reference_judgments())


@pytest.mark.parametrize("format", ["json", "markdown"])
def test_evaluation_example_runs(format):
    result = subprocess.run(
        [sys.executable, "examples/evaluate_retrieval.py", "--format", format],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if format == "json":
        assert json.loads(result.stdout)["metrics"]["3"]["macro_recall"] == 0.875
    else:
        assert "NOT a held-out" in result.stdout and "0.8750" in result.stdout


def test_evaluation_example_rejects_changed_input(tmp_path):
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(SAMPLE.read_bytes() + b"\n")
    result = subprocess.run(
        [sys.executable, "examples/evaluate_retrieval.py", "--input", str(changed)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "SHA-256 differs" in result.stderr


@pytest.mark.parametrize("only_positive", [True, False])
def test_empty_metric_denominator_is_explicit_and_example_renders_it(tmp_path, only_positive):
    judgments = reference_judgments()
    judgments.cases = [case for case in judgments.cases if bool(case.relevant) == only_positive]
    result = evaluate_retrieval(load_reference_conversations(SAMPLE), judgments)
    undefined = "no_answer_abstention_rate" if only_positive else "macro_recall"
    assert result["metrics"]["1"][undefined] is None
    subset = tmp_path / "subset.json"
    subset.write_text(judgments.model_dump_json())
    rendered = subprocess.run(
        [sys.executable, "examples/evaluate_retrieval.py", "--judgments", str(subset)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "n/a" in rendered.stdout


@pytest.mark.parametrize("kind", ["repeated", "too_many"])
def test_evaluation_rejects_invalid_ranked_lists(kind):
    class InvalidIndex:
        def __init__(self, conv):
            self.evidence = list(EvidenceIndex(conv).evidence.values())

        def search(self, query, *, top_k):
            selected = [self.evidence[0]] * 2 if kind == "repeated" else self.evidence[: top_k + 1]
            return [SearchHit(evidence=e, score=1) for e in selected]

    with pytest.raises(ValueError, match="repeated IDs or exceeded"):
        evaluate_retrieval(
            load_reference_conversations(SAMPLE), reference_judgments(), index_factory=InvalidIndex
        )
