"""Reviewable report contract and mock synthesis wiring, NOT an LLM evaluation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from discussion_intelligence import analyze
from discussion_intelligence.reports import InsightReport
from discussion_intelligence.sample import load_reference_conversations
from examples.synthesis_provider_demo import MockQuoteProvider

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/coarse-discourse-reference-sample.jsonl"


def test_report_contract_snapshot_and_real_reference_example():
    schema = json.loads((ROOT / "examples/insight-report.schema.json").read_text())
    assert schema == InsightReport.model_json_schema()
    conv = load_reference_conversations(SAMPLE)[1]
    report = analyze(conv, query="comment karma easier", top_k=2)
    assert (ROOT / "examples/karma-report.md").read_text() == report.to_markdown() + "\n"
    assert not report.agreements and not report.disagreements and not report.open_questions
    assert report.conversation_quality is None
    assert all(e.discourse_act is None for e in report.evidence.values())


def test_mock_provider_only_receives_selected_evidence_and_preserves_it():
    conv = load_reference_conversations(SAMPLE)[0]
    observed = []

    class RecordingMock(MockQuoteProvider):
        def synthesize(self, evidence, query):
            observed.extend(e.id for e in evidence)
            assert query == "RAID data drives"
            return super().synthesize(evidence, query)

    report = analyze(conv, query="RAID data drives", top_k=1, synthesis_provider=RecordingMock())
    assert observed == ["reddit:t1_c6fwalq"]
    assert report.summary[0].evidence_ids == observed
    assert report.summary[0].text == report.evidence[observed[0]].text_clean


@pytest.mark.parametrize("mode", ["quote", "fabricated", "error"])
def test_executable_mock_provider_demo_and_safe_fallback(mode):
    result = subprocess.run(
        [sys.executable, "examples/synthesis_provider_demo.py", "--mode", mode, "--format", "json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    report = InsightReport.model_validate_json(result.stdout)
    assert "no LLM was executed" in report.limitations[-1]
    assert report.components["synthesis"].state == ("available" if mode == "quote" else "failed")
    conv = load_reference_conversations(SAMPLE)[0]
    if mode != "quote":
        assert report.summary == analyze(conv, query="RAID data drives", top_k=1).summary
    assert "All participants agree" not in report.to_json()
    assert report.summary[0].evidence_ids == ["reddit:t1_c6fwalq"]
    assert all(e.discourse_act is None for e in report.evidence.values())
