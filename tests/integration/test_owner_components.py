"""Owner-code compatibility checks; run when the relevant PR code is present.

Missing owner modules are explicit skips on the standalone integration branch.
Synthetic setup and mocked responses are NOT production model/LLM results.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from discussion_intelligence import EvidenceIndex, analyze
from discussion_intelligence.cli import main
from discussion_intelligence.sample import load_reference_conversations

SAMPLE = Path(__file__).resolve().parents[2] / "data/coarse-discourse-reference-sample.jsonl"


def test_gia_fitted_scorer_bundle_through_cli(tmp_path, capsys):
    owner = pytest.importorskip(
        "utils.quality.scorers", reason="Gia's PR #22 code is not installed"
    )
    # Fitting belongs only to invented test setup, never CLI inference.
    scorer = owner.ConstantScorer().fit([["synthetic civil"], ["synthetic derailed"]], [1, 0])
    bundle = tmp_path / "synthetic-owner-interface.joblib"
    joblib.dump(
        {"civility_model": scorer, "civility_model_name": "synthetic-interface-only"}, bundle
    )
    assert (
        main(
            [
                "--input",
                str(SAMPLE),
                "--query",
                "RAID data drives",
                "--trusted-bundle",
                str(bundle),
                "--format",
                "json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["components"]["quality"]["state"] == "available"
    assert report["conversation_quality"]["scope"] == "conversation"
    assert report["conversation_quality"]["value"] == 0.5
    assert report["summary"][0]["evidence_ids"] == ["reddit:t1_c6fwalq"]
    assert all(e["argument_quality_score"] is None for e in report["evidence"].values())


@pytest.mark.parametrize("mode", ["quote", "fabricated", "error"])
def test_aisha_provider_on_real_retrieved_source_without_api(mode):
    owner = pytest.importorskip(
        "discussion_intelligence.llm_synthesis", reason="Aisha's PR #29 code is not installed"
    )
    conv = load_reference_conversations(SAMPLE)[0]
    evidence = EvidenceIndex(conv).search("RAID data drives", top_k=1)[0].evidence
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        assert evidence.id in kwargs["input"] and evidence.text_clean in kwargs["input"]
        assert "reddit:t1_c6fwdkh" not in kwargs["input"]  # Unretrieved source is absent.
        if mode == "error":
            raise RuntimeError("mock provider failure")
        text = (
            "All participants agree with this product."
            if mode == "fabricated"
            else evidence.text_clean
        )
        return SimpleNamespace(
            output_text=json.dumps({"points": [{"text": text, "evidence_ids": [evidence.id]}]})
        )

    # Use Aisha's actual factory with an injected client; no OpenAI client is
    # constructed, no credentials are read and no external model is called.
    provider = owner.create_openai_provider(
        client=SimpleNamespace(responses=SimpleNamespace(create=create))
    )
    report = analyze(conv, query="RAID data drives", top_k=1, synthesis_provider=provider)
    assert len(calls) == 1
    assert report.components["synthesis"].state == ("available" if mode == "quote" else "failed")
    assert report.summary[0].evidence_ids == [evidence.id]
    if mode != "quote":
        assert report.summary == analyze(conv, query="RAID data drives", top_k=1).summary
    assert "All participants agree" not in report.to_json()
