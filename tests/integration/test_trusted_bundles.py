"""Synthetic fitted-object interface checks, NOT production model results."""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from discussion_intelligence import analyze
from discussion_intelligence.bundles import load_trusted_bundle
from discussion_intelligence.cli import main
from discussion_intelligence.reports import InsightReport
from discussion_intelligence.sample import load_reference_conversations

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/coarse-discourse-reference-sample.jsonl"


@pytest.fixture
def synthetic_bundle(tmp_path):
    # Training occurs only in test setup on invented text, never in the CLI.
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(random_state=0))
    model.fit(
        ["RAID data drives", "disk storage answer", "agree graphics gaming", "agree blue card"],
        ["answer", "answer", "agreement", "agreement"],
    )
    bundle = tmp_path / "synthetic-interface-check.joblib"
    joblib.dump({"discourse_model": model}, bundle)
    return bundle, model


def test_cli_without_opt_in_never_loads_bundle(monkeypatch, capsys):
    def unexpected_load(*args, **kwargs):
        pytest.fail("No artifact may be deserialized without --trusted-bundle")

    monkeypatch.setattr(joblib, "load", unexpected_load)
    assert main(["--input", str(SAMPLE), "--format", "json"]) == 0
    report = InsightReport.model_validate_json(capsys.readouterr().out)
    assert report.to_json() == analyze(load_reference_conversations(SAMPLE)[0]).to_json()
    assert report.components["discourse"].state == "unavailable"
    assert all(e.discourse_act is None for e in report.evidence.values())


@pytest.mark.parametrize("kind", ["text", "context"])
def test_cli_fitted_predictions_preserve_sources_and_gold_isolation(
    kind, synthetic_bundle, tmp_path, conversation, monkeypatch
):
    bundle, model = synthetic_bundle
    texts = [" ".join(u.text.split()) for u in conversation.utterances]
    if kind == "context":
        parents = ["", texts[0], texts[0]]
        target = TfidfVectorizer().fit(texts)
        parent = TfidfVectorizer().fit(["synthetic parent", *parents])
        features = hstack([target.transform(texts), parent.transform(parents)], format="csr")
        model = LogisticRegression(random_state=0).fit(features, ["answer", "answer", "agreement"])
        joblib.dump(
            {"context_model": model, "target_vectorizer": target, "parent_vectorizer": parent},
            bundle,
        )
        expected = model.predict(features)
    else:
        expected = model.predict(texts)

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Fitted model inference must not train")

    monkeypatch.setattr(LogisticRegression, "fit", unexpected_fit)
    monkeypatch.setattr(Pipeline, "fit", unexpected_fit)
    monkeypatch.setattr(TfidfVectorizer, "fit", unexpected_fit)
    source = tmp_path / "canonical.json"
    original = conversation.model_dump_json()
    source.write_text(original, encoding="utf-8")
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--input",
                str(source),
                "--canonical",
                "--trusted-bundle",
                str(bundle),
                "--query",
                "data drives",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = InsightReport.model_validate_json(output.read_text())
    assert report.components["discourse"].detail == "Injected classifier predicted 3/3 labels."
    assert [report.evidence[u.id].discourse_act for u in conversation.utterances] == list(expected)
    for utterance in conversation.utterances:
        evidence = report.evidence[utterance.id]
        assert (evidence.source_id, evidence.parent_id, evidence.text, evidence.source_url) == (
            utterance.source_id,
            utterance.parent_id,
            utterance.text,
            utterance.metadata["source_url"],
        )
        assert evidence.metadata["discourse_act_gold"] == "question"
        assert evidence.discourse_act != "question"
    assert source.read_text() == original == conversation.model_dump_json()
    assert report.components["quality"].state == "unavailable"
    assert report.components["clustering"].state == "unavailable"
    assert report.components["embeddings"].state == "unavailable"
    assert report.summary
    assert "#evidence-reddit%3A" in report.to_markdown()
    report.summary[0].text = "An invented statement with a real citation"
    with pytest.raises(ValueError, match="exact excerpt"):
        report.to_json()
    with pytest.raises(ValueError, match="exact excerpt"):
        report.to_markdown()


def test_fitted_civility_bundle_stays_at_conversation_scope(tmp_path, conversation):
    model = DummyRegressor(strategy="constant", constant=0.37).fit([[0]], [0.37])
    bundle = tmp_path / "synthetic-civility.joblib"
    joblib.dump(
        {"civility_model": model, "civility_model_name": "synthetic-interface-only"}, bundle
    )
    report = analyze(conversation, **load_trusted_bundle(bundle))
    assert report.components["quality"].state == "available"
    assert report.components["discourse"].state == "unavailable"
    assert report.conversation_quality.value == pytest.approx(0.37)
    assert report.conversation_quality.model_name == "synthetic-interface-only"
    assert report.conversation_quality.evidence_ids == [u.id for u in conversation.utterances]
    assert all(e.argument_quality_score is None for e in report.evidence.values())


class AssignmentStub:
    def assign(self, utterances):
        return {u.id: "synthetic-interface-only" for u in utterances}


class EmbedderStub:
    def embed(self, texts):
        return np.ones((len(texts), 2))


def test_bundle_passes_existing_assign_and_embed_interfaces(tmp_path, conversation):
    bundle = tmp_path / "interfaces.joblib"
    joblib.dump({"topic_clusterer": AssignmentStub(), "embedder": EmbedderStub()}, bundle)
    report = analyze(conversation, **load_trusted_bundle(bundle))
    assert report.components["clustering"].state == "available"
    assert report.components["embeddings"].state == "available"
    assert set(report.topic_assignments) == {u.id for u in conversation.utterances}


@pytest.mark.parametrize(
    "payload, message",
    [
        ([1, 2], "must contain a mapping"),
        ({"context_model": None}, "Context bundle requires"),
        ({"target_vectorizer": None, "parent_vectorizer": None}, "Context bundle requires"),
    ],
)
def test_cli_rejects_malformed_bundle(tmp_path, payload, message, capsys):
    bundle = tmp_path / "malformed.joblib"
    joblib.dump(payload, bundle)
    with pytest.raises(SystemExit) as exc:
        main(["--input", str(SAMPLE), "--trusted-bundle", str(bundle)])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_cli_bundle_loading_error_is_clear(tmp_path, kind, capsys):
    bundle = tmp_path / "bad.joblib"
    if kind == "corrupt":
        bundle.write_bytes(b"not a joblib artifact")
    with pytest.raises(SystemExit) as exc:
        main(["--input", str(SAMPLE), "--trusted-bundle", str(bundle)])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_supplied_invalid_predictions_are_failed_not_gold(tmp_path, capsys):
    model = DummyClassifier(strategy="constant", constant="invalid-label").fit(
        [[0]], ["invalid-label"]
    )
    bundle = tmp_path / "invalid-predictions.joblib"
    joblib.dump({"discourse_model": model}, bundle)
    assert main(["--input", str(SAMPLE), "--trusted-bundle", str(bundle), "--format", "json"]) == 0
    report = InsightReport.model_validate_json(capsys.readouterr().out)
    assert report.components["discourse"].state == "failed"
    assert report.summary
    assert all(e.discourse_act is None for e in report.evidence.values())


def test_cli_help_discloses_deserialization_risk(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "deserialization can execute code" in " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("entrypoint", ["cli", "example"])
def test_executable_fitted_model_smoke(synthetic_bundle, entrypoint):
    bundle, _ = synthetic_bundle
    command = (
        [
            sys.executable,
            "-m",
            "discussion_intelligence",
            "--trusted-bundle",
            str(bundle),
            "--format",
            "json",
        ]
        if entrypoint == "cli"
        else [sys.executable, "examples/fitted_model_integration.py", "--bundle", str(bundle)]
    )
    result = subprocess.run(
        [*command, "--input", str(SAMPLE)], cwd=ROOT, capture_output=True, text=True, check=True
    )
    report = InsightReport.model_validate_json(result.stdout)
    assert report.components["discourse"].state == "available"
    assert report.conversation_id == "reddit:t3_10r2f8"
    assert len(report.evidence) == 7
    assert json.loads(report.to_json()) == json.loads(
        analyze(load_reference_conversations(SAMPLE)[0], **load_trusted_bundle(bundle)).to_json()
    )
