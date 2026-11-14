import json
import subprocess
import sys
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier

from discussion_intelligence.reports import InsightReport

ROOT = Path(__file__).resolve().parents[2]


def test_two_real_threads_generate_grounded_reports_and_preserve_existing_output(tmp_path):
    output = tmp_path / "reports"
    command = [sys.executable, "examples/end_to_end_demo.py", "--output-dir", str(output)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert "reddit:t3_10r2f8" in result.stdout and "reddit:t3_169c79" in result.stdout
    reports = [
        InsightReport.model_validate_json((output / f"thread-{i}.json").read_text())
        for i in range(2)
    ]
    assert [len(report.evidence) for report in reports] == [7, 4]
    assert reports[0].summary[0].evidence_ids == ["reddit:t1_c6fwalq"]
    for i, report in enumerate(reports):
        assert (output / f"thread-{i}.md").read_text() == report.to_markdown() + "\n"
        assert report.components["discourse"].state == "unavailable"
        assert all(e.discourse_act is None for e in report.evidence.values())
        assert not report.open_questions
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    rerun = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert rerun.returncode == 2
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_real_thread_demo_uses_explicit_synthetic_fitted_bundle(tmp_path):
    # Interface check only; no production classifier result is asserted.
    model = DummyClassifier(strategy="constant", constant="answer").fit([[0]], ["answer"])
    bundle = tmp_path / "synthetic-demo.joblib"
    joblib.dump({"discourse_model": model}, bundle)
    output = tmp_path / "reports"
    subprocess.run(
        [
            sys.executable,
            "examples/end_to_end_demo.py",
            "--output-dir",
            str(output),
            "--trusted-bundle",
            str(bundle),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for i in range(2):
        report = json.loads((output / f"thread-{i}.json").read_text())
        assert report["components"]["discourse"]["state"] == "available"
        assert report["components"]["quality"]["state"] == "unavailable"
        assert all(e["discourse_act"] == "answer" for e in report["evidence"].values())
