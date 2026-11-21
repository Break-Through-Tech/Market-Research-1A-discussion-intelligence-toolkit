"""Build/install regression: future owned subpackages must ship in the wheel."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_wheel_contains_future_subpackages_and_cli_runs_outside_checkout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ["pyproject.toml", "README.md"]:
        shutil.copy2(ROOT / name, source / name)
    for name in ["src", "artifacts", "utils"]:
        shutil.copytree(
            ROOT / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    # Synthetic package-discovery sentinel, not Gia's implementation. This
    # reproduces the omitted-subpackage failure without copying unmerged code.
    sentinel = source / "utils/integration_fixture"
    sentinel.mkdir()
    (sentinel / "__init__.py").write_text("TOKEN = 'package-discovery-check'\n")
    artifact_dir = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--offline", "--wheel", "--out-dir", str(artifact_dir)],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    wheel = next(artifact_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "utils/integration_fixture/__init__.py" in names
        assert "utils/modeling_data/normalize.py" in names
        assert "discussion_intelligence/cli.py" in names
        assert "artifacts/schema.py" in names
        assert not any(
            name.startswith(("tests/", "experiments/", "data/", "src/")) for name in names
        )
    # In particular, inherited PYTHONPATH must not make this a source-tree test.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    # Reuse the test environment's locked dependencies. Resolving a wheel's
    # dependencies again offline can require uncached registry metadata even
    # after uv sync downloaded every locked wheel (notably PR29's SDK).
    installed = tmp_path / "installed"
    subprocess.run(
        ["uv", "pip", "install", "--offline", "--no-deps", "--target", str(installed), str(wheel)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    bootstrap = (
        f"import sys; sys.path.insert(0, {str(installed)!r}); "
        "from pathlib import Path; import discussion_intelligence as d; "
        f"assert Path(d.__file__).is_relative_to(Path({str(installed)!r})); "
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            bootstrap + "from utils.integration_fixture import TOKEN; print(TOKEN)",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "package-discovery-check"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            bootstrap
            + "from importlib.metadata import distribution; entry = next(e for e in distribution('discussion-intelligence-toolkit').entry_points if e.name == 'discussion-intelligence'); raise SystemExit(entry.load()())",
            "--input",
            str(ROOT / "data/coarse-discourse-reference-sample.jsonl"),
            "--query",
            "RAID data drives",
            "--top-k",
            "3",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["summary"][0]["evidence_ids"] == ["reddit:t1_c6fwalq"]
    assert report["components"]["quality"]["state"] == "unavailable"
