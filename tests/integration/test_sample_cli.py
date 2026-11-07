import json
from pathlib import Path

import pytest

from discussion_intelligence import analyze
from discussion_intelligence.cli import main
from discussion_intelligence.sample import adapt_reference, load_reference_conversations
from utils.modeling_data.normalize import MISSING_TIMESTAMP

SAMPLE = Path(__file__).resolve().parents[2] / "data/coarse-discourse-reference-sample.jsonl"


def reference():
    return json.loads(SAMPLE.read_text(encoding="utf-8").splitlines()[0])


def test_real_reference_normalizes_without_faking_labels_dates_or_authors():
    conversations = load_reference_conversations(SAMPLE)
    assert len(conversations) == 2
    assert [len(conv) for conv in conversations] == [7, 4]
    for conv in conversations:
        assert conv.created_at == MISSING_TIMESTAMP
        assert all(u.metadata["timestamp_missing"] for u in conv.utterances)
        assert all(u.metadata["speaker_missing"] for u in conv.utterances)
        assert all(u.discourse_act is None for u in conv.utterances)
        assert not any(s.is_deleted for s in conv.speakers.values())
        report = analyze(conv)
        assert report.summary
        assert all(e.created_at is None for e in report.evidence.values())
        assert report.source_url == conv.url
        assert any("Source timestamps are missing" in item for item in report.limitations)


def test_reference_orphan_repair_preserves_original_parent():
    record = reference()
    record["utterances"][1]["parent_id"] = "t1_missing"
    conv = adapt_reference(record)
    reply = next(u for u in conv.utterances if u.source_id == "t1_c6fwalq")
    assert reply.parent_id == conv.root().id
    assert reply.metadata["source_parent_id"] == "t1_missing"
    assert reply.metadata["orphan_reparented"] is True


@pytest.mark.parametrize(
    "kind", ["duplicate", "missing_root", "invalid_id", "cycle", "wrong_source"]
)
def test_bad_reference_rejected(kind):
    record = reference()
    if kind == "duplicate":
        record["utterances"].append(record["utterances"][1])
    elif kind == "missing_root":
        record["utterances"].pop(0)
    elif kind == "invalid_id":
        record["utterances"][1]["id"] = ""
    elif kind == "cycle":
        record["utterances"][1]["parent_id"] = record["utterances"][2]["id"]
    else:
        record["source"] = "other"
    with pytest.raises(ValueError):
        adapt_reference(record)


def test_present_reference_timestamp_and_source_url_preserved():
    record = reference()
    record["utterances"][0]["timestamp"] = 100
    record["utterances"][0]["speaker"] = "original-op"
    record["utterances"][0]["url"] = "https://example.com/source"
    conv = adapt_reference(record)
    assert conv.created_at.timestamp() == 100
    assert not conv.root().metadata["speaker_missing"]
    report = analyze(conv)
    assert report.evidence[conv.root().id].created_at == conv.created_at
    assert report.evidence[conv.root().id].source_url == "https://example.com/source"


def test_malformed_jsonl_has_line_context(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("\nnot-json\n")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        load_reference_conversations(path)


@pytest.mark.parametrize("format", ["json", "markdown"])
def test_cli_real_sample_output(tmp_path, format):
    output = tmp_path / "reports" / f"report.{format}"
    assert (
        main(
            [
                "--input",
                str(SAMPLE),
                "--query",
                "RAID drives",
                "--format",
                format,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    text = output.read_text()
    if format == "json":
        report = json.loads(text)
        assert report["summary"][0]["evidence_ids"] == ["reddit:t1_c6fwalq"]
    else:
        assert "# Summary" in text and "#evidence-reddit%3At1_c6fwalq" in text


def test_cli_canonical_and_stdout(tmp_path, capsys, conversation):
    path = tmp_path / "canonical.json"
    path.write_text(conversation.model_dump_json())
    assert main(["--input", str(path), "--canonical", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["conversation_id"] == conversation.id


@pytest.mark.parametrize(
    "args", [["--conversation-index", "-1"], ["--conversation-index", "5"], ["--top-k", "-1"]]
)
def test_cli_bad_arguments_are_clear(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--input", str(SAMPLE), *args])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
