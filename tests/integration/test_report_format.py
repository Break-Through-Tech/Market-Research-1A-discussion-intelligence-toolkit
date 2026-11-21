"""The reader-facing Markdown layout described in docs/report-format.md."""

import re
from pathlib import Path

from discussion_intelligence import analyze
from discussion_intelligence.reports import Claim
from discussion_intelligence.sample import load_reference_conversations
from examples.report_format_demo import build_report

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/coarse-discourse-reference-sample.jsonl"


def headings(markdown, level="## "):
    return [line[len(level) :] for line in markdown.splitlines() if line.startswith(level)]


def glance(markdown):
    rows = re.findall(r"^\| ([^|]+) \| (.+) \|$", markdown.split("## Key points")[0], re.M)
    return dict(rows)


def test_full_example_is_current_and_sections_keep_their_order():
    report = build_report()
    markdown = report.to_markdown()
    assert (ROOT / "examples/full-report-example.md").read_text() == markdown + "\n"
    assert headings(markdown) == [
        "At a glance",
        "Key points",
        "Themes",
        "Agreement, disagreement and questions",
        "Component availability",
        "Limitations",
        "Source evidence",
    ]
    assert "### Theme 1 (3 of 7 comments)" in markdown
    assert "### Not assigned to a theme" in markdown
    assert glance(markdown)["Themes"] == "2 (6 of 7 comments assigned)"
    assert glance(markdown)["Discourse labels"] == "7 of 7 comments"


def test_missing_results_are_stated_and_empty_sections_are_left_out(conversation):
    markdown = analyze(conversation).to_markdown()
    assert headings(markdown) == [
        "At a glance",
        "Key points",
        "Component availability",
        "Limitations",
        "Source evidence",
    ]
    overview = glance(markdown)
    assert overview["Themes"] == overview["Civility estimate"] == "not available"
    assert overview["Discourse labels"] == "not available"
    assert overview["Participants"] == "1"
    assert overview["Dates"] == "2026-01-01"


def test_failed_component_is_reported_as_failed(conversation):
    class Broken:
        def assign(self, utterances):
            raise RuntimeError("boom")

    markdown = analyze(conversation, clusterer=Broken()).to_markdown()
    assert glance(markdown)["Themes"] == "failed"
    assert "| clustering | failed |" in markdown


def test_missing_speakers_and_dates_are_not_counted():
    report = analyze(load_reference_conversations(SAMPLE)[0], top_k=5)
    markdown = report.to_markdown()
    overview = glance(markdown)
    assert overview["Participants"] == overview["Dates"] == "not recorded in the source"
    assert "speaker not recorded" in markdown
    assert "anon_" not in markdown


def test_excerpts_are_not_repeated_without_themes():
    report = analyze(load_reference_conversations(SAMPLE)[0], top_k=5)
    assert len(report.summary) == 3 and len(report.sections[0].points) == 5
    markdown = report.to_markdown()
    assert "## More excerpts" in markdown
    assert markdown.count("> Source: ") == 5
    assert (
        "## More excerpts"
        not in analyze(load_reference_conversations(SAMPLE)[0], top_k=3).to_markdown()
    )


def test_cut_quotations_are_marked(conversation):
    report = analyze(conversation, query="RAID storage", top_k=1)
    uid = report.summary[0].evidence_ids[0]
    full = report.evidence[uid].text_clean
    assert f"> {full}\n>\n> Source: " in report.to_markdown()
    report.summary = [Claim(text=full[5:-9], evidence_ids=[uid])]
    report.sections = []
    assert f"> … {full[5:-9]} …\n" in report.to_markdown()


def test_markdown_shows_quoted_comments_with_parents_and_json_keeps_all():
    report = analyze(load_reference_conversations(SAMPLE)[0], query="higher base clock", top_k=1)
    (quoted,) = report.summary[0].evidence_ids
    parent = report.evidence[quoted].parent_id
    markdown = report.to_markdown()
    shown = re.findall(r'<a id="evidence-([^"]+)"></a>', markdown)
    assert shown == [parent, quoted]
    assert "Showing 2 of 7 analyzed comments" in markdown
    assert len(report.evidence) == 7 and report.to_json().count('"source_id"') == 7
