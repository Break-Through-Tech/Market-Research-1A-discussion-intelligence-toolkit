"""Strict JSON report contracts and human-readable, locally linked Markdown."""

from __future__ import annotations

import html
from collections import Counter
from datetime import datetime
from typing import Any, Literal
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, revalidate_instances="always")


class ComponentStatus(ReportModel):
    state: Literal["available", "unavailable", "failed"]
    detail: str


class Evidence(ReportModel):
    id: str
    conversation_id: str
    source_id: str
    parent_id: str | None
    speaker_id: str
    text: str
    text_clean: str
    source_url: str | None = None
    conversation_url: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    discourse_act: str | None = None
    argument_quality_score: float | None = Field(default=None, ge=0, le=1)


class Claim(ReportModel):
    """An extractive discussion point, not an assertion of objective truth."""

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class Section(ReportModel):
    heading: str
    points: list[Claim]


class ConversationQuality(ReportModel):
    scope: Literal["conversation"] = "conversation"
    target: Literal["civility_probability"] = "civility_probability"
    value: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)
    model_name: str


class SynthesisDraft(ReportModel):
    """Injected synthesis may select quotations; free-form claims are rejected."""

    points: list[Claim] = Field(min_length=1)


class InsightReport(ReportModel):
    report_version: Literal["1.0"] = "1.0"
    conversation_id: str
    conversation_source_id: str
    title: str | None
    source_url: str | None
    query: str | None
    summary: list[Claim]
    sections: list[Section]
    agreements: list[Claim] = Field(default_factory=list)
    disagreements: list[Claim] = Field(default_factory=list)
    questions: list[Claim] = Field(default_factory=list)
    open_questions: list[Claim] = Field(default_factory=list)
    evidence: dict[str, Evidence]
    conversation_quality: ConversationQuality | None = None
    topic_assignments: dict[str, str] = Field(default_factory=dict)
    components: dict[str, ComponentStatus]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_grounding(self) -> InsightReport:
        for uid, evidence in self.evidence.items():
            if uid != evidence.id or evidence.conversation_id != self.conversation_id:
                raise ValueError("Evidence identity/conversation mismatch")
            if evidence.text_clean != " ".join(evidence.text.split()) or not evidence.text_clean:
                raise ValueError("Evidence cleaned text must preserve the source text")
        claims = [
            *self.summary,
            *self.agreements,
            *self.disagreements,
            *self.questions,
            *self.open_questions,
        ]
        claims.extend(point for section in self.sections for point in section.points)
        for claim in claims:
            if len(set(claim.evidence_ids)) != len(claim.evidence_ids):
                raise ValueError("Repeated evidence ID in claim")
            # Every attached citation must contain the quote. Merely citing a
            # real ID cannot establish that an arbitrary generated claim is true.
            if not claim.text.strip() or any(
                uid not in self.evidence or claim.text not in self.evidence[uid].text_clean
                for uid in claim.evidence_ids
            ):
                raise ValueError("Claim is not an exact excerpt of its cited evidence")
        if set(self.topic_assignments) - self.evidence.keys():
            raise ValueError("Unknown topic-assignment evidence ID")
        if self.conversation_quality and any(
            uid not in self.evidence for uid in self.conversation_quality.evidence_ids
        ):
            raise ValueError("Unknown conversation-quality evidence ID")
        return self

    def to_json(self, *, indent: int = 2) -> str:
        # Revalidate mutable nested containers before either serialization path.
        validated = type(self).model_validate(self.model_dump())
        return validated.model_dump_json(indent=indent)

    def to_markdown(self) -> str:
        """Render the reader-facing layout documented in docs/report-format.md."""
        type(self).model_validate(self.model_dump())
        lines = [f"# {_escape(self.title or self.conversation_id)}", ""]
        if self.source_url and urlsplit(self.source_url).scheme in {"http", "https"}:
            lines.extend([f"Source: [link](<{html.escape(self.source_url, quote=True)}>)", ""])
        lines.extend([f"Conversation: {_code(self.conversation_id)}", ""])
        if self.query is not None:
            lines.extend([f"Query: {_escape(self.query)}", ""])

        lines.extend(["## At a glance", "", "| | |", "|---|---|"])
        lines.extend(f"| {name} | {value} |" for name, value in self._overview())
        lines.extend(["", "## Key points", ""])
        self._add_claims(lines, self.summary)

        if self.topic_assignments:
            sizes = Counter(self.topic_assignments.values())
            themes = {f"Topic cluster {cluster}": cluster for cluster in sizes}
            lines.extend(["## Themes", ""])
            for section in self.sections:
                heading = _escape(section.heading)
                if section.heading in themes:
                    cluster = themes[section.heading]
                    heading = (
                        f"Theme {_escape(cluster)} "
                        f"({sizes[cluster]} of {len(self.evidence)} comments)"
                    )
                elif section.heading == "Discussion excerpts":
                    heading = "Not assigned to a theme"
                lines.extend([f"### {heading}", ""])
                self._add_claims(lines, section.points)
        else:
            # Without themes the sections would only repeat the key points.
            rest = [
                point
                for section in self.sections
                for point in section.points
                if point not in self.summary
            ]
            if rest:
                lines.extend(["## More excerpts", ""])
                self._add_claims(lines, rest)

        labeled = sum(e.discourse_act is not None for e in self.evidence.values())
        if labeled:
            lines.extend(
                [
                    "## Agreement, disagreement and questions",
                    "",
                    "A label describes the role of one comment. It does not show that the "
                    "thread reached consensus or that a question was left unanswered.",
                    "",
                ]
            )
            for title, claims in [
                ("Agreement", self.agreements),
                ("Disagreement", self.disagreements),
                ("Questions", self.questions),
            ]:
                lines.extend([f"### {title}", ""])
                if claims:
                    self._add_claims(lines, claims)
                else:
                    lines.extend(["None among the selected excerpts.", ""])
        if self.open_questions:
            lines.extend(["## Open questions", ""])
            self._add_claims(lines, self.open_questions)

        lines.extend(
            ["## Component availability", "", "| Component | Status | Detail |", "|---|---|---|"]
        )
        for name, status in self.components.items():
            lines.append(f"| {_escape(name)} | {status.state} | {_escape(status.detail)} |")
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {_escape(item)}" for item in self.limitations)

        shown = self._cited_ids()
        shown |= {
            self.evidence[uid].parent_id
            for uid in shown
            if self.evidence[uid].parent_id in self.evidence
        }
        lines.extend(["", "## Source evidence", ""])
        if len(shown) < len(self.evidence):
            lines.extend(
                [
                    f"Showing {len(shown)} of {len(self.evidence)} analyzed comments: each quoted "
                    "comment and the comment it replies to. The JSON report contains all of them.",
                    "",
                ]
            )
        for uid in self._thread_order():
            if uid not in shown:
                continue
            evidence = self.evidence[uid]
            parent = evidence.parent_id
            if parent in shown:
                parent_ref = f"reply to {_cite(parent)}"
            else:
                parent_ref = f"reply to {_code(parent)}" if parent else "thread root"
            if evidence.metadata.get("speaker_missing"):
                details = [parent_ref, "speaker not recorded"]
            else:
                details = [parent_ref, f"speaker {_code(evidence.speaker_id)}"]
            if evidence.created_at is not None:
                details.append(evidence.created_at.date().isoformat())
            if evidence.discourse_act:
                details.append(f"label {_code(evidence.discourse_act)}")
            if uid in self.topic_assignments:
                details.append(f"theme {_code(self.topic_assignments[uid])}")
            if evidence.argument_quality_score is not None:
                details.append(f"argument-quality score {evidence.argument_quality_score:.2f}")
            for label, url in [
                ("source", evidence.source_url),
                ("conversation", evidence.conversation_url),
            ]:
                if url and urlsplit(url).scheme in {"http", "https"}:
                    details.append(f"[{label}](<{html.escape(url, quote=True)}>)")
            lines.extend(
                [
                    f'<a id="evidence-{html.escape(uid, quote=True)}"></a>',
                    "",
                    f"### {_escape(uid)}",
                    "",
                    " · ".join(details),
                    "",
                    f"> {_escape(evidence.text_clean)}",
                    "",
                ]
            )
        return "\n".join(lines).rstrip()

    def _cited_ids(self) -> set[str]:
        claims = [
            *self.summary,
            *self.agreements,
            *self.disagreements,
            *self.questions,
            *self.open_questions,
        ]
        claims.extend(point for section in self.sections for point in section.points)
        return {uid for claim in claims for uid in claim.evidence_ids}

    def _thread_order(self) -> list[str]:
        """Evidence IDs with every comment placed before its replies."""
        replies: dict[str | None, list[str]] = {}
        for uid, evidence in self.evidence.items():
            parent = evidence.parent_id if evidence.parent_id in self.evidence else None
            replies.setdefault(parent, []).append(uid)
        ordered: list[str] = []
        stack = list(reversed(replies.get(None, [])))
        while stack:
            uid = stack.pop()
            ordered.append(uid)
            stack.extend(reversed(replies.get(uid, [])))
        return ordered

    def _overview(self) -> list[tuple[str, str]]:
        """One row per result, so a missing result is stated instead of dropped."""

        def missing(component: str) -> str:
            status = self.components.get(component)
            return "failed" if status and status.state == "failed" else "not available"

        evidence = list(self.evidence.values())
        total = len(evidence)
        dates = sorted(e.created_at.date() for e in evidence if e.created_at is not None)
        if not dates:
            period = "not recorded in the source"
        elif dates[0] == dates[-1]:
            period = dates[0].isoformat()
        else:
            period = f"{dates[0].isoformat()} to {dates[-1].isoformat()}"
        if any(e.metadata.get("speaker_missing") for e in evidence):
            participants = "not recorded in the source"
        else:
            participants = str(len({e.speaker_id for e in evidence}))
        labeled = sum(e.discourse_act is not None for e in evidence)
        themes = len(set(self.topic_assignments.values()))
        quality = self.conversation_quality
        return [
            ("Comments analyzed", str(total)),
            ("Comments quoted", str(len(self._cited_ids()))),
            ("Participants", participants),
            ("Dates", period),
            (
                "Themes",
                f"{themes} ({len(self.topic_assignments)} of {total} comments assigned)"
                if themes
                else missing("clustering"),
            ),
            (
                "Discourse labels",
                f"{labeled} of {total} comments" if labeled else missing("discourse"),
            ),
            (
                "Civility estimate",
                f"{quality.value:.2f} probability that the thread stays civil "
                f"(model {_code(quality.model_name).replace('|', '&#124;')}, from {len(quality.evidence_ids)} comments)"
                if quality
                else missing("quality"),
            ),
        ]

    def _add_claims(self, lines: list[str], claims: list[Claim]) -> None:
        if not claims:
            lines.extend(["No supported excerpts selected.", ""])
        for claim in claims:
            # Mark a quotation that starts or stops inside the source comment.
            source = self.evidence[claim.evidence_ids[0]].text_clean
            start = source.find(claim.text)
            text = _escape(claim.text)
            if start > 0:
                text = "… " + text
            if start + len(claim.text) < len(source):
                text += " …"
            citations = ", ".join(_cite(uid) for uid in claim.evidence_ids)
            lines.extend([f"> {text}", ">", f"> Source: {citations}", ""])


def _cite(uid: str) -> str:
    return f"[{_escape(uid)}](#evidence-{quote(uid, safe='')})"


def _escape(text: str) -> str:
    text = html.escape(text, quote=False).replace("\n", " ")
    for char in ("\\", "`", "*", "_", "[", "]", "|", "#"):
        text = text.replace(char, "\\" + char)
    return text


def _code(text: str) -> str:
    return f"<code>{html.escape(text)}</code>"
