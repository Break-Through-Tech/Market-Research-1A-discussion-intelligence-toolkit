"""Strict JSON report contracts and human-readable, locally linked Markdown."""

from __future__ import annotations

import html
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
        type(self).model_validate(self.model_dump())
        lines = [
            f"# {_escape(self.title or self.conversation_id)}",
            "",
            f"Conversation: {_code(self.conversation_id)}",
            "",
            "## Summary",
            "",
        ]

        if self.query is not None:
            lines[4:4] = [f"Query: {_escape(self.query)}", ""]

        def add_claims(claims: list[Claim]) -> None:
            if not claims:
                lines.extend(["No supported excerpts selected.", ""])
            for claim in claims:
                citations = " ".join(
                    f"[{_escape(uid)}](#evidence-{quote(uid, safe='')})"
                    for uid in claim.evidence_ids
                )
                lines.extend([f"> {_escape(claim.text)}", f"> {citations}", ""])

        add_claims(self.summary)
        for section in self.sections:
            lines.extend([f"## {_escape(section.heading)}", ""])
            add_claims(section.points)
        for title, claims in [
            ("Labeled agreement excerpts", self.agreements),
            ("Labeled disagreement excerpts", self.disagreements),
            ("Labeled question excerpts", self.questions),
        ]:
            if claims:
                lines.extend([f"## {title}", ""])
                add_claims(claims)
        if self.conversation_quality:
            q = self.conversation_quality
            citations = " ".join(
                f"[{_escape(uid)}](#evidence-{quote(uid, safe='')})" for uid in q.evidence_ids
            )
            lines.extend(
                [
                    "## Conversation civility estimate",
                    "",
                    f"P(civil): {q.value:.4f}; model: {_escape(q.model_name)}. {citations}",
                    "This is a conversation-level estimate of civility.",
                    "",
                ]
            )
        lines.extend(["## Component availability", ""])
        for name, status in self.components.items():
            lines.append(f"- **{_escape(name)}**: {status.state} — {_escape(status.detail)}")
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {_escape(item)}" for item in self.limitations)
        lines.extend(["", "## Source evidence", ""])
        for uid, evidence in self.evidence.items():
            lines.extend(
                [
                    f'<a id="evidence-{html.escape(uid, quote=True)}"></a>',
                    "",
                    f"### {_escape(uid)}",
                    "",
                    f"Source ID: {_code(evidence.source_id)}; "
                    f"parent: {_code(evidence.parent_id or 'root')}; "
                    f"speaker: {_code(evidence.speaker_id)}.",
                    "",
                ]
            )
            if evidence.discourse_act:
                lines.extend([f"Discourse label: {_code(evidence.discourse_act)}.", ""])
            if evidence.argument_quality_score is not None:
                lines.extend(
                    [f"Supplied argument-quality score: {evidence.argument_quality_score:.4f}.", ""]
                )
            for label, url in [
                ("Source", evidence.source_url),
                ("Conversation", evidence.conversation_url),
            ]:
                if url and urlsplit(url).scheme in {"http", "https"}:
                    lines.extend([f"[{label}](<{html.escape(url, quote=True)}>)", ""])
            lines.extend([f"> {_escape(evidence.text_clean)}", ""])
        return "\n".join(lines).rstrip()


def _escape(text: str) -> str:
    text = html.escape(text, quote=False).replace("\n", " ")
    for char in ("\\", "`", "*", "_", "[", "]", "|", "#"):
        text = text.replace(char, "\\" + char)
    return text


def _code(text: str) -> str:
    return f"<code>{html.escape(text)}</code>"
