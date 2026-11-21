"""The public analyze() pipeline: validate, enrich, retrieve, extract and cite."""

from __future__ import annotations

from collections.abc import Mapping

from artifacts.schema import Conversation

from .adapters import (
    DiscourseClassifier,
    InferenceUtterance,
    QualityScorer,
    SynthesisProvider,
    TopicClusterer,
)
from .reports import (
    Claim,
    ComponentStatus,
    ConversationQuality,
    InsightReport,
    Section,
    SynthesisDraft,
)
from .retrieval import EvidenceIndex, TextEmbedder
from .validation import usable_text, validate_conversation

DISCOURSE_LABELS = frozenset(
    {
        "question",
        "answer",
        "agreement",
        "disagreement",
        "humor",
        "announcement",
        "appreciation",
        "negative_reaction",
        "elaboration",
        "other",
    }
)


def _status(state, detail):
    return ComponentStatus(state=state, detail=detail)


def _check_assignments(values, allowed_ids, *, labels=None):
    if not isinstance(values, Mapping) or set(values) - allowed_ids:
        raise ValueError("Component returned unknown utterance IDs or a non-mapping result")
    if any(
        not isinstance(v, str) or not v.strip() or (labels and v not in labels)
        for v in values.values()
    ):
        raise ValueError("Component returned an invalid label/cluster ID")
    return dict(values)


def analyze(
    conversation: Conversation,
    *,
    query: str | None = None,
    top_k: int = 5,
    classifier: DiscourseClassifier | None = None,
    quality_scorer: QualityScorer | None = None,
    clusterer: TopicClusterer | None = None,
    embedder: TextEmbedder | None = None,
    synthesis_provider: SynthesisProvider | None = None,
) -> InsightReport:
    """Analyze a canonical conversation without mutation or implicit training.

    Invalid input/arguments raise. Optional component failures are isolated and
    reported; extraction and TF-IDF remain available. No source gold label is
    copied into a prediction field. Scores and assignments must cite real IDs.
    """
    EvidenceIndex._check_k(top_k)
    if query is not None and not isinstance(query, str):
        raise TypeError("query must be a string or None")
    conv = validate_conversation(conversation)
    by_id = {u.id: u for u in conv.utterances}
    texts = {u.id: usable_text(u.text, u.is_deleted) for u in conv.utterances}
    # Existing canonical traversal gives parent-before-child source order. Never
    # sort on the timestamp sentinel or infer chronology from a missing value.
    inputs = (
        tuple(
            InferenceUtterance(
                id=u.id,
                text=texts[u.id],
                parent_id=u.parent_id,
                parent_text=texts.get(u.parent_id, ""),
                speaker_id=u.speaker_id,
                created_at=None if u.metadata.get("timestamp_missing") else u.created_at,
            )
            for u in conv.traverse()
            if texts[u.id]
        )
        if conv.utterances
        else ()
    )
    ids = {u.id for u in inputs}
    components = {
        "discourse": _status("unavailable", "No fitted classifier supplied."),
        "quality": _status("unavailable", "No fitted conversation civility scorer supplied."),
        "clustering": _status(
            "unavailable", "No topic clusterer supplied; excerpts are ungrouped."
        ),
        "embeddings": _status("unavailable", "No embedder supplied; TF-IDF is used."),
        "synthesis": _status("available", "Deterministic extractive fallback; no LLM required."),
    }
    if not inputs:
        for name in ("discourse", "quality", "clustering", "embeddings"):
            components[name] = _status("unavailable", "No usable utterances; stage not run.")
    if any(by_id[uid].discourse_act for uid in ids):
        components["discourse"] = _status("available", "Caller-supplied discourse labels retained.")
    if classifier is not None and inputs:
        try:
            predictions = _check_assignments(
                classifier.predict(inputs), ids, labels=DISCOURSE_LABELS
            )
            for uid, label in predictions.items():
                by_id[uid].discourse_act = label
            components["discourse"] = _status(
                "available",
                f"Injected classifier predicted {len(predictions)}/{len(inputs)} labels.",
            )
        except Exception as exc:
            components["discourse"] = _status(
                "failed", f"Classifier failed ({type(exc).__name__})."
            )
    quality = None
    if quality_scorer is not None and inputs:
        try:
            quality = ConversationQuality.model_validate(quality_scorer.score(inputs))
            if set(quality.evidence_ids) - ids:
                raise ValueError("Unknown quality context ID")
            components["quality"] = _status("available", "Injected conversation-level P(civil).")
        except Exception as exc:
            quality = None
            components["quality"] = _status(
                "failed", f"Civility scorer failed ({type(exc).__name__})."
            )
    topics = {}
    if clusterer is not None and inputs:
        try:
            topics = _check_assignments(clusterer.assign(inputs), ids)
            components["clustering"] = _status(
                "available", f"Injected clusterer assigned {len(topics)}/{len(inputs)} utterances."
            )
        except Exception as exc:
            components["clustering"] = _status(
                "failed", f"Clusterer failed ({type(exc).__name__})."
            )
    index = None
    if embedder is not None and inputs:
        try:
            index = EvidenceIndex(conv, embedder=embedder)
            hits = (
                index.search(query, top_k=top_k)
                if query is not None
                else index.representative(top_k=top_k)
            )
            components["embeddings"] = _status(
                "available", "Injected TextEmbedder; cosine retrieval."
            )
        except Exception as exc:
            components["embeddings"] = _status(
                "failed", f"Embedding retrieval failed ({type(exc).__name__}); using TF-IDF."
            )
            index = None
    if index is None:
        index = EvidenceIndex(conv)
        hits = (
            index.search(query, top_k=top_k)
            if query is not None
            else index.representative(top_k=top_k)
        )
    components["retrieval"] = _status(
        "available", f"{index.backend} cosine; {len(hits)} selected excerpts."
    )
    evidence = index.evidence
    points = [
        Claim(text=hit.evidence.text_clean[:400].rstrip(), evidence_ids=[hit.evidence.id])
        for hit in hits
    ]
    limitations = [
        "Excerpts represent participants' statements, not verified facts or a complete digest.",
        "TF-IDF measures lexical relevance; representative excerpts use centroid similarity.",
        "Labels identify utterance roles; they do not establish consensus, a disagreement target, or whether a question remains open.",
        "Civility probability applies to the observed conversation context, not per-comment argument strength; domain shift and calibration require evaluation.",
        "Extractive points contain at most the first 400 characters of a selected utterance; full source text remains in evidence.",
    ]
    if any(u.metadata.get("timestamp_missing") for u in conv.utterances):
        limitations.append(
            "Source timestamps are missing: canonical epoch sentinels are not real dates; evidence dates are null."
        )
        limitations.append(
            "Model inputs retain parent-before-child source order when chronological order cannot be established."
        )
    if any(u.metadata.get("speaker_missing") for u in conv.utterances):
        limitations.append(
            "Source speaker identities are missing; placeholder speaker references do not identify distinct participants."
        )
    if synthesis_provider is not None and hits:
        try:
            selected = [hit.evidence.model_copy(deep=True) for hit in hits]
            draft = SynthesisDraft.model_validate(synthesis_provider.synthesize(selected, query))
            selected_ids = {hit.evidence.id for hit in hits}
            for point in draft.points:
                if (
                    set(point.evidence_ids) - selected_ids
                    or len(set(point.evidence_ids)) != len(point.evidence_ids)
                    or any(point.text not in evidence[uid].text_clean for uid in point.evidence_ids)
                    or not point.text.strip()
                ):
                    raise ValueError("Synthesis must quote its retrieved evidence exactly")
            points = draft.points
            components["synthesis"] = _status(
                "available",
                "Injected provider; exact quotations and retrieved evidence IDs validated.",
            )
        except Exception as exc:
            components["synthesis"] = _status(
                "failed", f"Synthesis rejected ({type(exc).__name__}); extractive fallback used."
            )
    grouped: dict[str, list[Claim]] = {}
    for point in points:
        assignments = {topics.get(uid) for uid in point.evidence_ids}
        cluster = next(iter(assignments)) if len(assignments) == 1 else None
        heading = f"Topic cluster {cluster}" if cluster is not None else "Discussion excerpts"
        grouped.setdefault(heading, []).append(point)

    def labeled(label):
        return [
            point
            for point in points
            if all(evidence[uid].discourse_act == label for uid in point.evidence_ids)
        ]

    return InsightReport(
        conversation_id=conv.id,
        conversation_source_id=conv.source_id,
        title=conv.title,
        source_url=conv.url,
        query=query,
        summary=points[:3],
        sections=[Section(heading=h, points=p) for h, p in grouped.items()],
        agreements=labeled("agreement"),
        disagreements=labeled("disagreement"),
        questions=labeled("question"),
        evidence=evidence,
        conversation_quality=quality,
        topic_assignments=topics,
        components=components,
        limitations=limitations,
    )
