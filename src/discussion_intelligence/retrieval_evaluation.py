"""Curated retrieval evaluation; no retrieval algorithm or model training."""

from collections.abc import Callable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from artifacts.evaluation import retrieval_recall_at_k
from artifacts.schema import Conversation

from .retrieval import EvidenceIndex, SearchHit
from .validation import usable_text, validate_conversation


class JudgmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RelevantEvidence(JudgmentModel):
    id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class RetrievalCase(JudgmentModel):
    id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    relevant: list[RelevantEvidence]


class RetrievalJudgments(JudgmentModel):
    name: str
    scope: str
    curation_note: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    cases: list[RetrievalCase] = Field(min_length=1)


class RetrievalIndex(Protocol):
    def search(self, query: str, *, top_k: int) -> Sequence[SearchHit]: ...


def evaluate_retrieval(
    conversations: Sequence[Conversation],
    judgments: RetrievalJudgments,
    *,
    ks: Sequence[int] = (1, 2, 3),
    index_factory: Callable[[Conversation], RetrievalIndex] = EvidenceIndex,
    retriever_name: str | None = None,
) -> dict:
    """Score source-checked judgments with the existing harness's mean recall.

    Empty relevant sets have undefined recall and are evaluated separately for
    abstention. Invalid judgments, returned identities or provider exceptions
    raise rather than silently scoring a different retriever or source corpus.
    A future retriever can supply the existing search()/SearchHit interface.
    """
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("Evaluation cutoffs must be positive integers")
    cutoffs = sorted(set(ks))
    sources = {conv.id: validate_conversation(conv) for conv in conversations}
    if len(sources) != len(conversations):
        raise ValueError("Duplicate evaluation conversation ID")
    case_ids = [case.id for case in judgments.cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate evaluation case ID")
    indexes = {}
    rows = []
    for case in judgments.cases:
        if case.conversation_id not in sources:
            raise ValueError(f"Unknown evaluation conversation: {case.conversation_id}")
        conv = sources[case.conversation_id]
        utterances = {u.id: u for u in conv.utterances if usable_text(u.text, u.is_deleted)}
        relevant = {item.id for item in case.relevant}
        if len(relevant) != len(case.relevant):
            raise ValueError(f"Duplicate relevance ID in {case.id}")
        for item in case.relevant:
            if (
                item.id not in utterances
                or not item.quote.strip()
                or item.quote not in usable_text(utterances[item.id].text)
            ):
                raise ValueError(f"Relevance judgment lacks usable source support: {item.id}")
        if conv.id not in indexes:
            indexes[conv.id] = index_factory(conv.model_copy(deep=True))
        hits = list(indexes[conv.id].search(case.query, top_k=max(cutoffs)))
        ids = [hit.evidence.id for hit in hits]
        if len(ids) != len(set(ids)) or len(ids) > max(cutoffs):
            raise ValueError("Retriever returned repeated IDs or exceeded top_k")
        for hit in hits:
            evidence = hit.evidence
            if evidence.id not in utterances:
                raise ValueError("Retriever returned an unknown or unusable source ID")
            original = utterances[evidence.id]
            expected = (
                original.source_id,
                original.conversation_id,
                original.parent_id,
                original.speaker_id,
                original.text,
                usable_text(original.text),
                original.metadata.get("url") or original.metadata.get("source_url"),
                conv.url,
            )
            actual = (
                evidence.source_id,
                evidence.conversation_id,
                evidence.parent_id,
                evidence.speaker_id,
                evidence.text,
                evidence.text_clean,
                evidence.source_url,
                evidence.conversation_url,
            )
            if actual != expected:
                raise ValueError("Retriever changed source identity, text or links")
        rows.append(
            {
                "id": case.id,
                "conversation_id": conv.id,
                "query": case.query,
                "relevant_ids": sorted(relevant),
                "returned": [
                    {
                        "id": hit.evidence.id,
                        "source_id": hit.evidence.source_id,
                        "source_url": hit.evidence.source_url,
                        "conversation_url": hit.evidence.conversation_url,
                        "score": hit.score,
                    }
                    for hit in hits
                ],
                "recall_at_k": {
                    str(k): retrieval_recall_at_k([relevant], [ids], k) if relevant else None
                    for k in cutoffs
                },
                "abstained_at_k": {str(k): not ids[:k] for k in cutoffs},
            }
        )
    positives = [row for row in rows if row["relevant_ids"]]
    negatives = [row for row in rows if not row["relevant_ids"]]
    return {
        "evaluation": judgments.name,
        "scope": judgments.scope,
        "curation_note": judgments.curation_note,
        "judgments_source_sha256": judgments.source_sha256,
        "retriever": retriever_name
        or ("EvidenceIndex/tfidf" if index_factory is EvidenceIndex else "custom index (unnamed)"),
        "positive_queries": len(positives),
        "no_answer_queries": len(negatives),
        "metrics": {
            str(k): {
                "macro_recall": sum(row["recall_at_k"][str(k)] for row in positives)
                / len(positives)
                if positives
                else None,
                "no_answer_abstention_rate": sum(row["abstained_at_k"][str(k)] for row in negatives)
                / len(negatives)
                if negatives
                else None,
            }
            for k in cutoffs
        },
        "cases": rows,
    }
