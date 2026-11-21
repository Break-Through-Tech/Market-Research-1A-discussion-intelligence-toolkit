import numpy as np
import pytest

from discussion_intelligence import EvidenceIndex, analyze


def test_ranks_relevant_source_and_preserves_identity(conversation):
    hit = EvidenceIndex(conversation).search("RAID disk failure", top_k=1)[0]
    assert hit.evidence.id == "reddit:t1_storage"
    original = conversation.utterances[1]
    assert hit.evidence.source_id == original.source_id
    assert hit.evidence.parent_id == original.parent_id
    assert hit.evidence.speaker_id == original.speaker_id
    assert hit.evidence.source_url == original.metadata["source_url"]
    assert hit.evidence.conversation_url == conversation.url
    assert 0 < hit.score <= 1


@pytest.mark.parametrize("query", ["", "  ", "astronauts telescope", "!!!"])
def test_no_lexical_results(conversation, query):
    assert EvidenceIndex(conversation).search(query) == []


def test_empty_conversation(conversation):
    conversation.utterances = []
    assert EvidenceIndex(conversation).search("data") == []
    assert EvidenceIndex(conversation).representative() == []
    assert analyze(conversation).summary == []


def test_deleted_removed_and_empty_are_not_indexed(conversation):
    conversation.utterances[0].text = " [REMOVED] "
    conversation.utterances[1].is_deleted = True
    conversation.utterances[2].text = "\n  "
    assert EvidenceIndex(conversation).evidence == {}


def test_punctuation_only_has_no_vocabulary(conversation):
    for utterance in conversation.utterances:
        utterance.text = "... !!!"
    index = EvidenceIndex(conversation)
    assert index.search("!") == []
    assert index.representative() == []


def test_stable_ties_ignore_input_order(conversation):
    for utterance in conversation.utterances:
        utterance.text = "same text"
    expected = [u.id for u in sorted(conversation.utterances, key=lambda u: u.id)]
    first = EvidenceIndex(conversation).search("same")
    conversation.utterances.reverse()
    second = EvidenceIndex(conversation).search("same")
    assert [hit.evidence.id for hit in first] == expected
    assert first == second


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "blank",
        "unprefixed",
        "parent",
        "speaker",
        "conversation",
        "cycle",
        "roots",
        "empty_suffix",
    ],
)
def test_rejects_invalid_structure(conversation, kind):
    a, b, c = conversation.utterances
    if kind == "duplicate":
        c.id = b.id
    elif kind == "blank":
        b.id = " "
    elif kind == "unprefixed":
        b.id = "t1_storage"
    elif kind == "empty_suffix":
        b.id = "reddit:"
    elif kind == "parent":
        b.parent_id = "reddit:missing"
    elif kind == "speaker":
        b.speaker_id = "reddit:u/missing"
    elif kind == "conversation":
        b.conversation_id = "reddit:other"
    elif kind == "cycle":
        b.parent_id, c.parent_id = c.id, b.id
    else:
        c.parent_id = None
    with pytest.raises(ValueError):
        EvidenceIndex(conversation)


@pytest.mark.parametrize("top_k", [-1, 2.5, True])
def test_invalid_k(conversation, top_k):
    with pytest.raises(ValueError):
        EvidenceIndex(conversation).search("data", top_k=top_k)


def test_zero_k(conversation):
    assert EvidenceIndex(conversation).search("data", top_k=0) == []


class Embedder:
    def embed(self, texts):
        return np.array([[float("RAID" in text), float("gaming" in text)] for text in texts])


def test_injected_embedder_cosine_without_download(conversation):
    hit = EvidenceIndex(conversation, embedder=Embedder()).search("RAID", top_k=1)[0]
    assert hit.evidence.id == "reddit:t1_storage"
    assert hit.score == pytest.approx(1)
    assert (
        analyze(conversation, query="RAID", embedder=Embedder()).components["embeddings"].state
        == "available"
    )


@pytest.mark.parametrize("bad", [[], [[float("nan")]], [[0], [1]], [[0, 1, float("inf")]]])
def test_malformed_embeddings_fail_and_pipeline_falls_back(conversation, bad):
    class BadEmbedder:
        def embed(self, texts):
            return bad

    with pytest.raises(ValueError):
        EvidenceIndex(conversation, embedder=BadEmbedder())
    report = analyze(conversation, query="RAID", embedder=BadEmbedder())
    assert report.components["embeddings"].state == "failed"
    assert report.components["retrieval"].detail.startswith("tfidf")
    assert report.summary[0].evidence_ids == ["reddit:t1_storage"]


def test_query_embedding_dimension_mismatch_falls_back(conversation):
    class ChangingEmbedder:
        def embed(self, texts):
            return np.ones((len(texts), 2 if len(texts) > 1 else 3))

    report = analyze(conversation, query="RAID", embedder=ChangingEmbedder())
    assert report.components["embeddings"].state == "failed"
    assert report.summary[0].evidence_ids == ["reddit:t1_storage"]
