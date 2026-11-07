"""Validate thread identity and structure before traversing or indexing it."""

from artifacts.schema import Conversation


def validate_conversation(conversation: Conversation) -> Conversation:
    """Return a validated deep copy, preserving source order and source depth.

    Empty conversations are allowed (and produce empty reports). Nonempty
    conversations must form one connected tree. Depth is not recomputed: the
    existing normalizer deliberately preserves source depth for repaired orphans.
    """
    if not isinstance(conversation, Conversation):
        raise TypeError("Expected artifacts.schema.Conversation")
    conv = Conversation.model_validate(conversation.model_dump())
    if (
        not conv.source.strip()
        or not conv.id.startswith(f"{conv.source}:")
        or not conv.id.partition(":")[2]
        or any(c.isspace() for c in conv.id)
    ):
        raise ValueError("Conversation ID must be nonempty and source-prefixed")
    ids: set[str] = set()
    for u in conv.utterances:
        if (
            not u.id.startswith(f"{conv.source}:")
            or not u.id.partition(":")[2]
            or any(c.isspace() for c in u.id)
        ):
            raise ValueError(f"Invalid canonical utterance ID: {u.id!r}")
        if u.id in ids:
            raise ValueError(f"Duplicate utterance ID: {u.id}")
        ids.add(u.id)
        if not u.source_id.strip() or u.conversation_id != conv.id:
            raise ValueError(f"Invalid source/conversation reference for {u.id}")
        if u.speaker_id not in conv.speakers or conv.speakers[u.speaker_id].id != u.speaker_id:
            raise ValueError(f"Unknown or mismatched speaker for {u.id}")
    if not conv.utterances:
        return conv
    root = conv.root()
    children: dict[str, list[str]] = {}
    for u in conv.utterances:
        if u.parent_id is not None:
            if u.parent_id not in ids or u.parent_id == u.id:
                raise ValueError(f"Invalid parent reference for {u.id}")
            children.setdefault(u.parent_id, []).append(u.id)
    visited: set[str] = set()
    stack = [root.id]
    while stack:
        uid = stack.pop()
        if uid in visited:
            raise ValueError("Conversation contains a cycle")
        visited.add(uid)
        stack.extend(children.get(uid, []))
    if visited != ids:
        raise ValueError("Conversation contains a disconnected cycle")
    return conv


def usable_text(text: str, is_deleted: bool = False) -> str:
    """Conservative whitespace cleaning; never replace the original source text."""
    cleaned = " ".join(text.split())
    if is_deleted or cleaned.lower() in {"[deleted]", "[removed]"}:
        return ""
    return cleaned
