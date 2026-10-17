"""
Modeling-dataset preparation for labeled discussion corpora (issue #14).

Pipeline, in order:

    raw ConvoKit files ──> canonical Conversations ──> labeled examples
        (convokit_reader)      (normalize)              (examples + labels)
                                                              │
                         split manifest  <── conversation-level hash split (splits)
                                                              │
                                     leakage checks + dataset card (leakage, build)

Run the whole thing with:

    uv run --no-project --with pydantic python -m utils.modeling_data.build --help

Everything here is deterministic: the same corpus files, seed, and fractions
always produce byte-identical outputs and the same split manifest.
"""

from utils.modeling_data.labels import (
    CANONICAL_LABELS,
    LABEL_MAP,
    LABEL_MAP_VERSION,
    MODELING_DROP_LABELS,
)

__all__ = [
    "CANONICAL_LABELS",
    "LABEL_MAP",
    "LABEL_MAP_VERSION",
    "MODELING_DROP_LABELS",
]
