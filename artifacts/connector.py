"""
Abstract Connector interface for ingesting threaded discussions.

Every data source — Reddit dumps, Hacker News dumps, GitHub Discussions exports,
custom forum scrapes — implements this interface. The pipeline downstream sees
only Conversation objects; it never knows which Connector produced them.

A Connector has three responsibilities:

  1. Read raw data from its source (parquet, JSON, archive, etc.)
  2. Normalize that data into the canonical schema
  3. Yield Conversations either by ID (`fetch_conversation`) or in bulk (`list_conversations`)

The Connector is the ONLY place in the toolkit that knows about source-specific
formats. Anything source-specific that doesn't fit the canonical schema goes
into `Utterance.metadata` or `Conversation.metadata`.

Design notes:

  * Sync, not async. The reference implementation reads local parquet via
    pandas; wrapping that in async buys nothing. Future HTTP-based connectors
    can add an `AsyncConnector` Protocol alongside without touching this one.

  * Per-connector `Config` subclasses (Pydantic). Configuration is type-validated
    at construction time. No kwargs soup, no global config.

  * Malformed conversations: skipped with a warning by default. Set
    `config.strict=True` to raise instead. Orphan comments, deleted roots, and
    broken parent references are common in real dumps; failing on the first
    bad row kills a bulk run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from pydantic import BaseModel, ConfigDict

if __package__:
    from .schema import Conversation
else:  # Preserve the original starter-notebook import path.
    from schema import Conversation


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConnectorError(Exception):
    """Base exception for connector failures."""


class ConversationNotFoundError(ConnectorError):
    """Raised when `fetch_conversation` is given an ID that does not exist."""


class MalformedConversationError(ConnectorError):
    """Raised when a source record cannot be normalized into a valid Conversation.

    Common causes: missing root submission, all comments deleted, broken parent
    references. Bulk iteration skips these with a warning by default.
    """


# ---------------------------------------------------------------------------
# Config base
# ---------------------------------------------------------------------------


class ConnectorConfig(BaseModel):
    """Base configuration for connectors.

    Subclasses extend this with source-specific fields (paths, filters, etc.).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strict: bool = False
    """If True, malformed conversations raise instead of being skipped."""


# ---------------------------------------------------------------------------
# Connector interface
# ---------------------------------------------------------------------------


class Connector(ABC):
    """Abstract base for all data source connectors.

    Subclasses must:
      * declare a class-level `source` attribute (e.g. 'reddit', 'hackernews')
      * implement `fetch_conversation` and `list_conversations`
      * accept a `ConnectorConfig` (typically a subclass-specific one) in __init__
    """

    source: str  # class-level: 'reddit', 'hackernews', etc.

    def __init__(self, config: ConnectorConfig) -> None:
        self.config = config

    @abstractmethod
    def fetch_conversation(self, conversation_id: str) -> Conversation:
        """Return the Conversation with the given canonical ID.

        Raises:
            ConversationNotFoundError: if no conversation with that ID exists.
            MalformedConversationError: if the source data exists but cannot
                be normalized into a valid Conversation.
        """
        ...

    @abstractmethod
    def list_conversations(self, limit: int | None = None) -> Iterator[Conversation]:
        """Yield Conversations lazily from the source.

        Iteration order is connector-specific (typically chronological by
        thread creation). Malformed conversations are skipped with a warning
        unless `config.strict=True`.

        Args:
            limit: Stop after yielding this many. None = no limit.
        """
        ...
