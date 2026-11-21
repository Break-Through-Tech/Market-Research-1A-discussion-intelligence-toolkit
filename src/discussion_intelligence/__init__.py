"""Offline discussion analysis with explicit, evidence-linked results."""

from artifacts.schema import Conversation, Speaker, Utterance

from .pipeline import analyze
from .reports import InsightReport
from .retrieval import EvidenceIndex

__all__ = ["Conversation", "Speaker", "Utterance", "InsightReport", "EvidenceIndex", "analyze"]
