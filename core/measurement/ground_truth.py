"""Ground truth protocol and adapter stubs for the measurement layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GroundTruth(Protocol):
    """Protocol for ground-truth providers.

    Each provider maps query_ids to ground-truth spans and document text.
    Spans are half-open [start, end) character offsets matching
    LegalBench-RAG conventions.
    """

    def get_spans(self, query_id: str) -> list[tuple[int, int]]:
        """Return ground-truth character spans for *query_id*.

        Each span is a half-open ``[start, end)`` interval into the
        source document text.
        """
        ...

    def get_doc_id(self, query_id: str) -> str:
        """Return the document ID associated with *query_id*."""
        ...

    def all_query_ids(self) -> list[str]:
        """Return all evaluation query IDs."""
        ...

    def doc_text(self, doc_id: str) -> str:
        """Return the full document text for *doc_id*."""
        ...
