"""Span overlap calculation for half-open [start, end) character intervals.

Matches LegalBench-RAG conventions (confirmed from ZeroEntropy source):
spans use Python-slice semantics, length = end - start.

No external dependencies — stdlib only.
"""

from __future__ import annotations


class MissingGroundTruthError(Exception):
    """Raised when ground-truth spans are empty or None."""


def _char_set(spans: list[tuple[int, int]]) -> set[int]:
    """Expand a list of half-open spans into a set of character indices."""
    chars: set[int] = set()
    for start, end in spans:
        chars.update(range(start, end))
    return chars


def span_overlap(
    retrieved: list[tuple[int, int]],
    gt: list[tuple[int, int]],
) -> float:
    """Fraction of ground-truth characters covered by retrieved spans.

    Parameters
    ----------
    retrieved:
        Half-open ``[start, end)`` spans from the retriever.
    gt:
        Half-open ``[start, end)`` ground-truth spans.

    Returns
    -------
    float
        ``intersection_chars / gt_chars``, in ``[0.0, 1.0]``.

    Raises
    ------
    MissingGroundTruthError
        If *gt* is ``None`` or empty.
    """
    if not gt:
        raise MissingGroundTruthError(
            "Cannot compute span overlap: ground-truth spans are empty or None"
        )

    gt_chars = _char_set(gt)
    if not gt_chars:
        raise MissingGroundTruthError(
            "Cannot compute span overlap: ground-truth spans cover zero characters"
        )

    if not retrieved:
        return 0.0

    retrieved_chars = _char_set(retrieved)
    intersection = gt_chars & retrieved_chars
    return len(intersection) / len(gt_chars)
