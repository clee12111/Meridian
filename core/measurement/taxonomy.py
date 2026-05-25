"""Failure taxonomy for retrieval span evaluation.

Phase 1 covers span-computable failure types only.
Phase 2 types (DTM, XRF) are stubs that raise NotImplementedError.
"""

from __future__ import annotations

from enum import Enum, auto


class FailureType(Enum):
    """Retrieval failure categories."""

    # Phase 1 — span-computable
    DRM = auto()  # Document Retrieval Miss: no retrieved doc_id matches gt doc_id
    CBF = auto()  # Correct But Failed: gt doc retrieved but zero char overlap
    ICR = auto()  # Incomplete Retrieval: overlap > 0 but < 50% of gt span length
    OVR = auto()  # Over-Retrieval: retrieved span >= 3x gt span length
    OK = auto()   # No failure detected

    # Phase 2 — require LLM judge
    DTM = auto()  # Distractor Match
    XRF = auto()  # Cross-Reference Failure


def classify(
    retrieved_spans_with_doc_ids: list[tuple[str, int, int]],
    gt_spans_with_doc_ids: list[tuple[str, int, int]],
) -> FailureType:
    """Classify the failure type for a single query.

    Parameters
    ----------
    retrieved_spans_with_doc_ids:
        ``[(doc_id, start, end), ...]`` from the retriever.
    gt_spans_with_doc_ids:
        ``[(doc_id, start, end), ...]`` ground-truth spans.

    Returns ``FailureType.OK`` if none of the failure conditions apply.
    """
    if not retrieved_spans_with_doc_ids:
        return FailureType.DRM

    gt_doc_ids = {doc_id for doc_id, _, _ in gt_spans_with_doc_ids}
    retrieved_doc_ids = {doc_id for doc_id, _, _ in retrieved_spans_with_doc_ids}

    # DRM: no retrieved doc matches any gt doc
    if not (gt_doc_ids & retrieved_doc_ids):
        return FailureType.DRM

    # Build char sets scoped to matching docs
    gt_chars: set[int] = set()
    for doc_id, start, end in gt_spans_with_doc_ids:
        gt_chars.update(range(start, end))

    retrieved_chars: set[int] = set()
    for doc_id, start, end in retrieved_spans_with_doc_ids:
        if doc_id in gt_doc_ids:
            retrieved_chars.update(range(start, end))

    gt_total = len(gt_chars)
    retrieved_total = len(retrieved_chars)
    overlap = len(gt_chars & retrieved_chars)

    # CBF: correct doc but zero character overlap
    if overlap == 0:
        return FailureType.CBF

    # ICR: overlap > 0 but less than 50% of gt span length
    if gt_total > 0 and overlap < 0.5 * gt_total:
        return FailureType.ICR

    # OVR: retrieved span is >= 3x gt span length
    if gt_total > 0 and retrieved_total >= 3 * gt_total:
        return FailureType.OVR

    return FailureType.OK


def classify_dtm(
    retrieved_spans_with_doc_ids: list[tuple[str, int, int]],
    gt_spans_with_doc_ids: list[tuple[str, int, int]],
) -> FailureType:
    """Phase 2 stub: Distractor Match classification."""
    raise NotImplementedError("DTM classification requires LLM judge (Phase 2)")


def classify_xrf(
    retrieved_spans_with_doc_ids: list[tuple[str, int, int]],
    gt_spans_with_doc_ids: list[tuple[str, int, int]],
) -> FailureType:
    """Phase 2 stub: Cross-Reference Failure classification."""
    raise NotImplementedError("XRF classification requires LLM judge (Phase 2)")
