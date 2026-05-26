"""Failure taxonomy for retrieval span evaluation.

Phase 1 covers span-computable failure types only.
Phase 2 types (DTM, XRF) are stubs that raise NotImplementedError.

Classification uses PER-SPAN coverage analysis, not merged character
sets.  Each ground-truth span is individually scored against the
retrieved spans, then the per-span verdicts determine the failure type.
This correctly handles multi-span queries (43% of LegalBench-RAG)
where merged scoring masks coverage gaps.
"""

from __future__ import annotations

from enum import Enum, auto


class FailureType(Enum):
    """Retrieval failure categories."""

    # Phase 1 -- span-computable
    DRM = auto()  # Document Retrieval Miss: no retrieved doc_id matches gt doc_id
    CBF = auto()  # Correct But Failed: gt doc retrieved but zero char overlap on ALL spans
    SGP = auto()  # Span Gap: some gt spans covered (>=50%), others entirely missed (0%)
    ICR = auto()  # Incomplete Retrieval: overlap > 0 but < 50% of total gt span length
    OVR = auto()  # Over-Retrieval: retrieved span >= 3x gt span length
    OK = auto()   # No failure detected

    # Phase 2 -- require LLM judge
    DTM = auto()  # Distractor Match
    XRF = auto()  # Cross-Reference Failure


def _per_span_coverage(
    retrieved_chars: set[int],
    gt_spans: list[tuple[int, int]],
) -> list[dict]:
    """Compute per-span coverage against retrieved character set.

    Returns a list of dicts, one per gt span:
      - length: span length in chars
      - overlap: chars overlapping with retrieved
      - coverage: overlap / length (0.0 to 1.0)
      - status: "covered" (>=50%), "partial" (<50% >0), "missed" (0%)
    """
    results = []
    for start, end in gt_spans:
        span_chars = set(range(start, end))
        length = len(span_chars)
        overlap = len(span_chars & retrieved_chars) if length > 0 else 0
        coverage = overlap / length if length > 0 else 0.0

        if overlap == 0:
            status = "missed"
        elif coverage >= 0.5:
            status = "covered"
        else:
            status = "partial"

        results.append({
            "length": length,
            "overlap": overlap,
            "coverage": coverage,
            "status": status,
        })
    return results


def classify(
    retrieved_spans_with_doc_ids: list[tuple[str, int, int]],
    gt_spans_with_doc_ids: list[tuple[str, int, int]],
) -> FailureType:
    """Classify the failure type for a single query using per-span analysis.

    Parameters
    ----------
    retrieved_spans_with_doc_ids:
        ``[(doc_id, start, end), ...]`` from the retriever.
    gt_spans_with_doc_ids:
        ``[(doc_id, start, end), ...]`` ground-truth spans.

    Classification priority (first match wins):
      1. DRM  -- no retrieved doc matches any gt doc
      2. CBF  -- correct doc but zero overlap on every gt span
      3. SGP  -- at least one gt span covered (>=50%) AND at least one
                 gt span entirely missed (0%).  "Found some, missed others."
      4. ICR  -- total overlap > 0 but < 50% of total gt chars
      5. OVR  -- total retrieved chars (in gt docs) >= 3x total gt chars
      6. OK   -- none of the above

    For single-span queries, SGP can never trigger (there is only one
    span to cover or miss), so the behavior is identical to the prior
    merged-character-set classifier.
    """
    if not retrieved_spans_with_doc_ids:
        return FailureType.DRM

    gt_doc_ids = {doc_id for doc_id, _, _ in gt_spans_with_doc_ids}
    retrieved_doc_ids = {doc_id for doc_id, _, _ in retrieved_spans_with_doc_ids}

    # DRM: no retrieved doc matches any gt doc
    if not (gt_doc_ids & retrieved_doc_ids):
        return FailureType.DRM

    # Build retrieved char set scoped to gt docs
    retrieved_chars: set[int] = set()
    for doc_id, start, end in retrieved_spans_with_doc_ids:
        if doc_id in gt_doc_ids:
            retrieved_chars.update(range(start, end))

    # Per-span coverage analysis
    gt_spans = [(start, end) for _, start, end in gt_spans_with_doc_ids]
    span_results = _per_span_coverage(retrieved_chars, gt_spans)

    n_covered = sum(1 for s in span_results if s["status"] == "covered")
    n_missed = sum(1 for s in span_results if s["status"] == "missed")
    total_overlap = sum(s["overlap"] for s in span_results)

    # CBF: correct doc but zero overlap on ALL spans
    if total_overlap == 0:
        return FailureType.CBF

    # SGP: some spans covered, others entirely missed
    if n_covered > 0 and n_missed > 0:
        return FailureType.SGP

    # Aggregate metrics for ICR/OVR (same as before)
    gt_total = sum(s["length"] for s in span_results)
    retrieved_total = len(retrieved_chars)

    # ICR: overlap > 0 but less than 50% of total gt span length
    if gt_total > 0 and total_overlap < 0.5 * gt_total:
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
