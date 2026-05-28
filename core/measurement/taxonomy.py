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

    Returns a list of dicts, one per gt span (zero-width spans filtered):
      - length: span length in chars
      - overlap: chars overlapping with retrieved
      - coverage: overlap / length (0.0 to 1.0)
      - status: "covered" (>=50%), "partial" (<50% >0), "missed" (0%)

    Zero-width spans (start == end) are silently dropped: they cover zero
    characters and cannot be meaningfully missed or covered.
    """
    results = []
    for start, end in gt_spans:
        if start >= end:
            continue
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


# ── Confidence scoring (additive — does not change classify()) ────────────

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationResult:
    """classify() output plus confidence signals.

    failure_type is IDENTICAL to what classify() returns standalone.
    Confidence measures distance-to-boundary for the classification.
    """

    failure_type: FailureType
    confidence: str            # "HIGH" | "MEDIUM" | "LOW"
    ambiguous: bool            # True if confidence == "LOW"
    boundary_distance: float   # 0.0 = on boundary, 1.0 = far from it
    near_category: str | None  # the category it's closest to flipping into
    evidence: dict             # raw signals for downstream analysis


def classify_with_confidence(
    retrieved_spans_with_doc_ids: list[tuple[str, int, int]],
    gt_spans_with_doc_ids: list[tuple[str, int, int]],
) -> ClassificationResult:
    """Classify with confidence scoring.

    Calls classify() unchanged for the authoritative label, then
    recomputes the same internal signals to derive distance-to-boundary.
    """
    # STEP 1 — authoritative label (unchanged)
    failure_type = classify(
        retrieved_spans_with_doc_ids, gt_spans_with_doc_ids
    )

    # STEP 2 — recompute signals
    if not retrieved_spans_with_doc_ids or not gt_spans_with_doc_ids:
        return ClassificationResult(
            failure_type=failure_type,
            confidence="HIGH",
            ambiguous=False,
            boundary_distance=1.0,
            near_category=None,
            evidence={
                "overlap_ratio": 0.0,
                "retrieved_gt_ratio": 0.0,
                "doc_intersection_size": 0,
                "n_covered": 0, "n_missed": 0, "n_partial": 0,
                "per_span_coverage": [],
            },
        )

    gt_doc_ids = {doc_id for doc_id, _, _ in gt_spans_with_doc_ids}
    retrieved_doc_ids = {doc_id for doc_id, _, _ in retrieved_spans_with_doc_ids}
    doc_intersection_size = len(gt_doc_ids & retrieved_doc_ids)

    # Build retrieved char set scoped to gt docs
    retrieved_chars: set[int] = set()
    for doc_id, start, end in retrieved_spans_with_doc_ids:
        if doc_id in gt_doc_ids:
            retrieved_chars.update(range(start, end))

    gt_spans = [(start, end) for _, start, end in gt_spans_with_doc_ids]
    span_results = _per_span_coverage(retrieved_chars, gt_spans)

    n_covered = sum(1 for s in span_results if s["status"] == "covered")
    n_missed = sum(1 for s in span_results if s["status"] == "missed")
    n_partial = sum(1 for s in span_results if s["status"] == "partial")
    total_overlap = sum(s["overlap"] for s in span_results)
    gt_total = sum(s["length"] for s in span_results)
    retrieved_total = len(retrieved_chars)

    overlap_ratio = total_overlap / gt_total if gt_total > 0 else 0.0
    retrieved_gt_ratio = retrieved_total / gt_total if gt_total > 0 else 0.0
    per_span_cov = [s["coverage"] for s in span_results]

    evidence = {
        "overlap_ratio": round(overlap_ratio, 4),
        "retrieved_gt_ratio": round(retrieved_gt_ratio, 4),
        "doc_intersection_size": doc_intersection_size,
        "n_covered": n_covered,
        "n_missed": n_missed,
        "n_partial": n_partial,
        "per_span_coverage": [round(c, 4) for c in per_span_cov],
    }

    # STEP 3 — confidence by failure type
    confidence: str
    boundary_distance: float
    near_category: str | None

    if failure_type == FailureType.DRM:
        confidence = "HIGH"
        boundary_distance = 1.0
        near_category = None

    elif failure_type == FailureType.CBF:
        # Check if any retrieved span is close to touching a gt span
        min_gap = _min_span_gap(retrieved_spans_with_doc_ids, gt_spans_with_doc_ids)
        if min_gap is not None and min_gap <= 50:
            confidence = "LOW"
            boundary_distance = min_gap / 50.0
            near_category = "ICR"
        elif min_gap is not None and min_gap <= 200:
            confidence = "MEDIUM"
            boundary_distance = min(min_gap / 200.0, 1.0)
            near_category = "ICR"
        else:
            confidence = "HIGH"
            boundary_distance = 1.0
            near_category = None

    elif failure_type == FailureType.ICR:
        distance = abs(overlap_ratio - 0.5) / 0.5 if overlap_ratio < 0.5 else 0.0
        if overlap_ratio >= 0.40:
            confidence = "LOW"
            near_category = "OK"
        elif overlap_ratio >= 0.30:
            confidence = "MEDIUM"
            near_category = "OK"
        else:
            confidence = "HIGH"
            near_category = None
        boundary_distance = distance

    elif failure_type == FailureType.OK:
        distance = abs(overlap_ratio - 0.5) / 0.5
        near_category = None
        # Check ICR boundary
        if overlap_ratio <= 0.60:
            confidence = "LOW"
            near_category = "ICR"
        elif overlap_ratio <= 0.70:
            confidence = "MEDIUM"
            near_category = "ICR"
        else:
            confidence = "HIGH"
        # Also check OVR boundary
        if retrieved_gt_ratio >= 2.5 and retrieved_gt_ratio < 3.0:
            if confidence == "HIGH":
                confidence = "MEDIUM"
            near_category = "OVR"
        boundary_distance = distance

    elif failure_type == FailureType.OVR:
        distance = abs(retrieved_gt_ratio - 3.0) / 3.0
        if retrieved_gt_ratio < 3.5:
            confidence = "LOW"
            near_category = "OK"
        elif retrieved_gt_ratio < 4.5:
            confidence = "MEDIUM"
            near_category = "OK"
        else:
            confidence = "HIGH"
            near_category = None
        boundary_distance = min(distance, 1.0)

    elif failure_type == FailureType.SGP:
        total_counted = n_covered + n_missed
        balance = n_covered / total_counted if total_counted > 0 else 0.5
        if 0.4 <= balance <= 0.6:
            confidence = "LOW"
        elif 0.25 <= balance < 0.4 or 0.6 < balance <= 0.75:
            confidence = "MEDIUM"
        else:
            confidence = "HIGH"
        near_category = "OK" if balance > 0.5 else "CBF"
        boundary_distance = abs(balance - 0.5) / 0.5
        # Also flag LOW if any span has partial coverage near 0.5
        if any(0.45 <= c <= 0.55 for c in per_span_cov):
            if confidence == "HIGH":
                confidence = "MEDIUM"

    else:
        # Phase 2 types or unknown — default HIGH
        confidence = "HIGH"
        boundary_distance = 1.0
        near_category = None

    return ClassificationResult(
        failure_type=failure_type,
        confidence=confidence,
        ambiguous=(confidence == "LOW"),
        boundary_distance=round(boundary_distance, 4),
        near_category=near_category,
        evidence=evidence,
    )


def _min_span_gap(
    retrieved: list[tuple[str, int, int]],
    gt: list[tuple[str, int, int]],
) -> int | None:
    """Minimum character gap between any retrieved span and any gt span.

    Only considers spans from matching doc_ids. Returns None if no
    doc_ids match.
    """
    gt_doc_ids = {doc_id for doc_id, _, _ in gt}
    min_gap = None
    for r_doc, r_start, r_end in retrieved:
        if r_doc not in gt_doc_ids:
            continue
        for g_doc, g_start, g_end in gt:
            if r_doc != g_doc:
                continue
            # Gap = distance between closest edges
            if r_end <= g_start:
                gap = g_start - r_end
            elif g_end <= r_start:
                gap = r_start - g_end
            else:
                gap = 0  # overlapping
            if min_gap is None or gap < min_gap:
                min_gap = gap
    return min_gap


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
