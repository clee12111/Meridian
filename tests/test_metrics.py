"""Known-answer regression tests for metrics, sanity checks, and taxonomy."""

from __future__ import annotations

import pytest

from core.measurement.metrics import (
    precision_at_k,
    recall_at_k,
    compute_all_k,
    MetricResult,
)
from core.measurement.sanity import (
    check_monotonicity,
    check_improvement,
    run_sanity_checks,
)
from core.measurement.taxonomy import (
    classify,
    classify_dtm,
    classify_xrf,
    FailureType,
)
from core.measurement.span_overlap import MissingGroundTruthError


# ===================================================================
# Fixtures — hardcoded span tuples
# ===================================================================

# gt: chars 100..200 (100 chars)
GT = [(100, 200)]

# Perfect retrieval: exact match
PERFECT = [(100, 200)]

# Zero overlap: completely wrong region
ZERO = [(500, 600)]

# Partial overlap: chars 150..200 = 50 chars overlap out of 100 gt
PARTIAL = [(150, 250)]

# Too short (ICR): chars 190..200 = 10 chars overlap out of 100 gt
TOO_SHORT = [(190, 200)]

# Too long (OVR): chars 0..400 = 400 chars retrieved, 100 overlap
TOO_LONG = [(0, 400)]


# ===================================================================
# test_metrics: precision_at_k / recall_at_k
# ===================================================================


class TestPerfectRecall:
    def test_recall(self):
        assert recall_at_k(PERFECT, GT, k=1) == 1.0

    def test_precision(self):
        assert precision_at_k(PERFECT, GT, k=1) == 1.0

    def test_compute_all(self):
        result = compute_all_k(PERFECT, GT, k_values=[1])
        assert result.r_at_k[1] == 1.0
        assert result.p_at_k[1] == 1.0
        assert result.eval_mode == "SPAN_OVERLAP"


class TestZeroRecall:
    def test_recall(self):
        assert recall_at_k(ZERO, GT, k=1) == 0.0

    def test_precision(self):
        assert precision_at_k(ZERO, GT, k=1) == 0.0


class TestPartialOverlap:
    def test_recall(self):
        # 50 chars overlap / 100 gt chars = 0.5
        assert recall_at_k(PARTIAL, GT, k=1) == pytest.approx(0.5)

    def test_precision(self):
        # 50 chars overlap / 100 retrieved chars = 0.5
        assert precision_at_k(PARTIAL, GT, k=1) == pytest.approx(0.5)


class TestTooShortICR:
    def test_recall(self):
        # 10 chars overlap / 100 gt chars = 0.1
        assert recall_at_k(TOO_SHORT, GT, k=1) == pytest.approx(0.1)

    def test_precision(self):
        # 10 chars overlap / 10 retrieved chars = 1.0
        assert precision_at_k(TOO_SHORT, GT, k=1) == 1.0


class TestTooLongOVR:
    def test_recall(self):
        # 100 chars overlap / 100 gt chars = 1.0
        assert recall_at_k(TOO_LONG, GT, k=1) == 1.0

    def test_precision(self):
        # 100 chars overlap / 400 retrieved chars = 0.25
        assert precision_at_k(TOO_LONG, GT, k=1) == pytest.approx(0.25)


class TestKTruncation:
    """Verify that k actually limits which spans are considered."""

    def test_k1_ignores_second_span(self):
        retrieved = [ZERO[0], PERFECT[0]]  # first span is wrong
        assert recall_at_k(retrieved, GT, k=1) == 0.0

    def test_k2_includes_second_span(self):
        retrieved = [ZERO[0], PERFECT[0]]  # second span is correct
        assert recall_at_k(retrieved, GT, k=2) == 1.0


class TestEmptyRetrieved:
    def test_recall_zero(self):
        assert recall_at_k([], GT, k=1) == 0.0

    def test_precision_zero(self):
        assert precision_at_k([], GT, k=1) == 0.0


class TestMissingGT:
    def test_recall_raises(self):
        with pytest.raises(MissingGroundTruthError):
            recall_at_k(PERFECT, [], k=1)

    def test_precision_raises(self):
        with pytest.raises(MissingGroundTruthError):
            precision_at_k(PERFECT, [], k=1)


# ===================================================================
# test_metrics: MetricResult
# ===================================================================


def test_metric_result_eval_mode_required():
    """eval_mode must be set explicitly."""
    result = MetricResult(p_at_k={1: 0.5}, r_at_k={1: 0.5}, eval_mode="SPAN_OVERLAP")
    assert result.eval_mode == "SPAN_OVERLAP"

    result2 = MetricResult(p_at_k={1: 0.5}, r_at_k={1: 0.5}, eval_mode="LLM_JUDGE")
    assert result2.eval_mode == "LLM_JUDGE"


def test_metric_result_frozen():
    result = MetricResult(p_at_k={1: 0.5}, r_at_k={1: 0.5}, eval_mode="SPAN_OVERLAP")
    with pytest.raises(AttributeError):
        result.eval_mode = "LLM_JUDGE"  # type: ignore[misc]


# ===================================================================
# test_sanity: check_monotonicity
# ===================================================================


def test_monotonic_result():
    result = compute_all_k(TOO_LONG, GT, k_values=[1, 2, 4])
    assert check_monotonicity(result) is True


def test_non_monotonic_recall():
    """Manually constructed: recall drops at higher k (shouldn't happen in practice)."""
    bad = MetricResult(
        p_at_k={1: 1.0, 2: 0.5},
        r_at_k={1: 0.8, 2: 0.5},  # recall decreased
        eval_mode="SPAN_OVERLAP",
    )
    assert check_monotonicity(bad) is False


def test_non_monotonic_precision():
    """Precision increases at higher k (shouldn't happen in practice)."""
    bad = MetricResult(
        p_at_k={1: 0.5, 2: 0.8},  # precision increased
        r_at_k={1: 0.5, 2: 0.8},
        eval_mode="SPAN_OVERLAP",
    )
    assert check_monotonicity(bad) is False


# ===================================================================
# test_sanity: check_improvement
# ===================================================================


def test_improvement_within_threshold():
    baseline = MetricResult(p_at_k={1: 0.5}, r_at_k={1: 0.5}, eval_mode="SPAN_OVERLAP")
    new = MetricResult(p_at_k={1: 0.6}, r_at_k={1: 0.6}, eval_mode="SPAN_OVERLAP")
    assert check_improvement(new, baseline, threshold=0.3) is True


def test_improvement_exceeds_threshold():
    baseline = MetricResult(p_at_k={1: 0.3}, r_at_k={1: 0.3}, eval_mode="SPAN_OVERLAP")
    new = MetricResult(p_at_k={1: 0.9}, r_at_k={1: 0.9}, eval_mode="SPAN_OVERLAP")
    assert check_improvement(new, baseline, threshold=0.3) is False


# ===================================================================
# test_sanity: run_sanity_checks — violations are explicit
# ===================================================================


def test_sanity_result_violations_named():
    bad = MetricResult(
        p_at_k={1: 0.5, 2: 0.8},
        r_at_k={1: 0.8, 2: 0.5},
        eval_mode="SPAN_OVERLAP",
    )
    result = run_sanity_checks(bad)
    assert result.passed is False
    assert len(result.violations) >= 2
    assert any("recall" in v for v in result.violations)
    assert any("precision" in v for v in result.violations)


# ===================================================================
# test_taxonomy
# ===================================================================


DOC_A = "doc_a"
DOC_B = "doc_b"


def test_drm_no_matching_doc():
    retrieved = [(DOC_B, 100, 200)]
    gt = [(DOC_A, 100, 200)]
    assert classify(retrieved, gt) == FailureType.DRM


def test_drm_empty_retrieved():
    gt = [(DOC_A, 100, 200)]
    assert classify([], gt) == FailureType.DRM


def test_cbf_correct_doc_zero_overlap():
    retrieved = [(DOC_A, 500, 600)]
    gt = [(DOC_A, 100, 200)]
    assert classify(retrieved, gt) == FailureType.CBF


def test_icr_partial_overlap():
    # gt = 100 chars, overlap = 10 chars (< 50%)
    retrieved = [(DOC_A, 190, 210)]
    gt = [(DOC_A, 100, 200)]
    assert classify(retrieved, gt) == FailureType.ICR


def test_ovr_retrieved_too_long():
    # gt = 100 chars, retrieved = 400 chars (>= 3x), full overlap
    retrieved = [(DOC_A, 0, 400)]
    gt = [(DOC_A, 100, 200)]
    assert classify(retrieved, gt) == FailureType.OVR


def test_ok_good_retrieval():
    # gt = 100 chars, retrieved = 120 chars, 100 overlap (100% recall, < 3x)
    retrieved = [(DOC_A, 90, 210)]
    gt = [(DOC_A, 100, 200)]
    assert classify(retrieved, gt) == FailureType.OK


# ===================================================================
# test_taxonomy: Phase 2 stubs
# ===================================================================


def test_dtm_not_implemented():
    with pytest.raises(NotImplementedError):
        classify_dtm([], [])


def test_xrf_not_implemented():
    with pytest.raises(NotImplementedError):
        classify_xrf([], [])
