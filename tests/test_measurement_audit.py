"""Measurement library correctness audit — four independent fronts.

This is a READ-ONLY audit: it surfaces problems but fixes nothing.
Run with: pytest tests/test_measurement_audit.py -v --tb=short
"""

from __future__ import annotations

import math
from collections import Counter

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from core.measurement.span_overlap import span_overlap, MissingGroundTruthError
from core.measurement.metrics import (
    precision_at_k,
    recall_at_k,
    compute_all_k,
    MetricResult,
)
from core.measurement.sanity import check_monotonicity, run_sanity_checks
from core.measurement.taxonomy import (
    classify,
    _per_span_coverage,
    FailureType,
)


# =====================================================================
# Strategies for Hypothesis
# =====================================================================

# Small spans to avoid memory blowup
span_st = st.tuples(
    st.integers(min_value=0, max_value=500),
    st.integers(min_value=0, max_value=500),
)

# Non-empty, non-zero-width spans
valid_span_st = st.tuples(
    st.integers(min_value=0, max_value=500),
    st.integers(min_value=1, max_value=500),
).map(lambda t: (min(t[0], t[1] - 1), max(t[0] + 1, t[1])))

valid_span_list_st = st.lists(valid_span_st, min_size=1, max_size=10)

DOC_A = "doc_a"
DOC_B = "doc_b"
DOC_C = "doc_c"


# =====================================================================
# FRONT 1: ADVERSARIAL PROPERTY TESTING
# =====================================================================

class TestFront1_AdversarialSpanOverlap:
    """Attack span_overlap with pathological inputs."""

    def test_zero_width_gt_single(self):
        """Zero-width gt span (5,5) should raise, not return 0."""
        with pytest.raises(MissingGroundTruthError):
            span_overlap([(0, 10)], [(5, 5)])

    def test_zero_width_gt_mixed_with_valid(self):
        """Mix of zero-width and valid gt spans — what happens?
        gt = [(5,5), (10,20)]. Zero-width contributes 0 chars, valid has 10.
        gt_chars = {10..19}, should work normally."""
        result = span_overlap([(10, 20)], [(5, 5), (10, 20)])
        assert result == 1.0

    def test_zero_width_retrieved(self):
        """Zero-width retrieved span should contribute nothing."""
        result = span_overlap([(5, 5)], [(0, 10)])
        assert result == 0.0

    def test_all_zero_width_gt(self):
        """All gt spans zero-width → covers zero chars → should raise."""
        with pytest.raises(MissingGroundTruthError):
            span_overlap([(0, 10)], [(5, 5), (8, 8)])

    def test_inverted_span_retrieved(self):
        """Inverted span (10, 5) → range(10, 5) = empty → no crash, no chars."""
        result = span_overlap([(10, 5)], [(0, 10)])
        assert result == 0.0

    def test_inverted_span_gt(self):
        """Inverted gt span (10, 5) → range(10,5)=empty. If only gt span, should raise."""
        with pytest.raises(MissingGroundTruthError):
            span_overlap([(0, 10)], [(10, 5)])

    def test_adjacent_spans_no_overlap(self):
        """Spans [0,5) and [5,10) are adjacent but don't overlap (half-open).
        If gt=[0,5) and retrieved=[5,10), overlap should be 0."""
        assert span_overlap([(5, 10)], [(0, 5)]) == 0.0

    def test_off_by_one_boundary(self):
        """gt=[0,10), retrieved=[9,20). Only char 9 overlaps → 1/10 = 0.1."""
        result = span_overlap([(9, 20)], [(0, 10)])
        assert result == pytest.approx(0.1)

    def test_duplicate_retrieved_spans(self):
        """Identical duplicate retrieved spans should not double-count."""
        r1 = span_overlap([(0, 10)], [(0, 10)])
        r2 = span_overlap([(0, 10), (0, 10), (0, 10)], [(0, 10)])
        assert r1 == r2 == 1.0

    def test_duplicate_gt_spans(self):
        """Duplicate gt spans — chars are deduplicated, so coverage = normal."""
        result = span_overlap([(0, 10)], [(0, 10), (0, 10)])
        assert result == 1.0

    def test_very_large_span(self):
        """A single huge span should not crash."""
        result = span_overlap([(0, 10000)], [(0, 10000)])
        assert result == 1.0

    def test_touching_but_not_overlapping(self):
        """gt=[10,20), retrieved=[20,30). Char 20 is in retrieved but not gt."""
        assert span_overlap([(20, 30)], [(10, 20)]) == 0.0

    def test_single_char_overlap(self):
        """gt=[0,10), retrieved=[9,10). Only char 9 overlaps → 1/10."""
        result = span_overlap([(9, 10)], [(0, 10)])
        assert result == pytest.approx(0.1)

    @given(retrieved=valid_span_list_st, gt=valid_span_list_st)
    @settings(max_examples=10_000, suppress_health_check=[HealthCheck.too_slow])
    def test_never_nan(self, retrieved, gt):
        """span_overlap should never return NaN."""
        result = span_overlap(retrieved, gt)
        assert not math.isnan(result)

    @given(retrieved=valid_span_list_st, gt=valid_span_list_st)
    @settings(max_examples=10_000, suppress_health_check=[HealthCheck.too_slow])
    def test_always_bounded(self, retrieved, gt):
        """span_overlap must be in [0.0, 1.0]."""
        result = span_overlap(retrieved, gt)
        assert 0.0 <= result <= 1.0


class TestFront1_AdversarialPrecision:
    """Attack precision_at_k with pathological inputs."""

    def test_zero_width_gt_raises(self):
        """precision_at_k with zero-width gt [(5,5)] must raise,
        consistent with span_overlap's behavior."""
        with pytest.raises(MissingGroundTruthError):
            precision_at_k([(0, 10)], [(5, 5)], k=1)

    def test_zero_width_gt_mixed(self):
        """precision_at_k with [(5,5), (10,20)] — zero-width span has no gt chars."""
        # gt_chars = {10..19} (10 chars), retrieved_chars = {10..19}
        result = precision_at_k([(10, 20)], [(5, 5), (10, 20)], k=1)
        assert result == 1.0

    def test_k_zero(self):
        """k=0 → top_k is empty → should return 0.0."""
        result = precision_at_k([(0, 10)], [(0, 10)], k=0)
        assert result == 0.0

    def test_negative_k(self):
        """k=-1 → retrieved_spans[:-1] = all but last. Not documented."""
        # Python slice with negative k: [(0,10)][:-1] = []
        result = precision_at_k([(0, 10)], [(0, 10)], k=-1)
        assert result == 0.0  # or should it raise?

    @given(
        retrieved=valid_span_list_st,
        gt=valid_span_list_st,
        k=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=10_000, suppress_health_check=[HealthCheck.too_slow])
    def test_precision_bounded(self, retrieved, gt, k):
        """precision_at_k must be in [0.0, 1.0]."""
        result = precision_at_k(retrieved, gt, k)
        assert 0.0 <= result <= 1.0

    @given(
        retrieved=valid_span_list_st,
        gt=valid_span_list_st,
        k=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=10_000, suppress_health_check=[HealthCheck.too_slow])
    def test_precision_never_nan(self, retrieved, gt, k):
        """precision_at_k should never return NaN."""
        result = precision_at_k(retrieved, gt, k)
        assert not math.isnan(result)


class TestFront1_AdversarialRecall:
    """Attack recall_at_k."""

    @given(
        retrieved=valid_span_list_st,
        gt=valid_span_list_st,
        k=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=10_000, suppress_health_check=[HealthCheck.too_slow])
    def test_recall_bounded(self, retrieved, gt, k):
        result = recall_at_k(retrieved, gt, k)
        assert 0.0 <= result <= 1.0

    @given(
        retrieved=valid_span_list_st,
        gt=valid_span_list_st,
        k=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=10_000, suppress_health_check=[HealthCheck.too_slow])
    def test_recall_never_nan(self, retrieved, gt, k):
        result = recall_at_k(retrieved, gt, k)
        assert not math.isnan(result)


class TestFront1_AdversarialClassifier:
    """Attack the failure classifier — priority target."""

    def test_zero_width_gt_span_filtered(self):
        """Zero-width gt span must be filtered: (doc_a, 15, 15) is dropped,
        leaving only the valid span which is fully covered → OK, not SGP."""
        retrieved = [(DOC_A, 0, 20)]
        gt = [(DOC_A, 0, 10), (DOC_A, 15, 15)]  # second span is zero-width
        result = classify(retrieved, gt)
        assert result == FailureType.OK, (
            f"Expected OK after zero-width span filtered, got {result}"
        )

    def test_zero_width_gt_all_filtered(self):
        """All zero-width gt spans filtered → no spans to classify.
        With correct doc match but zero valid gt spans, total_overlap=0 → CBF.
        This is acceptable: degenerate gt with no valid spans."""
        retrieved = [(DOC_A, 0, 100)]
        gt = [(DOC_A, 50, 50), (DOC_A, 60, 60)]
        result = classify(retrieved, gt)
        # After filtering, span_results is empty → total_overlap=0 → CBF
        # This is the best available classification for degenerate input
        assert result == FailureType.CBF

    def test_inverted_gt_span_in_classifier(self):
        """Inverted gt span (100, 50): range(100,50) = empty → length=0."""
        retrieved = [(DOC_A, 0, 200)]
        gt = [(DOC_A, 100, 50)]  # inverted
        result = classify(retrieved, gt)
        # length=0, overlap=0, status="missed", total_overlap=0 → CBF
        # But this is garbage input. At minimum it shouldn't crash.
        assert isinstance(result, FailureType)

    def test_fully_nested_spans(self):
        """gt=[0,100), retrieved=[10,20). Nested inside gt."""
        retrieved = [(DOC_A, 10, 20)]
        gt = [(DOC_A, 0, 100)]
        result = classify(retrieved, gt)
        # overlap = 10/100 = 10% < 50% → ICR
        assert result == FailureType.ICR

    def test_spans_at_exact_boundary(self):
        """Retrieved exactly at chunk boundary: gt=[0,100), retrieved=[100,200).
        Adjacent but non-overlapping → overlap=0 → CBF."""
        retrieved = [(DOC_A, 100, 200)]
        gt = [(DOC_A, 0, 100)]
        assert classify(retrieved, gt) == FailureType.CBF

    def test_multi_doc_char_offset_collision_known_limitation(self):
        """KNOWN LIMITATION (documented in docs/architecture.md):
        Multi-doc gt with shared offsets produces incorrect classification.
        This test documents the current (wrong) behavior — NOT a regression
        test. When the fix is implemented, update this test."""
        retrieved = [(DOC_A, 0, 100)]
        gt = [(DOC_A, 0, 100), (DOC_B, 0, 100)]
        result = classify(retrieved, gt)
        # Current (wrong) behavior: OK. Correct answer: SGP.
        # This is the known multi-doc limitation, gated in architecture.md.
        assert result == FailureType.OK  # WRONG but documented

    def test_multi_doc_different_offsets_no_collision(self):
        """Multi-doc gt at different offsets — no collision expected."""
        retrieved = [(DOC_A, 0, 100)]
        gt = [(DOC_A, 0, 100), (DOC_B, 500, 600)]
        result = classify(retrieved, gt)
        # retrieved_chars = {0..99}
        # gt_spans: [(0,100), (500,600)]
        # First span: fully covered. Second span: missed.
        # n_covered=1, n_missed=1 → SGP
        # This is correct: doc_b's span is genuinely missed.
        assert result == FailureType.SGP

    def test_empty_gt_spans(self):
        """classify with empty gt list — should be robust."""
        result = classify([(DOC_A, 0, 100)], [])
        # gt_doc_ids = empty set, intersection = empty → DRM
        assert result == FailureType.DRM

    @given(
        n_spans=st.integers(min_value=1, max_value=8),
        data=st.data(),
    )
    @settings(max_examples=5_000, suppress_health_check=[HealthCheck.too_slow])
    def test_classifier_never_crashes(self, n_spans, data):
        """Fuzz the classifier: it should always return a FailureType."""
        retrieved = [
            (DOC_A, s, e)
            for s, e in data.draw(st.lists(
                st.tuples(st.integers(0, 500), st.integers(0, 500)),
                min_size=0, max_size=5,
            ))
        ]
        gt = [
            (DOC_A, s, e)
            for s, e in data.draw(st.lists(
                st.tuples(st.integers(0, 500), st.integers(0, 500)),
                min_size=1, max_size=5,
            ))
        ]
        result = classify(retrieved, gt)
        assert isinstance(result, FailureType)


class TestFront1_AdversarialPerSpanCoverage:
    """Attack _per_span_coverage directly."""

    def test_zero_width_span_filtered(self):
        """Zero-width span (5,5) is now filtered out — empty result."""
        results = _per_span_coverage({0, 1, 2, 3, 4, 5, 6}, [(5, 5)])
        assert len(results) == 0

    def test_zero_width_among_valid_spans(self):
        """Zero-width filtered, valid spans retained."""
        results = _per_span_coverage({0, 1, 2, 3, 4}, [(5, 5), (0, 10)])
        assert len(results) == 1  # only the valid span
        assert results[0]["length"] == 10
        assert results[0]["coverage"] == pytest.approx(0.5)

    def test_exactly_50pct_coverage(self):
        """Boundary: exactly 50% coverage → status should be 'covered' (>= 0.5)."""
        # span [0,10) = 10 chars, retrieved covers 5 of them
        results = _per_span_coverage({0, 1, 2, 3, 4}, [(0, 10)])
        assert results[0]["coverage"] == pytest.approx(0.5)
        assert results[0]["status"] == "covered"

    def test_just_below_50pct(self):
        """49.9...% coverage → status should be 'partial'."""
        # span [0, 100) = 100 chars, overlap with 49
        chars = set(range(49))
        results = _per_span_coverage(chars, [(0, 100)])
        assert results[0]["coverage"] == pytest.approx(0.49)
        assert results[0]["status"] == "partial"

    def test_single_char_covered(self):
        """1 char out of 100 → 1% → 'partial'."""
        results = _per_span_coverage({50}, [(0, 100)])
        assert results[0]["status"] == "partial"
        assert results[0]["coverage"] == pytest.approx(0.01)


# =====================================================================
# FRONT 2: KNOWN-ANSWER ORACLE TESTING
# =====================================================================

class TestFront2_KnownAnswerMetrics:
    """Hand-calculated metric values — independent of code."""

    def test_oracle_1_perfect_match(self):
        """gt=[100,200), retrieved=[100,200). P@1=1.0, R@1=1.0."""
        assert precision_at_k([(100, 200)], [(100, 200)], k=1) == 1.0
        assert recall_at_k([(100, 200)], [(100, 200)], k=1) == 1.0

    def test_oracle_2_half_overlap_symmetric(self):
        """gt=[0,10), retrieved=[5,15). overlap={5..9}=5 chars.
        R@1 = 5/10 = 0.5. P@1 = 5/10 = 0.5."""
        assert recall_at_k([(5, 15)], [(0, 10)], k=1) == pytest.approx(0.5)
        assert precision_at_k([(5, 15)], [(0, 10)], k=1) == pytest.approx(0.5)

    def test_oracle_3_precision_vs_recall_differ(self):
        """gt=[0,100), retrieved=[0,50). overlap=50.
        R@1 = 50/100 = 0.5. P@1 = 50/50 = 1.0."""
        assert recall_at_k([(0, 50)], [(0, 100)], k=1) == pytest.approx(0.5)
        assert precision_at_k([(0, 50)], [(0, 100)], k=1) == pytest.approx(1.0)

    def test_oracle_4_multi_span_gt(self):
        """gt = [(0,10), (20,30)] = 20 chars. retrieved=[(0,30)] = 30 chars.
        overlap = {0..9} + {20..29} = 20 chars.
        R@1 = 20/20 = 1.0. P@1 = 20/30 = 0.6667."""
        gt = [(0, 10), (20, 30)]
        ret = [(0, 30)]
        assert recall_at_k(ret, gt, k=1) == pytest.approx(1.0)
        assert precision_at_k(ret, gt, k=1) == pytest.approx(20 / 30)

    def test_oracle_5_overlapping_gt_spans(self):
        """gt = [(0,10), (5,15)] → gt_chars = {0..14} = 15 chars.
        retrieved = [(0,15)] → 15 chars. overlap = 15.
        R@1 = 15/15 = 1.0. P@1 = 15/15 = 1.0."""
        gt = [(0, 10), (5, 15)]
        ret = [(0, 15)]
        assert recall_at_k(ret, gt, k=1) == pytest.approx(1.0)
        assert precision_at_k(ret, gt, k=1) == pytest.approx(1.0)

    def test_oracle_6_k2_aggregation(self):
        """Two retrieved spans: [(0,5), (8,13)]. gt=[(0,10)].
        k=1: retrieved_chars={0..4}=5, overlap=5. R@1=5/10=0.5, P@1=5/5=1.0
        k=2: retrieved_chars={0..4,8..12}=10, overlap={0..4,8..9}=7.
        R@2=7/10=0.7, P@2=7/10=0.7."""
        ret = [(0, 5), (8, 13)]
        gt = [(0, 10)]
        assert recall_at_k(ret, gt, k=1) == pytest.approx(0.5)
        assert precision_at_k(ret, gt, k=1) == pytest.approx(1.0)
        assert recall_at_k(ret, gt, k=2) == pytest.approx(0.7)
        assert precision_at_k(ret, gt, k=2) == pytest.approx(0.7)


class TestFront2_KnownAnswerClassifier:
    """Hand-calculated failure type — the oracle knows the right answer."""

    def test_oracle_drm(self):
        """Retrieved from wrong doc entirely → DRM."""
        assert classify(
            [(DOC_B, 0, 100)], [(DOC_A, 0, 100)]
        ) == FailureType.DRM

    def test_oracle_cbf(self):
        """Right doc, zero overlap → CBF."""
        assert classify(
            [(DOC_A, 200, 300)], [(DOC_A, 0, 100)]
        ) == FailureType.CBF

    def test_oracle_sgp_2spans_cover1_miss1(self):
        """2 gt spans. Span A fully covered, Span B entirely missed → SGP."""
        retrieved = [(DOC_A, 0, 100)]
        gt = [(DOC_A, 0, 100), (DOC_A, 500, 600)]
        result = classify(retrieved, gt)
        assert result == FailureType.SGP, (
            f"Expected SGP for 2 spans (1 covered, 1 missed), got {result}"
        )

    def test_oracle_sgp_3spans_cover2_miss1(self):
        """3 gt spans. 2 fully covered, 1 entirely missed → SGP."""
        retrieved = [(DOC_A, 0, 200)]
        gt = [(DOC_A, 0, 50), (DOC_A, 100, 150), (DOC_A, 500, 600)]
        result = classify(retrieved, gt)
        assert result == FailureType.SGP

    def test_oracle_not_sgp_3spans_all_partial(self):
        """3 gt spans, ALL partially covered (each <50%, >0%) → NOT SGP.
        n_covered=0, n_missed=0 → falls to ICR."""
        # Each span is 100 chars, overlap 30 chars (30% < 50%)
        retrieved = [(DOC_A, 70, 130)]  # covers [70,130) = 60 chars
        gt = [
            (DOC_A, 0, 100),    # overlap [70,100) = 30 chars → 30% → partial
            (DOC_A, 100, 200),  # overlap [100,130) = 30 chars → 30% → partial
            (DOC_A, 200, 300),  # overlap = 0 → missed
        ]
        result = classify(retrieved, gt)
        # Actually: third span has 0 overlap → "missed". First two are "partial".
        # n_covered=0, n_missed=1. SGP requires n_covered > 0 → NOT SGP.
        # total_overlap=60, gt_total=300. 60 < 150 → ICR.
        assert result == FailureType.ICR, (
            f"Expected ICR for 3 spans (2 partial, 1 missed), got {result}"
        )

    def test_oracle_3spans_all_partial_no_missed(self):
        """3 gt spans, each 10-49% covered → NOT SGP, should be ICR.
        No missed, no covered → n_covered=0, n_missed=0."""
        # span1=[0,100) overlap with {40..59} = 20 chars → 20%
        # span2=[100,200) overlap with {140..159} = 20 chars → 20%
        # span3=[200,300) overlap with {240..259} = 20 chars → 20%
        retrieved_chars = list(range(40, 60)) + list(range(140, 160)) + list(range(240, 260))
        # Build retrieved as 3 small spans
        retrieved = [(DOC_A, 40, 60), (DOC_A, 140, 160), (DOC_A, 240, 260)]
        gt = [(DOC_A, 0, 100), (DOC_A, 100, 200), (DOC_A, 200, 300)]
        result = classify(retrieved, gt)
        # total_overlap=60, gt_total=300. 60 < 150 → ICR
        assert result == FailureType.ICR

    def test_oracle_cbf_vs_sgp_precedence(self):
        """CBF must win over SGP: if total_overlap=0, it's CBF not SGP.
        Even if there are multiple gt spans — if ALL are missed → CBF."""
        retrieved = [(DOC_A, 500, 600)]
        gt = [(DOC_A, 0, 50), (DOC_A, 100, 150)]
        result = classify(retrieved, gt)
        # Both missed (0 overlap). total_overlap=0 → CBF (before SGP check)
        assert result == FailureType.CBF

    def test_oracle_icr_single_span(self):
        """Single gt span, 10% overlap → ICR (not SGP, single span can't SGP)."""
        retrieved = [(DOC_A, 90, 110)]
        gt = [(DOC_A, 0, 100)]
        # overlap = {90..99} = 10 chars, gt_total=100, 10 < 50 → ICR
        # Also: coverage=10% → partial. n_covered=0, n_missed=0 → no SGP.
        assert classify(retrieved, gt) == FailureType.ICR

    def test_oracle_ovr(self):
        """Retrieved >= 3x gt size, with >= 50% coverage → OVR."""
        retrieved = [(DOC_A, 0, 400)]
        gt = [(DOC_A, 50, 100)]
        # gt_total=50, retrieved_total=400 >= 150 (3x50)
        # overlap=50, 50 >= 25 (50% of gt_total) → not ICR
        # → OVR
        assert classify(retrieved, gt) == FailureType.OVR

    def test_oracle_ok(self):
        """Good retrieval: high overlap, not too long → OK."""
        retrieved = [(DOC_A, 90, 210)]
        gt = [(DOC_A, 100, 200)]
        # overlap=100 chars (full), gt_total=100, retrieved_total=120
        # 100 >= 50 → not ICR. 120 < 300 → not OVR.
        assert classify(retrieved, gt) == FailureType.OK

    def test_oracle_sgp_partial_plus_missed(self):
        """EDGE: 1 span 'partial' (40%), 1 span 'covered' (80%), 1 span 'missed'.
        n_covered=1, n_missed=1 → SGP."""
        retrieved = [(DOC_A, 60, 180)]  # covers [60,180) = 120 chars
        gt = [
            (DOC_A, 0, 100),    # overlap [60,100)=40 chars → 40% → partial
            (DOC_A, 100, 200),  # overlap [100,180)=80 chars → 80% → covered
            (DOC_A, 300, 400),  # overlap = 0 → missed
        ]
        result = classify(retrieved, gt)
        assert result == FailureType.SGP

    def test_oracle_sgp_impossible_single_span(self):
        """Single gt span: SGP can NEVER trigger (confirmed in spec)."""
        # Even if covered at exactly 50% → covered, not both covered+missed
        retrieved = [(DOC_A, 50, 100)]
        gt = [(DOC_A, 0, 100)]
        result = classify(retrieved, gt)
        # overlap=50, coverage=50% → covered. n_covered=1, n_missed=0 → no SGP
        # gt_total=100, retrieved_total=50 < 300 → not OVR.
        # overlap=50 >= 50 → not ICR.
        assert result == FailureType.OK

    def test_oracle_icr_vs_ovr_precedence(self):
        """ICR checked before OVR. Can both trigger? If overlap < 50% gt AND
        retrieved >= 3x gt → ICR wins."""
        # gt=[0,100), retrieved=[0,400). overlap=100 → NOT < 50. So ICR skipped.
        # Actually need: low overlap + high retrieved total.
        retrieved = [(DOC_A, 0, 400)]
        gt = [(DOC_A, 380, 400)]
        # gt_total=20, overlap=20 (full). 20 >= 10 → not ICR. 400 >= 60 → OVR.
        assert classify(retrieved, gt) == FailureType.OVR

    def test_oracle_icr_wins_over_ovr(self):
        """Force ICR + OVR conditions simultaneously. ICR should win."""
        # gt=[0,1000), retrieved covers [0,30) but also [5000,10000)
        # overlap=30, gt_total=1000, 30 < 500 → ICR
        # retrieved_total=5030, 5030 >= 3000 → OVR would also trigger
        # But ICR is checked first → ICR wins
        retrieved = [(DOC_A, 0, 30), (DOC_A, 5000, 10000)]
        gt = [(DOC_A, 0, 1000)]
        result = classify(retrieved, gt)
        assert result == FailureType.ICR, (
            f"Expected ICR (precedence over OVR), got {result}"
        )


# =====================================================================
# FRONT 3: DIFFERENTIAL TEST vs LEGALBENCH-RAG PAPER
# =====================================================================

class TestFront3_PaperDifferential:
    """Compare our metric definitions against the LegalBench-RAG paper.

    The paper defines RCTS (Retrieval Character Token Score) as:
        RCTS@k = |chars(retrieved[:k]) ∩ chars(gt)| / |chars(gt)|

    This is identical to our recall_at_k / span_overlap.

    The paper defines RCTP (precision) as:
        RCTP@k = |chars(retrieved[:k]) ∩ chars(gt)| / |chars(retrieved[:k])|

    This is identical to our precision_at_k.

    Key question: does the paper aggregate per-query metrics then average,
    or compute a corpus-level metric?
    """

    def test_recall_matches_rcts_definition(self):
        """Verify our recall_at_k matches paper's RCTS formula exactly."""
        gt = [(100, 300)]  # 200 chars
        ret = [(150, 350)]  # 200 chars, overlap = [150,300) = 150 chars
        # RCTS@1 = 150 / 200 = 0.75
        assert recall_at_k(ret, gt, k=1) == pytest.approx(0.75)

    def test_precision_matches_rctp_definition(self):
        """Verify our precision_at_k matches paper's RCTP formula exactly."""
        gt = [(100, 300)]
        ret = [(150, 350)]
        # RCTP@1 = 150 / 200 = 0.75
        assert precision_at_k(ret, gt, k=1) == pytest.approx(0.75)

    def test_multi_chunk_k2_matches_paper(self):
        """Paper says RCTS@k uses first k CHUNKS, not first k spans.
        Our code uses retrieved_spans[:k]. If each span = one chunk, matches.
        Verify with 2 chunks."""
        gt = [(0, 100)]
        ret = [(0, 30), (60, 90)]  # 2 chunks
        # k=2: chars = {0..29, 60..89} = 60 chars. overlap with gt = 60 chars.
        # RCTS@2 = 60/100 = 0.6
        assert recall_at_k(ret, gt, k=2) == pytest.approx(0.6)
        # RCTP@2 = 60/60 = 1.0
        assert precision_at_k(ret, gt, k=2) == pytest.approx(1.0)

    def test_maud_query_count_explains_discrepancy(self):
        """Check MAUD query count: our 194-query mini split vs paper's full set.
        If paper uses 1676 queries and we use 194, P@1 difference is expected."""
        import json
        with open("data/benchmarks/maud.json") as f:
            data = json.load(f)
        n_queries = len(data["tests"])
        # Paper likely uses more queries than our 194 cap
        # This test documents the count for the audit
        assert n_queries == 194, (
            f"MAUD benchmark has {n_queries} queries — if paper uses a different "
            f"count, that explains the P@1 discrepancy"
        )

    def test_contractnli_multi_span_distribution(self):
        """Document multi-span distribution — relevant to SGP impact."""
        import json
        with open("data/benchmarks/contractnli.json") as f:
            data = json.load(f)
        span_counts = Counter(len(t["snippets"]) for t in data["tests"])
        multi = sum(c for k, c in span_counts.items() if k > 1)
        total = len(data["tests"])
        # Document for audit
        assert multi > 0, "Expected some multi-span queries in ContractNLI"
        # The distribution matters for SGP prevalence
        print(f"\nContractNLI span distribution: {dict(span_counts)}")
        print(f"Multi-span: {multi}/{total} ({100*multi/total:.1f}%)")


# =====================================================================
# FRONT 4: INVARIANT / METAMORPHIC TESTING
# =====================================================================

class TestFront4_RecallMonotonicity:
    """R@k must be non-decreasing in k."""

    @given(retrieved=valid_span_list_st, gt=valid_span_list_st)
    @settings(max_examples=10_000, suppress_health_check=[HealthCheck.too_slow])
    def test_recall_monotonic(self, retrieved, gt):
        """R@k >= R@(k-1) for all k."""
        ks = [1, 2, 4, 8]
        recalls = [recall_at_k(retrieved, gt, k) for k in ks]
        for i in range(1, len(recalls)):
            assert recalls[i] >= recalls[i - 1] - 1e-12, (
                f"R@{ks[i]}={recalls[i]} < R@{ks[i-1]}={recalls[i-1]}"
            )

    def test_recall_monotonic_known(self):
        """Known case: adding more chunks can only help recall."""
        gt = [(0, 100)]
        ret = [(50, 70), (80, 100), (0, 30)]
        r1 = recall_at_k(ret, gt, k=1)
        r2 = recall_at_k(ret, gt, k=2)
        r3 = recall_at_k(ret, gt, k=3)
        assert r1 <= r2 <= r3


class TestFront4_PrecisionMonotonicity:
    """P@k non-increasing is NOT a mathematical invariant for char-level metrics.
    This front tests whether the sanity check is too strict."""

    def test_precision_can_increase_with_k(self):
        """AUDIT: Demonstrate P@k CAN legitimately increase with k.
        k=1: chunk with 0 overlap. k=2: add chunk with full overlap.
        P@1=0, P@2>0. This is a legitimate result, not a bug."""
        gt = [(100, 200)]  # 100 chars
        ret = [(0, 50), (100, 200)]  # first chunk: 0 overlap, second: 100 overlap
        p1 = precision_at_k(ret, gt, k=1)
        p2 = precision_at_k(ret, gt, k=2)
        # P@1 = 0/50 = 0.0, P@2 = 100/150 = 0.6667
        assert p1 == pytest.approx(0.0)
        assert p2 == pytest.approx(100 / 150)
        # P increased! This is a legitimate result.
        assert p2 > p1, (
            "This test demonstrates that P@k CAN increase with k "
            "in character-level precision — the sanity check is too strict"
        )

    def test_sanity_check_no_false_positive(self):
        """After fix: P@k monotonicity is NOT checked, so this passes cleanly."""
        gt = [(100, 200)]
        ret = [(0, 50), (100, 200)]
        result = compute_all_k(ret, gt, k_values=[1, 2])
        # P@1=0.0, P@2=0.6667 → precision increased — legitimate
        assert check_monotonicity(result) is True
        sanity = run_sanity_checks(result)
        assert sanity.passed is True
        assert len(sanity.violations) == 0


class TestFront4_AddIrrelevantNeverDecreaseRecall:
    """Adding an irrelevant chunk must never decrease recall."""

    @given(retrieved=valid_span_list_st, gt=valid_span_list_st)
    @settings(max_examples=10_000, suppress_health_check=[HealthCheck.too_slow])
    def test_add_irrelevant_chunk(self, retrieved, gt):
        """Appending a non-overlapping chunk to the end → recall unchanged."""
        base_r = recall_at_k(retrieved, gt, k=len(retrieved))
        # Add a chunk far away from any gt span
        extended = retrieved + [(9000, 9100)]
        extended_r = recall_at_k(extended, gt, k=len(extended))
        assert extended_r >= base_r - 1e-12


class TestFront4_CoveredCountBound:
    """Per-span covered count must never exceed total gt span count."""

    @given(data=st.data())
    @settings(max_examples=5_000, suppress_health_check=[HealthCheck.too_slow])
    def test_covered_count_bounded(self, data):
        n_gt = data.draw(st.integers(min_value=1, max_value=8))
        gt_spans = data.draw(st.lists(
            valid_span_st, min_size=n_gt, max_size=n_gt
        ))
        retrieved_chars = set()
        for s, e in data.draw(st.lists(
            valid_span_st, min_size=0, max_size=8
        )):
            retrieved_chars.update(range(s, e))
        results = _per_span_coverage(retrieved_chars, gt_spans)
        assert len(results) == n_gt
        n_covered = sum(1 for r in results if r["status"] == "covered")
        n_missed = sum(1 for r in results if r["status"] == "missed")
        n_partial = sum(1 for r in results if r["status"] == "partial")
        assert n_covered + n_missed + n_partial == n_gt, (
            "Statuses must partition spans: "
            f"covered={n_covered} missed={n_missed} partial={n_partial} total={n_gt}"
        )


class TestFront4_FailureTypeExhaustivePartition:
    """The 6 failure types must partition all inputs:
    - Every input maps to exactly one type (exhaustive)
    - No input maps to two types (mutually exclusive — guaranteed by first-match)
    """

    @given(data=st.data())
    @settings(max_examples=5_000, suppress_health_check=[HealthCheck.too_slow])
    def test_always_returns_phase1_type(self, data):
        """classify always returns a Phase 1 type (never Phase 2, never None)."""
        gt = [
            (DOC_A, s, e)
            for s, e in data.draw(st.lists(valid_span_st, min_size=1, max_size=5))
        ]
        # Mix of doc_a and doc_b retrieved
        doc_choice = data.draw(st.sampled_from([DOC_A, DOC_B]))
        retrieved = [
            (doc_choice, s, e)
            for s, e in data.draw(st.lists(valid_span_st, min_size=0, max_size=5))
        ]
        result = classify(retrieved, gt)
        phase1_types = {
            FailureType.DRM, FailureType.CBF, FailureType.SGP,
            FailureType.ICR, FailureType.OVR, FailureType.OK,
        }
        assert result in phase1_types, f"Got non-Phase-1 type: {result}"

    def test_all_types_reachable(self):
        """Every Phase 1 failure type is reachable with some input."""
        cases = {
            FailureType.DRM: ([(DOC_B, 0, 100)], [(DOC_A, 0, 100)]),
            FailureType.CBF: ([(DOC_A, 500, 600)], [(DOC_A, 0, 100)]),
            FailureType.SGP: ([(DOC_A, 0, 100)], [(DOC_A, 0, 100), (DOC_A, 500, 600)]),
            FailureType.ICR: ([(DOC_A, 90, 110)], [(DOC_A, 0, 100)]),
            FailureType.OVR: ([(DOC_A, 0, 400)], [(DOC_A, 50, 100)]),
            FailureType.OK:  ([(DOC_A, 90, 210)], [(DOC_A, 100, 200)]),
        }
        for expected_type, (ret, gt) in cases.items():
            result = classify(ret, gt)
            assert result == expected_type, (
                f"Expected {expected_type}, got {result}"
            )


class TestFront4_MetricResultIntegrity:
    """MetricResult invariants."""

    def test_mismatched_keys_rejected(self):
        """p_at_k and r_at_k must have same keys."""
        with pytest.raises(ValueError):
            MetricResult(
                p_at_k={1: 0.5, 2: 0.3},
                r_at_k={1: 0.5},
                eval_mode="SPAN_OVERLAP",
            )

    @given(retrieved=valid_span_list_st, gt=valid_span_list_st)
    @settings(max_examples=5_000, suppress_health_check=[HealthCheck.too_slow])
    def test_compute_all_k_bounded(self, retrieved, gt):
        """All values in compute_all_k result must be in [0, 1]."""
        result = compute_all_k(retrieved, gt)
        for k in result.p_at_k:
            assert 0.0 <= result.p_at_k[k] <= 1.0, f"P@{k} out of bounds"
            assert 0.0 <= result.r_at_k[k] <= 1.0, f"R@{k} out of bounds"


class TestFront4_SanityCheckProperties:
    """Sanity check meta-properties."""

    def test_no_violations_means_passed(self):
        """SanityResult.passed must be True iff violations is empty."""
        good = MetricResult(
            p_at_k={1: 1.0, 2: 0.5, 4: 0.25},
            r_at_k={1: 0.5, 2: 0.7, 4: 0.9},
            eval_mode="SPAN_OVERLAP",
        )
        result = run_sanity_checks(good)
        assert result.passed is True
        assert len(result.violations) == 0

    def test_improvement_threshold_symmetric_concern(self):
        """AUDIT: check_improvement only flags IMPROVEMENT, not regression.
        A massive regression (new much WORSE than baseline) passes silently.
        Is this intentional?"""
        baseline = MetricResult(p_at_k={1: 0.9}, r_at_k={1: 0.9}, eval_mode="SPAN_OVERLAP")
        terrible = MetricResult(p_at_k={1: 0.0}, r_at_k={1: 0.0}, eval_mode="SPAN_OVERLAP")
        # Regression of -0.9 at every k
        from core.measurement.sanity import check_improvement
        result = check_improvement(terrible, baseline, threshold=0.3)
        # Returns True because the IMPROVEMENT is negative (regression)
        # This means a catastrophic regression passes the improvement check
        assert result is True, (
            "check_improvement does NOT flag regressions — only suspicious improvements. "
            "Intentional? A 90% regression would pass silently."
        )
