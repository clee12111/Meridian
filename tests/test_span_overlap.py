"""Property-based and unit tests for span_overlap."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.measurement.span_overlap import span_overlap, MissingGroundTruthError


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A span is a half-open [start, end) with start < end, small enough to
# avoid blowing up memory when expanded to a char set.
span_st = st.tuples(
    st.integers(min_value=0, max_value=500),
    st.integers(min_value=1, max_value=500),
).map(lambda t: (min(t), max(t)) if t[0] != t[1] else (t[0], t[0] + 1))

span_list_st = st.lists(span_st, min_size=1, max_size=10)


# ---------------------------------------------------------------------------
# Property tests (10 000 examples each)
# ---------------------------------------------------------------------------


@given(retrieved=span_list_st, gt=span_list_st)
@settings(max_examples=10_000)
def test_overlap_bounded_0_1(retrieved, gt):
    """0.0 <= span_overlap(a, b) <= 1.0 for all inputs."""
    result = span_overlap(retrieved, gt)
    assert 0.0 <= result <= 1.0


@given(spans=span_list_st)
@settings(max_examples=10_000)
def test_idempotence(spans):
    """span_overlap(a, a) == 1.0."""
    assert span_overlap(spans, spans) == 1.0


@given(retrieved=span_list_st, gt=span_list_st)
@settings(max_examples=10_000)
def test_extending_retrieved_never_decreases_recall(retrieved, gt):
    """Adding more retrieved spans cannot decrease recall."""
    base = span_overlap(retrieved, gt)
    extended = retrieved + [(0, 1)]  # add a tiny extra span
    result = span_overlap(extended, gt)
    assert result >= base - 1e-12  # float tolerance


@given(gt=span_list_st, extra=span_st)
@settings(max_examples=10_000)
def test_superset_retrieved_has_full_recall(gt, extra):
    """If retrieved is a superset of gt, recall == 1.0."""
    retrieved = gt + [extra]
    assert span_overlap(retrieved, gt) == 1.0


# ---------------------------------------------------------------------------
# Unit tests — MissingGroundTruthError
# ---------------------------------------------------------------------------


def test_empty_gt_raises():
    with pytest.raises(MissingGroundTruthError):
        span_overlap([(0, 10)], [])


def test_none_gt_raises():
    with pytest.raises(MissingGroundTruthError):
        span_overlap([(0, 10)], None)  # type: ignore[arg-type]


def test_zero_width_gt_raises():
    """Spans like (5, 5) cover zero characters."""
    with pytest.raises(MissingGroundTruthError):
        span_overlap([(0, 10)], [(5, 5)])


# ---------------------------------------------------------------------------
# Unit tests — known values
# ---------------------------------------------------------------------------


def test_no_overlap():
    assert span_overlap([(0, 5)], [(10, 20)]) == 0.0


def test_full_overlap():
    assert span_overlap([(10, 20)], [(10, 20)]) == 1.0


def test_partial_overlap():
    # gt = [10, 20) = 10 chars; retrieved = [15, 25) → overlap = [15, 20) = 5 chars
    assert span_overlap([(15, 25)], [(10, 20)]) == pytest.approx(0.5)


def test_empty_retrieved():
    assert span_overlap([], [(10, 20)]) == 0.0


def test_multiple_gt_spans():
    # gt covers chars 0..5 and 10..15 = 10 chars total
    # retrieved covers 0..15 = 15 chars → overlap = 10 chars
    assert span_overlap([(0, 15)], [(0, 5), (10, 15)]) == 1.0


def test_multiple_retrieved_spans():
    # gt = [0, 10) = 10 chars
    # retrieved = [0, 3) + [7, 10) = 6 chars overlap
    assert span_overlap([(0, 3), (7, 10)], [(0, 10)]) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Note on symmetry
# ---------------------------------------------------------------------------
# span_overlap is NOT symmetric: it measures gt coverage by retrieved.
# span_overlap(a, b) != span_overlap(b, a) in general.
# This is by design — it's recall, not a symmetric similarity.

def test_asymmetry():
    """span_overlap is directional: it's recall (gt coverage), not symmetric."""
    a = [(0, 100)]
    b = [(0, 10)]
    # a covers all of b → span_overlap(a, b) == 1.0
    assert span_overlap(a, b) == 1.0
    # b covers 10% of a → span_overlap(b, a) == 0.1
    assert span_overlap(b, a) == pytest.approx(0.1)
