"""Precision/recall at k over character spans, plus MetricResult."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.measurement.span_overlap import span_overlap, MissingGroundTruthError


@dataclass(frozen=True)
class MetricResult:
    """Container for span-overlap metrics across multiple k values.

    ``eval_mode`` must be set explicitly — there is no default.
    ``p_at_k`` and ``r_at_k`` must have identical key sets.
    """

    p_at_k: dict[int, float]
    r_at_k: dict[int, float]
    eval_mode: Literal["SPAN_OVERLAP", "LLM_JUDGE"]

    def __post_init__(self) -> None:
        if self.p_at_k.keys() != self.r_at_k.keys():
            raise ValueError(
                f"p_at_k and r_at_k must have the same keys, "
                f"got {sorted(self.p_at_k.keys())} vs {sorted(self.r_at_k.keys())}"
            )


def precision_at_k(
    retrieved_spans: list[tuple[int, int]],
    gt_spans: list[tuple[int, int]],
    k: int,
) -> float:
    """Fraction of *retrieved* characters (top-k) that are in ground truth.

    Uses the first *k* spans from *retrieved_spans*.

    Raises ``MissingGroundTruthError`` if *gt_spans* is empty/None.
    Returns 0.0 if the top-k retrieved spans cover zero characters.
    """
    if not gt_spans:
        raise MissingGroundTruthError(
            "Cannot compute precision: ground-truth spans are empty or None"
        )

    top_k = retrieved_spans[:k]
    if not top_k:
        return 0.0

    gt_chars: set[int] = set()
    for start, end in gt_spans:
        gt_chars.update(range(start, end))

    retrieved_chars: set[int] = set()
    for start, end in top_k:
        retrieved_chars.update(range(start, end))

    if not retrieved_chars:
        return 0.0

    intersection = gt_chars & retrieved_chars
    return len(intersection) / len(retrieved_chars)


def recall_at_k(
    retrieved_spans: list[tuple[int, int]],
    gt_spans: list[tuple[int, int]],
    k: int,
) -> float:
    """Fraction of ground-truth characters covered by the top-k retrieved spans.

    Equivalent to ``span_overlap(retrieved_spans[:k], gt_spans)``.
    """
    return span_overlap(retrieved_spans[:k], gt_spans)


def compute_all_k(
    retrieved_spans: list[tuple[int, int]],
    gt_spans: list[tuple[int, int]],
    k_values: list[int] | None = None,
) -> MetricResult:
    """Compute precision and recall at every k in *k_values*.

    Parameters
    ----------
    k_values:
        Defaults to ``[1, 2, 4, 8, 16, 32, 64]``.

    Returns a ``MetricResult`` with ``eval_mode="SPAN_OVERLAP"``.
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32, 64]

    p: dict[int, float] = {}
    r: dict[int, float] = {}
    for k in k_values:
        p[k] = precision_at_k(retrieved_spans, gt_spans, k)
        r[k] = recall_at_k(retrieved_spans, gt_spans, k)

    return MetricResult(p_at_k=p, r_at_k=r, eval_mode="SPAN_OVERLAP")
