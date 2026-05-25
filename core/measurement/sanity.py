"""Sanity checks for MetricResult: monotonicity and improvement gating."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.measurement.metrics import MetricResult


@dataclass
class SanityResult:
    """Outcome of a sanity check. Every violation is named explicitly."""

    passed: bool
    violations: list[str] = field(default_factory=list)


def check_monotonicity(metric_result: MetricResult) -> bool:
    """R@k must be non-decreasing; P@k must be non-increasing as k grows."""
    sorted_ks = sorted(metric_result.r_at_k.keys())

    for i in range(1, len(sorted_ks)):
        prev_k, cur_k = sorted_ks[i - 1], sorted_ks[i]

        if metric_result.r_at_k[cur_k] < metric_result.r_at_k[prev_k]:
            return False
        if metric_result.p_at_k[cur_k] > metric_result.p_at_k[prev_k]:
            return False

    return True


def check_improvement(
    new: MetricResult,
    baseline: MetricResult,
    threshold: float,
) -> bool:
    """Return False (quarantine) if any k shows improvement > threshold.

    "Improvement" means ``new.r_at_k[k] - baseline.r_at_k[k] > threshold``
    OR ``new.p_at_k[k] - baseline.p_at_k[k] > threshold``.

    Returns True if improvement is within bounds at every k.
    """
    common_ks = set(new.r_at_k.keys()) & set(baseline.r_at_k.keys())

    for k in common_ks:
        if new.r_at_k[k] - baseline.r_at_k[k] > threshold:
            return False
        if new.p_at_k[k] - baseline.p_at_k[k] > threshold:
            return False

    return True


def run_sanity_checks(
    metric_result: MetricResult,
    baseline: MetricResult | None = None,
    improvement_threshold: float = 0.3,
) -> SanityResult:
    """Run all sanity checks and return a ``SanityResult`` with violations."""
    violations: list[str] = []
    sorted_ks = sorted(metric_result.r_at_k.keys())

    # Monotonicity checks
    for i in range(1, len(sorted_ks)):
        prev_k, cur_k = sorted_ks[i - 1], sorted_ks[i]

        if metric_result.r_at_k[cur_k] < metric_result.r_at_k[prev_k]:
            violations.append(
                f"R@{cur_k} ({metric_result.r_at_k[cur_k]:.4f}) < "
                f"R@{prev_k} ({metric_result.r_at_k[prev_k]:.4f}): "
                f"recall is not non-decreasing"
            )
        if metric_result.p_at_k[cur_k] > metric_result.p_at_k[prev_k]:
            violations.append(
                f"P@{cur_k} ({metric_result.p_at_k[cur_k]:.4f}) > "
                f"P@{prev_k} ({metric_result.p_at_k[prev_k]:.4f}): "
                f"precision is not non-increasing"
            )

    # Improvement checks
    if baseline is not None:
        common_ks = set(metric_result.r_at_k.keys()) & set(baseline.r_at_k.keys())
        for k in sorted(common_ks):
            r_delta = metric_result.r_at_k[k] - baseline.r_at_k[k]
            if r_delta > improvement_threshold:
                violations.append(
                    f"R@{k} improved by {r_delta:.4f} "
                    f"(> threshold {improvement_threshold}): quarantine"
                )
            p_delta = metric_result.p_at_k[k] - baseline.p_at_k[k]
            if p_delta > improvement_threshold:
                violations.append(
                    f"P@{k} improved by {p_delta:.4f} "
                    f"(> threshold {improvement_threshold}): quarantine"
                )

    return SanityResult(passed=len(violations) == 0, violations=violations)
