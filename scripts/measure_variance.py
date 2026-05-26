#!/usr/bin/env python3
"""Measure baseline variance: run the same config N times, report stats.

Uses the existing Qdrant Cloud collection (skip_index=True) so there is
zero embedding cost — only BM25 indexing + retrieval + measurement.

Usage:
    python scripts/measure_variance.py --runs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation.run_eval import evaluate_config, K_VALUES
from core.measurement.taxonomy import FailureType

FAILURE_TYPES = ["DRM", "CBF", "SGP", "ICR", "OVR", "OK"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    config = {
        "chunk_size": 512,
        "chunk_overlap": 128,
        "bm25_top_k": 32,
        "dense_top_k": 32,
        "fusion_top_n": 64,
        "retrieval_mode": "hybrid",
    }

    all_p1: list[float] = []
    all_r8: list[float] = []
    all_failures: list[dict[str, int]] = []

    # Collect all P@k and R@k for full table
    all_p: dict[int, list[float]] = {k: [] for k in K_VALUES}
    all_r: dict[int, list[float]] = {k: [] for k in K_VALUES}

    for i in range(args.runs):
        print(f"\n{'='*50}")
        print(f"  Run {i+1}/{args.runs}")
        print(f"{'='*50}")
        result, failures = evaluate_config(
            config, "contractnli", args.data_dir, skip_index=True
        )
        p1 = result.p_at_k[1] * 100
        r8 = result.r_at_k[8] * 100
        all_p1.append(p1)
        all_r8.append(r8)
        all_failures.append(failures)

        for k in K_VALUES:
            all_p[k].append(result.p_at_k[k] * 100)
            all_r[k].append(result.r_at_k[k] * 100)

        n = sum(failures.values())
        print(f"  P@1={p1:.4f}%  R@8={r8:.4f}%")
        for ft in FAILURE_TYPES:
            c = failures.get(ft, 0)
            print(f"    {ft}: {c} ({c/n*100:.1f}%)")

    # ── Statistics ──
    import statistics

    print(f"\n{'='*60}")
    print(f"  VARIANCE REPORT ({args.runs} runs)")
    print(f"{'='*60}")

    for label, values in [("P@1", all_p1), ("R@8", all_r8)]:
        mean = statistics.mean(values)
        if len(values) > 1:
            sd = statistics.stdev(values)
        else:
            sd = 0.0
        print(f"\n  {label}:")
        print(f"    values: {[f'{v:.4f}' for v in values]}")
        print(f"    mean:   {mean:.4f}%")
        print(f"    stdev:  {sd:.4f}pp")
        print(f"    range:  [{min(values):.4f}, {max(values):.4f}]")
        print(f"    spread: {max(values)-min(values):.4f}pp")

    # Full k table
    print(f"\n  Per-k statistics:")
    print(f"  {'k':>4}  {'P@k mean':>10}  {'P@k sd':>8}  {'R@k mean':>10}  {'R@k sd':>8}")
    for k in K_VALUES:
        pm = statistics.mean(all_p[k])
        ps = statistics.stdev(all_p[k]) if len(all_p[k]) > 1 else 0.0
        rm = statistics.mean(all_r[k])
        rs = statistics.stdev(all_r[k]) if len(all_r[k]) > 1 else 0.0
        print(f"  {k:>4}  {pm:>9.4f}%  {ps:>7.4f}  {rm:>9.4f}%  {rs:>7.4f}")

    # Failure type variance
    print(f"\n  Failure type counts (per run):")
    print(f"  {'Type':>4}", end="")
    for i in range(args.runs):
        print(f"  {'Run'+str(i+1):>6}", end="")
    print(f"  {'mean':>7}  {'sd':>6}")

    for ft in FAILURE_TYPES:
        counts = [f.get(ft, 0) for f in all_failures]
        mean = statistics.mean(counts)
        sd = statistics.stdev(counts) if len(counts) > 1 else 0.0
        print(f"  {ft:>4}", end="")
        for c in counts:
            print(f"  {c:>6}", end="")
        print(f"  {mean:>7.2f}  {sd:>6.2f}")

    # Suggested VARIANCE_FLOOR_PP
    p1_sd = statistics.stdev(all_p1) if len(all_p1) > 1 else 0.0
    r8_sd = statistics.stdev(all_r8) if len(all_r8) > 1 else 0.0
    print(f"\n  Suggested VARIANCE_FLOOR_PP (3-sigma):")
    print(f"    p_at_1: {3*p1_sd:.2f}pp  (sd={p1_sd:.4f})")
    print(f"    r_at_8: {3*r8_sd:.2f}pp  (sd={r8_sd:.4f})")

    # Failure vector floor: use max observed sd across failure types
    ft_sds = []
    for ft in FAILURE_TYPES:
        counts = [f.get(ft, 0) for f in all_failures]
        if len(counts) > 1:
            ft_sds.append(statistics.stdev(counts))
    if ft_sds:
        max_ft_sd_pct = max(ft_sds) / 194 * 100
        print(f"    failure types: {3*max_ft_sd_pct:.2f}pp  (max count sd={max(ft_sds):.2f}, N=194)")


if __name__ == "__main__":
    main()
