#!/usr/bin/env python3
"""Experiment 2: Chunk-size sweep over ContractNLI and MAUD.

Hypothesis: Larger chunks reduce CBF (chunk boundary failure) by keeping
conditional clauses intact. Expected tradeoff: OVR (over-retrieval) rises.
The real question is whether CBF drops MORE than OVR rises.

Sweep: chunk_size in {512, 1024, 2048}, overlap proportional (25%):
  512/128, 1024/256, 2048/512.

Sanity check: 512-char ContractNLI must reproduce the locked baseline
(P@1 9.24%, R@8 50.41%) or the harness is inconsistent and we stop.

Cost note: Each (dataset, chunk_size) pair requires re-chunking and
re-embedding. ContractNLI + MAUD are ~5k chunks total at 512 — cheap.
CUAD excluded (96k chunks / 95% of embedding cost).

Usage:
    python scripts/run_experiment2.py [--data-dir data/]
    python scripts/run_experiment2.py --bm25-only
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

import os

from core.evaluation.run_eval import evaluate_config, K_VALUES
from core.measurement.taxonomy import FailureType

# --- Sweep configuration ---
DATASETS = ["contractnli", "maud"]
CHUNK_CONFIGS = [
    {"chunk_size": 512, "chunk_overlap": 128},
    {"chunk_size": 1024, "chunk_overlap": 256},
    {"chunk_size": 2048, "chunk_overlap": 512},
]

# Locked baseline for sanity check (512-char ContractNLI hybrid)
LOCKED_P1 = 9.24
LOCKED_R8 = 50.41
SANITY_TOLERANCE = 0.01  # absolute % tolerance for float comparison


def _format_failures(failure_counts: dict[str, int], n: int) -> dict[str, str]:
    """Format failure distribution as {type: 'count (pct%)'}."""
    result = {}
    for ft in FailureType:
        if ft in (FailureType.DTM, FailureType.XRF):
            continue
        count = failure_counts.get(ft.name, 0)
        pct = count / n * 100 if n > 0 else 0
        result[ft.name] = f"{count} ({pct:.1f}%)"
    return result


def run_sweep(data_dir: str = "data", bm25_only: bool = False) -> None:
    retrieval_mode = "hybrid"
    if bm25_only:
        retrieval_mode = "bm25"
    elif not os.environ.get("VOYAGE_API_KEY"):
        print("VOYAGE_API_KEY not set — falling back to BM25-only.")
        retrieval_mode = "bm25"

    mode_label = "Hybrid" if retrieval_mode == "hybrid" else "BM25-only"
    print("=" * 70)
    print(f"EXPERIMENT 2: Chunk-Size Sweep ({mode_label})")
    print("=" * 70)

    # Collect all results: {(dataset, chunk_size): (metric_result, failure_counts, n)}
    results: dict[tuple[str, int], tuple] = {}

    for chunk_cfg in CHUNK_CONFIGS:
        cs = chunk_cfg["chunk_size"]
        co = chunk_cfg["chunk_overlap"]
        for ds in DATASETS:
            print(f"\n{'-' * 60}")
            print(f"  {ds} | chunk_size={cs} overlap={co}")
            print(f"{'-' * 60}")

            config = {
                "chunk_size": cs,
                "chunk_overlap": co,
                "bm25_top_k": 32,
                "dense_top_k": 32,
                "fusion_top_n": 64,
                "retrieval_mode": retrieval_mode,
            }

            metric_result, failure_counts = evaluate_config(config, ds, data_dir)
            n = sum(failure_counts.values())
            results[(ds, cs)] = (metric_result, failure_counts, n)

            # Print per-run summary
            p1 = metric_result.p_at_k[1] * 100
            r8 = metric_result.r_at_k[8] * 100
            print(f"  P@1: {p1:.2f}%  R@8: {r8:.2f}%  (n={n})")
            formatted = _format_failures(failure_counts, n)
            for ft_name, ft_str in formatted.items():
                print(f"    {ft_name:>4}: {ft_str}")

    # --- Sanity check: 512 ContractNLI ---
    print(f"\n{'=' * 70}")
    print("SANITY CHECK: 512-char ContractNLI vs locked baseline")
    print("=" * 70)

    m512, _, _ = results[("contractnli", 512)]
    actual_p1 = m512.p_at_k[1] * 100
    actual_r8 = m512.r_at_k[8] * 100

    p1_match = abs(actual_p1 - LOCKED_P1) < SANITY_TOLERANCE
    r8_match = abs(actual_r8 - LOCKED_R8) < SANITY_TOLERANCE

    print(f"  P@1: {actual_p1:.2f}% (locked: {LOCKED_P1:.2f}%)  {'MATCH' if p1_match else 'MISMATCH'}")
    print(f"  R@8: {actual_r8:.2f}% (locked: {LOCKED_R8:.2f}%)  {'MATCH' if r8_match else 'MISMATCH'}")

    if not (p1_match and r8_match):
        print("  >>> SANITY FAILED — harness inconsistent, results unreliable <<<")
    else:
        print("  >>> SANITY PASSED <<<")

    # --- Summary table ---
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 2 RESULTS ({mode_label})")
    print("=" * 70)

    # Header
    print(f"\n{'Dataset':<14} {'Chunk':>5} {'P@1':>7} {'R@8':>7}   "
          f"{'DRM':>10} {'CBF':>10} {'ICR':>10} {'OVR':>10} {'OK':>10}")
    print(f"{'-'*14} {'-'*5} {'-'*7} {'-'*7}   "
          f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for ds in DATASETS:
        for chunk_cfg in CHUNK_CONFIGS:
            cs = chunk_cfg["chunk_size"]
            metric_result, failure_counts, n = results[(ds, cs)]
            p1 = metric_result.p_at_k[1] * 100
            r8 = metric_result.r_at_k[8] * 100
            formatted = _format_failures(failure_counts, n)

            print(
                f"{ds:<14} {cs:>5} {p1:>6.2f}% {r8:>6.2f}%   "
                f"{formatted['DRM']:>10} {formatted['CBF']:>10} "
                f"{formatted['ICR']:>10} {formatted['OVR']:>10} {formatted['OK']:>10}"
            )
        print()  # blank line between datasets

    # --- CBF/OVR delta analysis ---
    print(f"\n{'=' * 70}")
    print("CBF / OVR TRADEOFF ANALYSIS")
    print("=" * 70)

    for ds in DATASETS:
        print(f"\n  {ds}:")
        base_m, base_f, base_n = results[(ds, 512)]
        base_cbf = base_f.get("CBF", 0) / base_n * 100
        base_ovr = base_f.get("OVR", 0) / base_n * 100

        for chunk_cfg in CHUNK_CONFIGS[1:]:  # skip 512 (baseline)
            cs = chunk_cfg["chunk_size"]
            m, f, n = results[(ds, cs)]
            cbf = f.get("CBF", 0) / n * 100
            ovr = f.get("OVR", 0) / n * 100

            cbf_delta = cbf - base_cbf
            ovr_delta = ovr - base_ovr
            cbf_rel = cbf_delta / base_cbf * 100 if base_cbf > 0 else float('inf')

            print(f"    {cs}-char vs 512-char:")
            print(f"      CBF: {base_cbf:.1f}% -> {cbf:.1f}% (delta {cbf_delta:+.1f}pp, {cbf_rel:+.1f}% relative)")
            print(f"      OVR: {base_ovr:.1f}% -> {ovr:.1f}% (delta {ovr_delta:+.1f}pp)")
            net = abs(cbf_delta) - abs(ovr_delta)
            direction = "CBF drop > OVR rise" if cbf_delta < 0 and net > 0 else "OVR rise >= CBF drop"
            print(f"      Net: {direction}")

    # --- Full P@k / R@k tables ---
    print(f"\n{'=' * 70}")
    print("FULL P@k / R@k TABLES")
    print("=" * 70)

    for ds in DATASETS:
        print(f"\n  {ds}:")
        print(f"  {'k':>4}", end="")
        for chunk_cfg in CHUNK_CONFIGS:
            cs = chunk_cfg["chunk_size"]
            print(f"  {'P@k(' + str(cs) + ')':>11}  {'R@k(' + str(cs) + ')':>11}", end="")
        print()

        for k in K_VALUES:
            print(f"  {k:>4}", end="")
            for chunk_cfg in CHUNK_CONFIGS:
                cs = chunk_cfg["chunk_size"]
                m, _, _ = results[(ds, cs)]
                print(f"  {m.p_at_k[k]*100:>10.2f}%  {m.r_at_k[k]*100:>10.2f}%", end="")
            print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 2: chunk-size sweep")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    run_sweep(data_dir=args.data_dir, bm25_only=args.bm25_only)


if __name__ == "__main__":
    main()
