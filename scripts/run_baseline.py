#!/usr/bin/env python3
"""Phase 2 baseline: ContractNLI retrieval evaluation.

Manual end-to-end run. Not automated — run once to validate the
Phase 2 gate.

Usage:
    python scripts/run_baseline.py [--data-dir data/]
    python scripts/run_baseline.py --bm25-only   # skip dense retrieval

Locked gate (hybrid baseline):
    P@1 in [6.8, 10.8]   (8.84 +/- 2.0)
    R@8 in [47.4, 53.4]  (50.41 +/- 3.0)

Replication mode (--replicate):
    python scripts/run_baseline.py --replicate
    Uses text-embedding-3-large (OpenAI) to match ZeroEntropy paper.
    Separate Qdrant collection "contractnli_replication".
    Paper targets: P@1 6.41%, R@8 26.30%.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation.run_eval import evaluate_config, K_VALUES
from core.measurement.taxonomy import FailureType


def _print_results(label: str, metric_result, failure_counts: dict[str, int], n_queries: int) -> None:
    """Print metric table and failure distribution for one run."""
    print(f"\n--- {label} ---")
    print(f"{'k':>4}  {'P@k':>8}  {'R@k':>8}")
    print(f"{'---':>4}  {'---':>8}  {'---':>8}")
    for k in K_VALUES:
        print(f"{k:>4}  {metric_result.p_at_k[k] * 100:>7.2f}%  {metric_result.r_at_k[k] * 100:>7.2f}%")

    print(f"\n  Failure distribution:")
    for ft in FailureType:
        if ft in (FailureType.DTM, FailureType.XRF):
            continue
        count = failure_counts.get(ft.name, 0)
        pct = count / n_queries * 100
        print(f"    {ft.name:>4}: {count:>4} ({pct:>5.1f}%)")


def _print_comparison(
    bm25_result, bm25_failures: dict[str, int], bm25_n: int,
    hybrid_result, hybrid_failures: dict[str, int], hybrid_n: int,
) -> None:
    """Print side-by-side comparison."""
    print(f"\n{'=' * 60}")
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print(f"{'k':>4}  {'BM25 P@k':>10}  {'Hybrid P@k':>12}  {'BM25 R@k':>10}  {'Hybrid R@k':>12}")
    print(f"{'---':>4}  {'---':>10}  {'---':>12}  {'---':>10}  {'---':>12}")
    for k in K_VALUES:
        bp = bm25_result.p_at_k[k] * 100
        hp = hybrid_result.p_at_k[k] * 100
        br = bm25_result.r_at_k[k] * 100
        hr = hybrid_result.r_at_k[k] * 100
        print(f"{k:>4}  {bp:>9.2f}%  {hp:>11.2f}%  {br:>9.2f}%  {hr:>11.2f}%")

    # Failure comparison
    print(f"\n  Failure distribution:")
    print(f"  {'Type':>4}  {'BM25':>12}  {'Hybrid':>12}")
    for ft in FailureType:
        if ft in (FailureType.DTM, FailureType.XRF):
            continue
        bc = bm25_failures.get(ft.name, 0)
        hc = hybrid_failures.get(ft.name, 0)
        print(f"  {ft.name:>4}  {bc:>4} ({bc/bm25_n*100:>5.1f}%)  {hc:>4} ({hc/hybrid_n*100:>5.1f}%)")


P1_GATE = (6.8, 10.8)    # 8.84 +/- 2.0
R8_GATE = (47.4, 53.4)   # 50.41 +/- 3.0


def _gate_check(label: str, metric_result) -> bool:
    """Print gate check and return whether it passed."""
    mean_p1 = metric_result.p_at_k[1] * 100
    mean_r8 = metric_result.r_at_k[8] * 100

    print(f"\nGATE CHECK ({label})")
    print(f"  P@1: {mean_p1:.2f}%  (range: [{P1_GATE[0]}, {P1_GATE[1]}])")
    print(f"  R@8: {mean_r8:.2f}%  (range: [{R8_GATE[0]}, {R8_GATE[1]}])")

    p1_ok = P1_GATE[0] <= mean_p1 <= P1_GATE[1]
    r8_ok = R8_GATE[0] <= mean_r8 <= R8_GATE[1]

    if p1_ok and r8_ok:
        print(f"  >>> GATE PASSED <<<")
        return True
    else:
        violations = []
        if not p1_ok:
            violations.append(f"P@1={mean_p1:.2f}% outside [{P1_GATE[0]}, {P1_GATE[1]}]")
        if not r8_ok:
            violations.append(f"R@8={mean_r8:.2f}% outside [{R8_GATE[0]}, {R8_GATE[1]}]")
        print(f"  >>> GATE FAILED: {'; '.join(violations)} <<<")
        print("  QUARANTINE — investigate before proceeding.")
        return False


def run_baseline(
    data_dir: str = "data",
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    bm25_top_k: int = 32,
    dense_top_k: int = 32,
    fusion_top_n: int = 64,
    bm25_only: bool = False,
) -> None:
    print("=" * 60)
    print("PHASE 2 BASELINE: ContractNLI")
    print("=" * 60)

    base_config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "bm25_top_k": bm25_top_k,
        "dense_top_k": dense_top_k,
        "fusion_top_n": fusion_top_n,
    }

    # BM25 evaluation (always runs)
    print(f"\nRunning BM25-only evaluation...")
    bm25_config = {**base_config, "retrieval_mode": "bm25"}
    bm25_result, bm25_failures = evaluate_config(bm25_config, "contractnli", data_dir)
    bm25_n = sum(bm25_failures.values())
    _print_results("BM25-only", bm25_result, bm25_failures, bm25_n)

    # Hybrid evaluation (if not --bm25-only)
    hybrid_result = None
    if not bm25_only:
        voyage_key = os.environ.get("VOYAGE_API_KEY")
        if not voyage_key:
            print("\n  VOYAGE_API_KEY not set — skipping hybrid evaluation.")
        else:
            print(f"\nRunning hybrid evaluation...")
            hybrid_config = {**base_config, "retrieval_mode": "hybrid"}
            hybrid_result, hybrid_failures = evaluate_config(hybrid_config, "contractnli", data_dir)
            hybrid_n = sum(hybrid_failures.values())
            _print_results("Hybrid (BM25 + Qdrant RRF)", hybrid_result, hybrid_failures, hybrid_n)

    # Report
    print(f"\n{'=' * 60}")
    print("GATE RESULTS")
    print("=" * 60)

    print("\n  (BM25-only results are informational — gate applies to hybrid)")

    if hybrid_result is not None:
        _print_comparison(
            bm25_result, bm25_failures, bm25_n,
            hybrid_result, hybrid_failures, hybrid_n,
        )
        _gate_check("Hybrid", hybrid_result)
    elif not bm25_only:
        print("\n  Hybrid evaluation skipped (no VOYAGE_API_KEY).")
        print("  Set VOYAGE_API_KEY and optionally QDRANT_URL to run hybrid.")


def run_replication(
    data_dir: str = "data",
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    bm25_top_k: int = 32,
    dense_top_k: int = 32,
    fusion_top_n: int = 64,
) -> None:
    """Paper replication run using text-embedding-3-large (OpenAI).

    This uses a different embedding model (OpenAI, not Voyage) so it
    cannot go through evaluate_config which is locked to the production
    Voyage pipeline. Kept as a standalone path.
    """
    from campaigns.legalbench_rag.ground_truth_adapter import ContractNLIGroundTruth
    from campaigns.legalbench_rag.ingestion.contractnli_loader import ingest_contractnli
    from core.measurement.metrics import compute_all_k
    from core.measurement.taxonomy import classify

    data_path = Path(data_dir)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set. Required for --replicate.")
        sys.exit(1)

    print("=" * 60)
    print("REPLICATION RUN: text-embedding-3-large (ZeroEntropy paper)")
    print("=" * 60)

    print(f"\n[1/4] Ingesting ContractNLI (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    corpus_df, queries = ingest_contractnli(
        data_path, chunk_size, chunk_overlap, max_queries=194
    )

    print(f"\n[2/4] Loading ground truth...")
    gt = ContractNLIGroundTruth(
        benchmark_paths=data_path / "benchmarks" / "contractnli.json",
        corpus_dir=data_path / "corpus",
    )
    query_ids = gt.all_query_ids()
    print(f"  {len(query_ids)} queries loaded")

    print(f"\n[3/4] Building retrievers...")
    from core.retrieval.bm25_retriever import BM25Retriever
    bm25 = BM25Retriever(corpus_df, top_k=bm25_top_k)
    print(f"  BM25 ready ({len(corpus_df)} chunks indexed)")

    from core.retrieval.openai_retriever import OpenAIRetriever
    from core.retrieval.fusion import rrf
    from qdrant_client import QdrantClient

    qdrant_client = QdrantClient(":memory:")
    openai_retriever = OpenAIRetriever(
        client=qdrant_client,
        collection_name="contractnli_replication",
        corpus_df=corpus_df,
        top_k=dense_top_k,
    )
    print(f"  Indexing with text-embedding-3-large...")
    openai_retriever.index()
    print(f"  Qdrant (replication) ready")

    print(f"\n[4/4] Running hybrid evaluation on {len(query_ids)} queries...")

    all_p: dict[int, list[float]] = {k: [] for k in K_VALUES}
    all_r: dict[int, list[float]] = {k: [] for k in K_VALUES}
    failure_counts: Counter[FailureType] = Counter()

    for i, qid in enumerate(query_ids):
        query_text = gt.get_query_text(qid)
        gt_spans = gt.get_spans(qid)
        gt_doc_id = gt.get_doc_id(qid)

        sparse_result = bm25.retrieve(query_text, top_k=bm25_top_k)
        dense_result = openai_retriever.retrieve(query_text, top_k=dense_top_k)
        retrieval_result = rrf(sparse_result, dense_result, top_n=fusion_top_n)

        result = compute_all_k(retrieval_result.spans, gt_spans, k_values=K_VALUES)
        for k in K_VALUES:
            all_p[k].append(result.p_at_k[k])
            all_r[k].append(result.r_at_k[k])

        retrieved_with_docs = [
            (cid.split("#")[0], s[0], s[1])
            for cid, s in zip(retrieval_result.ids, retrieval_result.spans)
        ]
        gt_with_docs = [(gt_doc_id, s[0], s[1]) for s in gt_spans]
        failure = classify(retrieved_with_docs, gt_with_docs)
        failure_counts[failure] += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(query_ids):
            print(f"  [Replication] {i + 1}/{len(query_ids)} queries evaluated")

    n = len(query_ids)
    mean_p = {k: sum(all_p[k]) / n * 100 for k in K_VALUES}
    mean_r = {k: sum(all_r[k]) / n * 100 for k in K_VALUES}

    print(f"\n--- Replication (BM25 + text-embedding-3-large RRF) ---")
    print(f"{'k':>4}  {'P@k':>8}  {'R@k':>8}")
    print(f"{'---':>4}  {'---':>8}  {'---':>8}")
    for k in K_VALUES:
        print(f"{k:>4}  {mean_p[k]:>7.2f}%  {mean_r[k]:>7.2f}%")

    print(f"\n  Failure distribution:")
    for ft in FailureType:
        if ft in (FailureType.DTM, FailureType.XRF):
            continue
        count = failure_counts.get(ft, 0)
        pct = count / n * 100
        print(f"    {ft.name:>4}: {count:>4} ({pct:>5.1f}%)")

    # Paper comparison table
    paper_p1 = 6.41
    paper_r8 = 26.30
    actual_p1 = mean_p[1]
    actual_r8 = mean_r[8]
    delta_p1 = actual_p1 - paper_p1
    delta_r8 = actual_r8 - paper_r8

    print(f"\n{'=' * 60}")
    print("REPLICATION RUN (text-embedding-3-large vs paper targets)")
    print(f"{'=' * 60}")
    print(f"+--------+--------+--------------+--------+")
    print(f"| Metric | Actual | Paper target | Delta  |")
    print(f"+--------+--------+--------------+--------+")
    print(f"| P@1    | {actual_p1:5.2f}% | {paper_p1:5.2f}%       | {delta_p1:+5.2f} |")
    print(f"+--------+--------+--------------+--------+")
    print(f"| R@8    | {actual_r8:5.2f}% | {paper_r8:5.2f}%       | {delta_r8:+5.2f} |")
    print(f"+--------+--------+--------------+--------+")

    # Paper gate (not the permanent gate)
    paper_p1_ok = 4.4 <= actual_p1 <= 8.4
    paper_r8_ok = 23.3 <= actual_r8 <= 29.3
    print(f"\nPaper gate: P@1 [4.4, 8.4], R@8 [23.3, 29.3]")
    if paper_p1_ok and paper_r8_ok:
        print("  >>> PAPER GATE PASSED <<<")
    else:
        violations = []
        if not paper_p1_ok:
            violations.append(f"P@1={actual_p1:.2f}% outside [4.4, 8.4]")
        if not paper_r8_ok:
            violations.append(f"R@8={actual_r8:.2f}% outside [23.3, 29.3]")
        print(f"  >>> PAPER GATE FAILED: {'; '.join(violations)} <<<")

    # Reminder: production baseline is voyage-4-large
    print(f"\nPRODUCTION BASELINE (voyage-4-large — locked)")
    print(f"  P@1: 8.84%  R@8: 50.41%  DRM: 39.2%")
    print(f"  Permanent gate: P@1 [{P1_GATE[0]}, {P1_GATE[1]}], "
          f"R@8 [{R8_GATE[0]}, {R8_GATE[1]}]")


# Paper targets per dataset (text-embedding-3-large, RCTS, no reranker)
PAPER_TARGETS = {
    "contractnli": {"p1": 6.63, "r8": 24.99},
    "cuad":        {"p1": 1.97, "r8": 31.68},
    "maud":        {"p1": 2.65, "r8": 6.18},
    "privacy_qa":  {"p1": 14.38, "r8": 42.37},
}
DATASET_NAMES = ["contractnli", "cuad", "maud", "privacy_qa"]


def run_full_baseline(
    data_dir: str = "data",
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    bm25_top_k: int = 32,
    dense_top_k: int = 32,
    fusion_top_n: int = 64,
    persist: bool = False,
    eval_only: bool = False,
    retrieval_mode: str = "hybrid",
) -> None:
    """Full 776-query baseline across all four LegalBench-RAG datasets."""

    if eval_only:
        persist = True  # eval-only implies persist (needs existing collection)

    print("=" * 60)
    print("FULL BASELINE: All 4 LegalBench-RAG datasets")
    if eval_only:
        print("  (eval-only mode — using existing Qdrant index)")
    print("=" * 60)

    base_config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "bm25_top_k": bm25_top_k,
        "dense_top_k": dense_top_k,
        "fusion_top_n": fusion_top_n,
        "retrieval_mode": retrieval_mode,
    }

    mode = f"Hybrid (BM25 + voyage-4-large RRF)" if retrieval_mode == "hybrid" else "BM25-only"

    # Per-dataset evaluation
    per_dataset: dict[str, tuple] = {}
    for ds in DATASET_NAMES:
        benchmark_path = Path(data_dir) / "benchmarks" / f"{ds}.json"
        if not benchmark_path.exists():
            print(f"  Skipping {ds} — no benchmark file")
            continue
        print(f"\n  Evaluating {ds}...")
        metric_result, failure_counts = evaluate_config(base_config, ds, data_dir)
        n = sum(failure_counts.values())
        per_dataset[ds] = (metric_result, failure_counts, n)
        _print_results(f"{ds} ({mode})", metric_result, failure_counts, n)

    # Compute overall by weighted average
    total_queries = sum(t[2] for t in per_dataset.values())
    overall_p: dict[int, float] = {}
    overall_r: dict[int, float] = {}
    total_failures: Counter = Counter()
    for metric_result, failure_counts, n in per_dataset.values():
        for k in K_VALUES:
            overall_p[k] = overall_p.get(k, 0.0) + metric_result.p_at_k[k] * 100 * n
            overall_r[k] = overall_r.get(k, 0.0) + metric_result.r_at_k[k] * 100 * n
        total_failures += Counter(failure_counts)
    overall_p = {k: overall_p[k] / total_queries for k in K_VALUES}
    overall_r = {k: overall_r[k] / total_queries for k in K_VALUES}

    print(f"\n--- OVERALL ({mode}) ---")
    print(f"{'k':>4}  {'P@k':>8}  {'R@k':>8}")
    print(f"{'---':>4}  {'---':>8}  {'---':>8}")
    for k in K_VALUES:
        print(f"{k:>4}  {overall_p[k]:>7.2f}%  {overall_r[k]:>7.2f}%")

    print(f"\n  Failure distribution:")
    for ft in FailureType:
        if ft in (FailureType.DTM, FailureType.XRF):
            continue
        count = total_failures.get(ft.name, 0)
        pct = count / total_queries * 100
        print(f"    {ft.name:>4}: {count:>4} ({pct:>5.1f}%)")

    # Print comparison against paper targets
    print(f"\n{'=' * 70}")
    print(f"PER-DATASET COMPARISON vs PAPER TARGETS ({mode})")
    print("=" * 70)
    print(f"{'Dataset':<14} {'P@1':>7} {'Paper':>7} {'Delta':>7}   {'R@8':>7} {'Paper':>7} {'Delta':>7}   {'Gate':>6}")
    print(f"{'-'*14} {'-'*7} {'-'*7} {'-'*7}   {'-'*7} {'-'*7} {'-'*7}   {'-'*6}")

    all_pass = True
    for ds in DATASET_NAMES:
        if ds not in per_dataset:
            continue
        m_result, _, _ = per_dataset[ds]
        t = PAPER_TARGETS[ds]
        p1 = m_result.p_at_k[1] * 100
        r8 = m_result.r_at_k[8] * 100
        dp1 = p1 - t["p1"]
        dr8 = r8 - t["r8"]

        p1_ok = abs(dp1) <= 2.0
        r8_ok = abs(dr8) <= 5.0
        gate = "PASS" if (p1_ok and r8_ok) else "FAIL"
        if gate == "FAIL":
            all_pass = False

        print(
            f"{ds:<14} {p1:>6.2f}% {t['p1']:>6.2f}% {dp1:>+6.2f}   "
            f"{r8:>6.2f}% {t['r8']:>6.2f}% {dr8:>+6.2f}   {gate:>6}"
        )

    # Overall average
    avg_p1 = overall_p[1]
    avg_r8 = overall_r[8]
    print(f"\n{'Overall':<14} {avg_p1:>6.2f}%                 {avg_r8:>6.2f}%")

    if all_pass:
        print(f"\n  >>> ALL PER-DATASET GATES PASSED <<<")
    else:
        print(f"\n  >>> SOME PER-DATASET GATES FAILED — see above <<<")

    # Print suggested permanent gate values
    if retrieval_mode == "hybrid":
        print(f"\n  Suggested PERMANENT_GATE (voyage-4-large, 4 datasets):")
        print(f"    P@1: [{avg_p1 - 2.0:.1f}, {avg_p1 + 2.0:.1f}]  (actual: {avg_p1:.2f}%)")
        print(f"    R@8: [{avg_r8 - 3.0:.1f}, {avg_r8 + 3.0:.1f}]  (actual: {avg_r8:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 baseline evaluation")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    parser.add_argument("--bm25-top-k", type=int, default=32)
    parser.add_argument("--dense-top-k", type=int, default=32)
    parser.add_argument("--fusion-top-n", type=int, default=64)
    parser.add_argument("--bm25-only", action="store_true",
                        help="Skip dense retrieval, use BM25 only")
    parser.add_argument("--replicate", action="store_true",
                        help="Run paper replication with text-embedding-3-large")
    parser.add_argument("--full", action="store_true",
                        help="Run all 4 LegalBench-RAG datasets (776 queries)")
    parser.add_argument("--persist", action="store_true",
                        help="Write index to Qdrant Cloud (requires QDRANT_URL). "
                             "Skips re-embedding if collection already exists.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip embedding, use existing Qdrant index. Implies --persist.")
    args = parser.parse_args()

    if args.full:
        retrieval_mode = "bm25" if args.bm25_only else "hybrid"
        if retrieval_mode == "hybrid" and not os.environ.get("VOYAGE_API_KEY"):
            retrieval_mode = "bm25"
            print("  VOYAGE_API_KEY not set — using BM25-only mode.")
        run_full_baseline(
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            fusion_top_n=args.fusion_top_n,
            persist=args.persist,
            eval_only=args.eval_only,
            retrieval_mode=retrieval_mode,
        )
    elif args.replicate:
        run_replication(
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            fusion_top_n=args.fusion_top_n,
        )
    else:
        run_baseline(
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            fusion_top_n=args.fusion_top_n,
            bm25_only=args.bm25_only,
        )


if __name__ == "__main__":
    main()
