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

from campaigns.legalbench_rag.ground_truth_adapter import ContractNLIGroundTruth, LegalBenchGroundTruth
from campaigns.legalbench_rag.ingestion.contractnli_loader import ingest_contractnli
from campaigns.legalbench_rag.ingestion.full_loader import ingest_all
from core.measurement.metrics import compute_all_k, MetricResult
from core.measurement.taxonomy import classify, FailureType
from core.retrieval.bm25_retriever import BM25Retriever


K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def _evaluate(
    query_ids: list[str],
    gt: ContractNLIGroundTruth,
    retrieve_fn,
    label: str,
) -> dict:
    """Run evaluation loop, return metrics dict."""
    all_p: dict[int, list[float]] = {k: [] for k in K_VALUES}
    all_r: dict[int, list[float]] = {k: [] for k in K_VALUES}
    failure_counts: Counter[FailureType] = Counter()

    for i, qid in enumerate(query_ids):
        query_text = gt._queries[qid]["query"]  # noqa: SLF001
        gt_spans = gt.get_spans(qid)
        gt_doc_id = gt.get_doc_id(qid)

        retrieval_result = retrieve_fn(query_text)

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
            print(f"  [{label}] {i + 1}/{len(query_ids)} queries evaluated")

    mean_p = {k: sum(all_p[k]) / len(all_p[k]) * 100 for k in K_VALUES}
    mean_r = {k: sum(all_r[k]) / len(all_r[k]) * 100 for k in K_VALUES}

    return {
        "mean_p": mean_p,
        "mean_r": mean_r,
        "failure_counts": failure_counts,
        "n_queries": len(query_ids),
    }


def _print_results(label: str, metrics: dict) -> None:
    """Print metric table and failure distribution for one run."""
    print(f"\n--- {label} ---")
    print(f"{'k':>4}  {'P@k':>8}  {'R@k':>8}")
    print(f"{'---':>4}  {'---':>8}  {'---':>8}")
    for k in K_VALUES:
        print(f"{k:>4}  {metrics['mean_p'][k]:>7.2f}%  {metrics['mean_r'][k]:>7.2f}%")

    print(f"\n  Failure distribution:")
    for ft in FailureType:
        if ft in (FailureType.DTM, FailureType.XRF):
            continue
        count = metrics["failure_counts"].get(ft, 0)
        pct = count / metrics["n_queries"] * 100
        print(f"    {ft.name:>4}: {count:>4} ({pct:>5.1f}%)")


def _print_comparison(bm25_metrics: dict, hybrid_metrics: dict) -> None:
    """Print side-by-side comparison."""
    print(f"\n{'=' * 60}")
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print(f"{'k':>4}  {'BM25 P@k':>10}  {'Hybrid P@k':>12}  {'BM25 R@k':>10}  {'Hybrid R@k':>12}")
    print(f"{'---':>4}  {'---':>10}  {'---':>12}  {'---':>10}  {'---':>12}")
    for k in K_VALUES:
        bp = bm25_metrics["mean_p"][k]
        hp = hybrid_metrics["mean_p"][k]
        br = bm25_metrics["mean_r"][k]
        hr = hybrid_metrics["mean_r"][k]
        print(f"{k:>4}  {bp:>9.2f}%  {hp:>11.2f}%  {br:>9.2f}%  {hr:>11.2f}%")

    # Failure comparison
    print(f"\n  Failure distribution:")
    print(f"  {'Type':>4}  {'BM25':>12}  {'Hybrid':>12}")
    for ft in FailureType:
        if ft in (FailureType.DTM, FailureType.XRF):
            continue
        bc = bm25_metrics["failure_counts"].get(ft, 0)
        hc = hybrid_metrics["failure_counts"].get(ft, 0)
        bn = bm25_metrics["n_queries"]
        hn = hybrid_metrics["n_queries"]
        print(f"  {ft.name:>4}  {bc:>4} ({bc/bn*100:>5.1f}%)  {hc:>4} ({hc/hn*100:>5.1f}%)")


P1_GATE = (6.8, 10.8)    # 8.84 +/- 2.0
R8_GATE = (47.4, 53.4)   # 50.41 +/- 3.0


def _gate_check(label: str, metrics: dict) -> bool:
    """Print gate check and return whether it passed."""
    mean_p1 = metrics["mean_p"][1]
    mean_r8 = metrics["mean_r"][8]

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
    data_path = Path(data_dir)

    # 1. Ingest
    print("=" * 60)
    print("PHASE 2 BASELINE: ContractNLI")
    print("=" * 60)
    print(f"\n[1/5] Ingesting ContractNLI (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    corpus_df, queries = ingest_contractnli(
        data_path, chunk_size, chunk_overlap, max_queries=194
    )

    # 2. Ground truth
    print(f"\n[2/5] Loading ground truth...")
    gt = ContractNLIGroundTruth(
        benchmark_paths=data_path / "benchmarks" / "contractnli.json",
        corpus_dir=data_path / "corpus",
    )
    query_ids = gt.all_query_ids()
    print(f"  {len(query_ids)} queries loaded")

    # 3. Build retrievers
    print(f"\n[3/5] Building retrievers...")
    bm25 = BM25Retriever(corpus_df, top_k=bm25_top_k)
    print(f"  BM25 ready ({len(corpus_df)} chunks indexed)")

    # 4. BM25-only evaluation (always runs)
    print(f"\n[4/5] Running BM25-only evaluation on {len(query_ids)} queries...")
    bm25_metrics = _evaluate(
        query_ids, gt,
        lambda q: bm25.retrieve(q, top_k=fusion_top_n),
        label="BM25-only",
    )
    _print_results("BM25-only", bm25_metrics)

    # 5. Hybrid evaluation (if keys are set and not --bm25-only)
    hybrid_metrics = None
    if not bm25_only:
        voyage_key = os.environ.get("VOYAGE_API_KEY")
        if not voyage_key:
            print("\n  VOYAGE_API_KEY not set — skipping hybrid evaluation.")
        else:
            from core.retrieval.qdrant_retriever import QdrantRetriever
            from core.retrieval.fusion import rrf
            from qdrant_client import QdrantClient

            qdrant_url = os.environ.get("QDRANT_URL", "")
            qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")

            if qdrant_url:
                client_kwargs = {"url": qdrant_url}
                if qdrant_api_key:
                    client_kwargs["api_key"] = qdrant_api_key
                qdrant_client = QdrantClient(**client_kwargs)
            else:
                # No QDRANT_URL: use in-memory for local testing
                qdrant_client = QdrantClient(":memory:")

            qdrant = QdrantRetriever(
                client=qdrant_client,
                collection_name="contractnli_baseline",
                corpus_df=corpus_df,
                top_k=dense_top_k,
            )
            print(f"\n[4.5/5] Indexing into Qdrant...")
            qdrant.index()
            print(f"  Qdrant ready")

            print(f"\n[5/5] Running hybrid evaluation on {len(query_ids)} queries...")

            def hybrid_retrieve(q: str):
                sparse_result = bm25.retrieve(q, top_k=bm25_top_k)
                dense_result = qdrant.retrieve(q, top_k=dense_top_k)
                return rrf(sparse_result, dense_result, top_n=fusion_top_n)

            hybrid_metrics = _evaluate(
                query_ids, gt, hybrid_retrieve, label="Hybrid",
            )
            _print_results("Hybrid (BM25 + Qdrant RRF)", hybrid_metrics)

    # Report
    print(f"\n{'=' * 60}")
    print("GATE RESULTS")
    print("=" * 60)

    # BM25-only results (informational, not gated)
    print("\n  (BM25-only results are informational — gate applies to hybrid)")

    if hybrid_metrics is not None:
        # Side-by-side
        _print_comparison(bm25_metrics, hybrid_metrics)

        # Hybrid gate (locked to Experiment 0 baseline)
        _gate_check("Hybrid", hybrid_metrics)
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
    """Paper replication run using text-embedding-3-large (OpenAI)."""
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

    def hybrid_retrieve(q: str):
        sparse_result = bm25.retrieve(q, top_k=bm25_top_k)
        dense_result = openai_retriever.retrieve(q, top_k=dense_top_k)
        return rrf(sparse_result, dense_result, top_n=fusion_top_n)

    metrics = _evaluate(query_ids, gt, hybrid_retrieve, label="Replication")
    _print_results("Replication (BM25 + text-embedding-3-large RRF)", metrics)

    # Paper comparison table
    paper_p1 = 6.41
    paper_r8 = 26.30
    actual_p1 = metrics["mean_p"][1]
    actual_r8 = metrics["mean_r"][8]
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
) -> None:
    """Full 776-query baseline across all four LegalBench-RAG datasets."""
    data_path = Path(data_dir)

    if eval_only:
        persist = True  # eval-only implies persist (needs existing collection)

    print("=" * 60)
    print("FULL BASELINE: All 4 LegalBench-RAG datasets")
    if eval_only:
        print("  (eval-only mode — using existing Qdrant index)")
    print("=" * 60)

    # 1. Ingest (or load existing)
    corpus_parquet = data_path / "corpus.parquet"
    if eval_only and corpus_parquet.exists():
        import pandas as pd
        print(f"\n[1/4] Loading existing corpus.parquet...")
        corpus_df = pd.read_parquet(corpus_parquet)
        print(f"  Loaded {len(corpus_df)} chunks from {corpus_parquet}")
    else:
        print(f"\n[1/4] Ingesting all datasets (chunk_size={chunk_size}, overlap={chunk_overlap})...")
        corpus_df, all_queries = ingest_all(
            data_path, chunk_size, chunk_overlap, max_queries_per_dataset=194
        )

    # 2. Ground truth
    print(f"\n[2/4] Loading ground truth...", flush=True)
    benchmark_paths = [
        data_path / "benchmarks" / f"{ds}.json"
        for ds in DATASET_NAMES
        if (data_path / "benchmarks" / f"{ds}.json").exists()
    ]
    gt = LegalBenchGroundTruth(benchmark_paths, data_path / "corpus")
    query_ids = gt.all_query_ids()
    print(f"  {len(query_ids)} total queries loaded", flush=True)
    for ds in DATASET_NAMES:
        ds_ids = gt.query_ids_for_dataset(ds)
        print(f"    {ds}: {len(ds_ids)} queries", flush=True)

    # 3. Build retrievers
    print(f"\n[3/4] Building retrievers...")
    bm25 = BM25Retriever(corpus_df, top_k=bm25_top_k)
    print(f"  BM25 ready ({len(corpus_df)} chunks indexed)")

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    qdrant = None
    if not voyage_key:
        print("  VOYAGE_API_KEY not set — BM25-only mode.")
    else:
        from core.retrieval.qdrant_retriever import QdrantRetriever
        from qdrant_client import QdrantClient

        collection_name = "legalbench_rag_full"

        if persist:
            qdrant_url = os.environ.get("QDRANT_URL", "")
            qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
            if not qdrant_url:
                print("  ERROR: --persist requires QDRANT_URL in .env")
                sys.exit(1)
            client_kwargs = {"url": qdrant_url}
            if qdrant_api_key:
                client_kwargs["api_key"] = qdrant_api_key
            qdrant_client = QdrantClient(**client_kwargs)
            print(f"  Connected to Qdrant Cloud: {qdrant_url}")
        else:
            qdrant_client = QdrantClient(":memory:")

        qdrant = QdrantRetriever(
            client=qdrant_client,
            collection_name=collection_name,
            corpus_df=corpus_df,
            top_k=dense_top_k,
        )

        total_chunks = len(corpus_df)
        existing_points = qdrant.collection_point_count() if persist else 0

        if persist and existing_points == total_chunks:
            print(
                f"  Collection '{collection_name}' exists with {existing_points} points, "
                f"skipping embedding.", flush=True,
            )
        else:
            if persist and existing_points > 0:
                print(
                    f"  Found {existing_points}/{total_chunks} points, "
                    f"rebuilding.", flush=True,
                )
            print(f"  Indexing into Qdrant (this may take a few minutes)...", flush=True)
            qdrant.index()
        print(f"  Qdrant ready", flush=True)

    # 4. Evaluate per-dataset and overall
    print(f"\n[4/4] Running evaluation...")

    if qdrant is not None:
        from core.retrieval.fusion import rrf

        def _make_hybrid_fn(ds_name: str):
            def retrieve_fn(q: str):
                sparse = bm25.retrieve(q, top_k=bm25_top_k, dataset_name=ds_name)
                dense = qdrant.retrieve(q, top_k=dense_top_k, dataset_name=ds_name)
                return rrf(sparse, dense, top_n=fusion_top_n)
            return retrieve_fn

        mode = "Hybrid (BM25 + voyage-4-large RRF)"
    else:
        def _make_hybrid_fn(ds_name: str):
            def retrieve_fn(q: str):
                return bm25.retrieve(q, top_k=fusion_top_n, dataset_name=ds_name)
            return retrieve_fn

        mode = "BM25-only"

    # Per-dataset evaluation
    per_dataset: dict[str, dict] = {}
    for ds in DATASET_NAMES:
        ds_ids = gt.query_ids_for_dataset(ds)
        if not ds_ids:
            print(f"  Skipping {ds} — no queries")
            continue
        metrics = _evaluate(ds_ids, gt, _make_hybrid_fn(ds), label=ds)
        per_dataset[ds] = metrics
        _print_results(f"{ds} ({mode})", metrics)

    # Compute overall by averaging per-dataset results (avoids redundant API calls)
    overall_p: dict[int, float] = {}
    overall_r: dict[int, float] = {}
    total_failures: Counter[FailureType] = Counter()
    total_queries = 0
    for m in per_dataset.values():
        for k in K_VALUES:
            overall_p[k] = overall_p.get(k, 0.0) + m["mean_p"][k] * m["n_queries"]
            overall_r[k] = overall_r.get(k, 0.0) + m["mean_r"][k] * m["n_queries"]
        total_failures += m["failure_counts"]
        total_queries += m["n_queries"]
    overall = {
        "mean_p": {k: overall_p[k] / total_queries for k in K_VALUES},
        "mean_r": {k: overall_r[k] / total_queries for k in K_VALUES},
        "failure_counts": total_failures,
        "n_queries": total_queries,
    }
    _print_results(f"OVERALL ({mode})", overall)

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
        m = per_dataset[ds]
        t = PAPER_TARGETS[ds]
        p1 = m["mean_p"][1]
        r8 = m["mean_r"][8]
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
    avg_p1 = overall["mean_p"][1]
    avg_r8 = overall["mean_r"][8]
    print(f"\n{'Overall':<14} {avg_p1:>6.2f}%                 {avg_r8:>6.2f}%")

    if all_pass:
        print(f"\n  >>> ALL PER-DATASET GATES PASSED <<<")
    else:
        print(f"\n  >>> SOME PER-DATASET GATES FAILED — see above <<<")

    # Print suggested permanent gate values
    if qdrant is not None:
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
        run_full_baseline(
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            fusion_top_n=args.fusion_top_n,
            persist=args.persist,
            eval_only=args.eval_only,
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
