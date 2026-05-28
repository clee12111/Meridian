"""
v2 Eval harness: runs ContractNLI benchmark through the Meridian pipeline.

Usage:
    python scripts/run_eval.py                   # full 194-query run
    python scripts/run_eval.py --limit 5         # first 5 queries
    python scripts/run_eval.py --single-shot     # max_iterations=1
    python scripts/run_eval.py --fresh           # wipe output, start over

Output: one JSON record per line in data/eval_results_v2.jsonl.
Each record contains P@k, R@k, failure_type, verification score,
iteration count, and latency.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

BENCHMARK_PATH   = Path("data/benchmarks/contractnli.json")
CORPUS_DIR       = Path("data/corpus")
DEFAULT_OUT_PATH = Path("data/eval_results_v2.jsonl")

# Locked baseline from v1 (7-run measured, 194 queries)
BASELINE_P_AT_1 = 0.0884
BASELINE_R_AT_8 = 0.5029


# ── Retry logic ──────────────────────────────────────────────────────────────

_RETRY_WAITS = [5, 10, 20]


def _run_with_retry(fn):
    """Call fn(), retrying up to 3 times on transient API errors.

    Retries on exceptions whose class name contains RateLimit,
    ServiceUnavailable, ServerError, Overloaded, or '529' in message.
    """
    for attempt, wait in enumerate(_RETRY_WAITS, start=1):
        try:
            return fn()
        except Exception as exc:
            cls = type(exc).__name__
            msg = str(exc)
            retryable = any(
                kw in cls for kw in
                ("RateLimit", "ServiceUnavailable", "ServerError", "Overloaded")
            ) or "529" in msg
            if not retryable:
                raise
            print(f"  transient error ({cls}) — retry {attempt}/3 in {wait}s")
            time.sleep(wait)
    # Final attempt — let any exception propagate
    return fn()


# ── Benchmark loader ─────────────────────────────────────────────────────────

def _load_benchmark(benchmark_path: Path) -> list[dict]:
    """Load contractnli.json, return list of {query_id, query} dicts."""
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    return [
        {"query_id": t["query_id"], "query": t["query"]}
        for t in data["tests"]
    ]


# ── Resume logic ─────────────────────────────────────────────────────────────

def _load_completed_ids(output: Path) -> set[str]:
    """Read already-completed query IDs from an existing output file."""
    if not output.exists():
        return set()
    completed: set[str] = set()
    with output.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(json.loads(line)["query_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


# ── Summary printer ──────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    """Print v2 metrics summary with baseline comparison."""
    from collections import Counter

    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    mean_p1 = sum(r["p_at_1"] for r in results) / n
    mean_r8 = sum(r["r_at_8"] for r in results) / n
    delta_p1 = (mean_p1 - BASELINE_P_AT_1) * 100
    delta_r8 = (mean_r8 - BASELINE_R_AT_8) * 100

    failure_counts = Counter(r["failure_type"] for r in results)
    avg_vscore = sum(r["verification_score"] for r in results) / n
    avg_iter = sum(r["iterations"] for r in results) / n

    print()
    print("=" * 50)
    print(f"EVAL RESULTS — ContractNLI ({n} queries)")
    print("=" * 50)
    print(f"P@1:  {mean_p1:.4f}  (baseline: {BASELINE_P_AT_1:.4f}, delta: {delta_p1:+.2f} pp)")
    print(f"R@8:  {mean_r8:.4f}  (baseline: {BASELINE_R_AT_8:.4f}, delta: {delta_r8:+.2f} pp)")
    print()
    print("Failure distribution:")
    for ft in ["OK", "DRM", "CBF", "SGP", "ICR", "OVR"]:
        count = failure_counts.get(ft, 0)
        pct = count / n * 100 if n else 0
        print(f"  {ft}:  {count:>3}  ({pct:.1f}%)")
    print()
    print(f"Avg verification score: {avg_vscore:.2f}")
    print(f"Avg iterations:         {avg_iter:.2f}")
    print("=" * 50)


# ── Main eval loop ───────────────────────────────────────────────────────────

def run(
    limit: int | None = None,
    output: Path = DEFAULT_OUT_PATH,
    fresh: bool = False,
    ids: set[str] | None = None,
    max_iterations: int = 3,
    workers: int = 8,
    collection: str = "contractnli_baseline",
) -> None:
    """Run the v2 pipeline on ContractNLI and score with Tier B measurement."""
    from core.evaluation.ground_truth_contractnli import ContractNLIGroundTruth
    from core.measurement.metrics import compute_all_k
    from core.measurement.taxonomy import classify_with_confidence
    from core.supervisor.context import PipelineContext
    from core.supervisor.graph import compile_graph
    from core.supervisor.phoenix_tracing import setup_phoenix
    from core.supervisor.tracing import start_trace, end_trace, flush

    # Phoenix: separate project per collection for side-by-side comparison
    setup_phoenix(project_name=f"meridian-{collection}")

    # ── Setup ─────────────────────────────────────────────────────────────
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    context = PipelineContext.build(
        corpus_path=Path("data/corpus_contractnli.parquet"),
        qdrant_url=qdrant_url,
        collection_name=collection,
        dataset_name="contractnli",
        top_k=50,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )

    compiled, saver = compile_graph(
        db_path="data/eval_pipeline.sqlite",
        context=context,
    )

    ground_truth = ContractNLIGroundTruth(
        benchmark_path=BENCHMARK_PATH,
        corpus_dir=CORPUS_DIR,
    )

    questions = _load_benchmark(BENCHMARK_PATH)

    # Apply filters
    if ids:
        questions = [q for q in questions if q["query_id"] in ids]
    if limit:
        questions = questions[:limit]

    if not questions:
        print("No questions matched the filter.", file=sys.stderr)
        sys.exit(1)

    # Resume logic
    output.parent.mkdir(parents=True, exist_ok=True)
    if fresh and output.exists():
        output.unlink()

    completed_ids = _load_completed_ids(output)
    pending = [q for q in questions if q["query_id"] not in completed_ids]

    if not pending:
        print("All questions already completed. Nothing to do.")
        return

    total = len(pending)
    effective_workers = min(workers, 12)
    print(f"Running {total} queries (max_iterations={max_iterations}, workers={effective_workers})")
    if completed_ids:
        print(f"  Resuming — {len(completed_ids)} already completed, skipping")

    # ── Thread-safe helpers ───────────────────────────────────────────────
    write_lock = threading.Lock()
    results: list[dict] = []

    def run_one_query(q: dict, out_f) -> dict | None:
        """Run one query through the full pipeline. Returns record or None."""
        query_id = q["query_id"]
        question = q["query"]
        t0 = time.perf_counter()

        trace_uuid = uuid.uuid4().hex

        trace = start_trace(
            context,
            name=f"eval_{query_id}",
            input={"query": question, "query_id": query_id},
        )

        initial_state = {
            "raw_query": question,
            "iteration": 1,
            "max_iterations": max_iterations,
            "loop_complete": False,
            "trace_id": trace.trace_id if trace else trace_uuid,
        }

        thread_id = str(uuid.uuid4())

        try:
            final_state = _run_with_retry(
                lambda: compiled.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": thread_id}},
                )
            )
        except Exception as exc:
            logger.error("Query %s failed: %s", query_id, exc)
            with write_lock:
                print(f"  [{query_id}] FAILED: {exc}", file=sys.stderr)
            end_trace(trace, {"error": str(exc)})
            return None

        latency_ms = round((time.perf_counter() - t0) * 1000)

        # ── Measurement scoring ───────────────────────────────────────
        reranked = final_state.get("reranked_result", {})
        retrieved_spans = [tuple(s) for s in reranked.get("spans", [])]
        retrieved_ids = reranked.get("ids", [])

        gt_spans = ground_truth.get_spans(query_id)
        gt_doc_id = ground_truth.get_doc_id(query_id)

        metric_result = compute_all_k(retrieved_spans, gt_spans)

        retrieved_with_docs = [
            (cid.split("#")[0], s[0], s[1])
            for cid, s in zip(retrieved_ids, retrieved_spans)
        ]
        gt_with_docs = [(gt_doc_id, s[0], s[1]) for s in gt_spans]
        classification = classify_with_confidence(
            retrieved_with_docs, gt_with_docs
        )

        end_trace(trace, {
            "answer": final_state.get("answer", "")[:200],
            "p_at_1": metric_result.p_at_k.get(1, 0.0),
            "r_at_8": metric_result.r_at_k.get(8, 0.0),
            "failure_type": classification.failure_type.name,
            "verification_score": final_state.get(
                "verification_result", {}
            ).get("score", 0.0),
            "iterations": final_state.get("iteration", 1),
        })

        # ── Build record ──────────────────────────────────────────────
        record = {
            "query_id": query_id,
            "query": question,
            "answer": final_state.get("answer", ""),
            "claims": final_state.get("claims", []),
            "chunks": [
                {
                    "chunk_id": cid,
                    "score": round(float(score), 4),
                    "text": text,
                }
                for cid, score, text in zip(
                    reranked.get("ids", []),
                    reranked.get("scores", []),
                    reranked.get("contents", []),
                )
            ],
            "p_at_1": metric_result.p_at_k.get(1, 0.0),
            "p_at_4": metric_result.p_at_k.get(4, 0.0),
            "r_at_8": metric_result.r_at_k.get(8, 0.0),
            "failure_type": classification.failure_type.name,
            "confidence": classification.confidence,
            "ambiguous": classification.ambiguous,
            "near_category": classification.near_category,
            "boundary_distance": classification.boundary_distance,
            "taxonomy_evidence": classification.evidence,
            "verification_score": final_state.get(
                "verification_result", {}
            ).get("score", 0.0),
            "verification_details": final_state.get(
                "verification_result", {}
            ).get("details", []),
            "iterations": final_state.get("iteration", 1),
            "max_iterations": max_iterations,
            "latency_ms": latency_ms,
            "generation_model": "deepseek-v4-flash",
            "reranker_model": "rerank-2.5",
            "trace_id": initial_state["trace_id"],
        }

        p1 = record["p_at_1"]
        r8 = record["r_at_8"]
        ft = record["failure_type"]
        conf = record["confidence"]
        itr = record["iterations"]
        vs = record["verification_score"]

        with write_lock:
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            results.append(record)
            print(
                f"  [{query_id}] "
                f"P@1={p1:.2f} R@8={r8:.2f} fail={ft}({conf}) "
                f"iter={itr} vscore={vs:.2f} ({latency_ms}ms)"
            )

        return record

    # ── Run with thread pool ─────────────────────────────────────────────
    with output.open("a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(run_one_query, q, out_f): q
                for q in pending
            }
            for future in as_completed(futures):
                future.result()  # surfaces exceptions if any escaped

    _print_summary(results)
    total_in_file = len(completed_ids) + len(results)
    print(f"\nResults written to {output} ({len(results)} new, {total_in_file} total)")

    # Flush all pending Langfuse events before exit
    flush(context)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meridian v2 eval harness")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N questions")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT_PATH,
                   help="Override output path")
    p.add_argument("--fresh", action="store_true", default=False,
                   help="Wipe output file and start fresh")
    p.add_argument("--ids", default=None,
                   help="Comma-separated query IDs (e.g. contractnli-0762,contractnli-0769)")
    p.add_argument("--max-iterations", type=int, default=3,
                   help="Max Phase 10 loop iterations (default: 3)")
    p.add_argument("--single-shot", action="store_true", default=False,
                   help="Set max_iterations=1 (no Phase 10 looping)")
    p.add_argument("--no-rewrite", action="store_true", default=False,
                   help="Disable Phase 3 query rewriting (A/B test)")
    p.add_argument("--no-rerank", action="store_true", default=False,
                   help="Disable Phase 6 reranking (A/B test)")
    p.add_argument("--collection", type=str,
                   default="contractnli_baseline",
                   help="Qdrant collection name to use "
                        "(default: contractnli_baseline)")
    p.add_argument("--top-k", type=int, default=None,
                   help="Override retriever top_k (default: context default)")
    p.add_argument("--fusion-top-n", type=int, default=None,
                   help="Override RRF fusion top_n (default: 50)")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel workers (default: 8, max: 12 for API rate limit safety)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    max_iter = 1 if args.single_shot else args.max_iterations
    ids = set(args.ids.split(",")) if args.ids else None

    # A/B: set env flag and override output path when --no-rewrite
    if args.no_rewrite:
        os.environ["MERIDIAN_NO_REWRITE"] = "1"
        if args.output == DEFAULT_OUT_PATH:
            args.output = Path("data/eval_results_v2_norewrite.jsonl")

    # A/B: set env flag and override output path when --no-rerank
    if args.no_rerank:
        os.environ["MERIDIAN_NO_RERANK"] = "1"
        if args.output == DEFAULT_OUT_PATH:
            args.output = Path("data/eval_results_v2_norerank.jsonl")

    if args.top_k is not None:
        os.environ["MERIDIAN_TOP_K"] = str(args.top_k)
    if args.fusion_top_n is not None:
        os.environ["MERIDIAN_FUSION_TOP_N"] = str(args.fusion_top_n)

    run(
        limit=args.limit,
        output=args.output,
        fresh=args.fresh,
        ids=ids,
        max_iterations=max_iter,
        workers=min(args.workers, 12),
        collection=args.collection,
    )


if __name__ == "__main__":
    main()
