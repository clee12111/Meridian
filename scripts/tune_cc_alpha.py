"""CC fusion alpha tuning sweep across ContractNLI and PrivacyQA.

Runs a grid of alpha values on 50 queries per corpus and reports
P@1, R@8, DRM for each. Includes RRF as baseline row.

Usage:
    python scripts/tune_cc_alpha.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)

ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CORPORA = [
    {
        "name": "ContractNLI",
        "benchmark": Path("data/benchmarks/contractnli.json"),
        "corpus_parquet": Path("data/corpus_contractnli.parquet"),
        "corpus_dir": Path("data/corpus"),
        "collection": "contractnli_sac",
        "dataset_name": "contractnli",
        "gt_class": "ContractNLIGroundTruth",
        "gt_module": "core.evaluation.ground_truth_contractnli",
        "n_queries": 50,
    },
    {
        "name": "PrivacyQA",
        "benchmark": Path("data/benchmarks/privacy_qa.json"),
        "corpus_parquet": Path("data/corpus_privacy_qa.parquet"),
        "corpus_dir": Path("data/corpus"),
        "collection": "privacyqa_baseline",
        "dataset_name": "privacy_qa",
        "gt_class": "PrivacyQAGroundTruth",
        "gt_module": "core.evaluation.ground_truth_privacyqa",
        "n_queries": 50,
    },
]

_RETRY_WAITS = [5, 10, 20]


def _run_with_retry(fn):
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
            time.sleep(wait)
    return fn()


def run_sweep_for_corpus(corpus_cfg: dict, alpha: float | None) -> dict:
    """Run N queries on a corpus with given alpha. Returns metrics dict.

    alpha=None means use RRF (no CC).
    """
    import importlib

    from core.measurement.metrics import compute_all_k
    from core.measurement.taxonomy import classify_with_confidence
    from core.supervisor.context import PipelineContext
    from core.supervisor.graph import compile_graph

    # Set env flags
    os.environ["MERIDIAN_NO_RERANK"] = "1"
    os.environ["MERIDIAN_NO_REWRITE"] = "1"
    if alpha is not None:
        os.environ["MERIDIAN_CC_ALPHA"] = str(alpha)
    elif "MERIDIAN_CC_ALPHA" in os.environ:
        del os.environ["MERIDIAN_CC_ALPHA"]

    # Build context
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    context = PipelineContext.build(
        corpus_path=corpus_cfg["corpus_parquet"],
        qdrant_url=qdrant_url,
        collection_name=corpus_cfg["collection"],
        dataset_name=corpus_cfg["dataset_name"],
        top_k=50,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )

    compiled, saver = compile_graph(
        db_path="data/tune_pipeline.sqlite",
        context=context,
    )

    # Load ground truth
    mod = importlib.import_module(corpus_cfg["gt_module"])
    gt_class = getattr(mod, corpus_cfg["gt_class"])
    ground_truth = gt_class(
        benchmark_path=corpus_cfg["benchmark"],
        corpus_dir=corpus_cfg["corpus_dir"],
    )

    # Load queries
    data = json.loads(corpus_cfg["benchmark"].read_text(encoding="utf-8"))
    questions = [
        {"query_id": t["query_id"], "query": t["query"]}
        for t in data["tests"]
    ][:corpus_cfg["n_queries"]]

    # Run queries in parallel
    write_lock = threading.Lock()
    results: list[dict] = []

    def run_one(q: dict) -> dict | None:
        query_id = q["query_id"]
        question = q["query"]
        thread_id = str(uuid.uuid4())

        initial_state = {
            "raw_query": question,
            "iteration": 1,
            "max_iterations": 1,
            "loop_complete": False,
            "trace_id": "local",
        }

        try:
            final_state = _run_with_retry(
                lambda: compiled.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": thread_id}},
                )
            )
        except Exception as exc:
            with write_lock:
                print(f"  [{query_id}] FAILED: {exc}", file=sys.stderr)
            return None

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

        record = {
            "query_id": query_id,
            "p_at_1": metric_result.p_at_k.get(1, 0.0),
            "r_at_8": metric_result.r_at_k.get(8, 0.0),
            "failure_type": classification.failure_type.name,
        }

        with write_lock:
            results.append(record)

        return record

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_one, q): q for q in questions}
        for future in as_completed(futures):
            future.result()

    # Aggregate
    n = len(results)
    if n == 0:
        return {"p1": 0, "r8": 0, "drm": 0, "ok": 0, "n": 0}

    mean_p1 = sum(r["p_at_1"] for r in results) / n
    mean_r8 = sum(r["r_at_8"] for r in results) / n
    ft_counts = Counter(r["failure_type"] for r in results)
    drm_pct = ft_counts.get("DRM", 0) / n * 100
    ok_pct = ft_counts.get("OK", 0) / n * 100

    return {
        "p1": mean_p1,
        "r8": mean_r8,
        "drm": drm_pct,
        "ok": ok_pct,
        "n": n,
    }


def main() -> None:
    print("=" * 100)
    print("CC FUSION alpha TUNING SWEEP")
    print("=" * 100)

    # Results: {corpus_name: {alpha_label: metrics_dict}}
    all_results: dict[str, dict[str, dict]] = {}

    for corpus_cfg in CORPORA:
        corpus_name = corpus_cfg["name"]
        all_results[corpus_name] = {}

        # RRF baseline
        label = "RRF"
        print(f"\n  [{corpus_name}] Running {label} (50 queries)...", flush=True)
        metrics = run_sweep_for_corpus(corpus_cfg, alpha=None)
        all_results[corpus_name][label] = metrics
        print(f"    P@1={metrics['p1']:.4f}  R@8={metrics['r8']:.4f}  "
              f"DRM={metrics['drm']:.1f}%  OK={metrics['ok']:.1f}%")

        # CC grid
        for alpha in ALPHA_GRID:
            label = f"{alpha:.1f}"
            print(f"  [{corpus_name}] Running alpha={label} (50 queries)...", flush=True)
            metrics = run_sweep_for_corpus(corpus_cfg, alpha=alpha)
            all_results[corpus_name][label] = metrics
            print(f"    P@1={metrics['p1']:.4f}  R@8={metrics['r8']:.4f}  "
                  f"DRM={metrics['drm']:.1f}%  OK={metrics['ok']:.1f}%")

    # Print results table
    print()
    print("=" * 100)
    print("RESULTS TABLE")
    print("=" * 100)

    header = (f"{'alpha':>5} | "
              f"{'CNL P@1':>8} {'CNL R@8':>8} {'CNL DRM':>8} | "
              f"{'PQA P@1':>8} {'PQA R@8':>8} {'PQA DRM':>8} | "
              f"{'Avg R@8':>8}")
    print(header)
    print("-" * len(header))

    best_cnl = {"label": "", "r8": -1}
    best_pqa = {"label": "", "r8": -1}
    best_combined = {"label": "", "avg_r8": -1}

    labels = ["RRF"] + [f"{a:.1f}" for a in ALPHA_GRID]
    for label in labels:
        cnl = all_results["ContractNLI"].get(label, {})
        pqa = all_results["PrivacyQA"].get(label, {})
        avg_r8 = (cnl.get("r8", 0) + pqa.get("r8", 0)) / 2

        print(f"{label:>5} | "
              f"{cnl.get('p1', 0):>8.4f} {cnl.get('r8', 0):>8.4f} "
              f"{cnl.get('drm', 0):>7.1f}% | "
              f"{pqa.get('p1', 0):>8.4f} {pqa.get('r8', 0):>8.4f} "
              f"{pqa.get('drm', 0):>7.1f}% | "
              f"{avg_r8:>8.4f}")

        if cnl.get("r8", 0) > best_cnl["r8"]:
            best_cnl = {"label": label, "r8": cnl["r8"],
                        "p1": cnl["p1"], "drm": cnl["drm"]}
        if pqa.get("r8", 0) > best_pqa["r8"]:
            best_pqa = {"label": label, "r8": pqa["r8"],
                        "p1": pqa["p1"], "drm": pqa["drm"]}
        if avg_r8 > best_combined["avg_r8"]:
            best_combined = {"label": label, "avg_r8": avg_r8}

    print()
    print(f"ContractNLI best alpha: {best_cnl['label']} "
          f"(R@8={best_cnl['r8']:.4f}, DRM={best_cnl['drm']:.1f}%)")
    print(f"PrivacyQA best alpha:   {best_pqa['label']} "
          f"(R@8={best_pqa['r8']:.4f}, DRM={best_pqa['drm']:.1f}%)")
    print(f"Combined best alpha:    {best_combined['label']} "
          f"(avg R@8={best_combined['avg_r8']:.4f})")

    # Compare per-dataset winners
    cnl_val = float(best_cnl["label"]) if best_cnl["label"] != "RRF" else -1
    pqa_val = float(best_pqa["label"]) if best_pqa["label"] != "RRF" else -1

    if best_cnl["label"] == "RRF" or best_pqa["label"] == "RRF":
        print("\nOne corpus prefers RRF over CC — per-dataset routing warranted")
    elif abs(cnl_val - pqa_val) <= 0.1:
        avg = (cnl_val + pqa_val) / 2
        print(f"\nSingle alpha sufficient — {avg:.1f} works for both corpora")
    else:
        print(f"\nPer-dataset alpha warranted — corpora prefer different "
              f"BM25/dense balance")


if __name__ == "__main__":
    main()
