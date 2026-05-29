"""Run eval on any corpus and save per-query JSONL with answers+claims.

Unlike the harness (ContractNLI-only), this handles all four corpora.

Usage:
    python scripts/run_corpus_eval.py --corpus cuad --chunk-alpha 0.1 --output data/eval_cuad.jsonl
    python scripts/run_corpus_eval.py --corpus maud --chunk-alpha 0.2 --routing-topk 3 --routing-alpha 0.7
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)

CORPUS_CONFIG = {
    "contractnli": {
        "parquet": Path("data/corpus_contractnli.parquet"),
        "benchmark": Path("data/benchmarks/contractnli.json"),
        "collection": "contractnli_sac_v4",
        "dataset_name": "contractnli",
        "gt_module": "core.evaluation.ground_truth_contractnli",
        "gt_class": "ContractNLIGroundTruth",
        "routing_index": "data/routing_index_v4.npz",
        "corpus_dir": Path("data/corpus"),
    },
    "privacyqa": {
        "parquet": Path("data/corpus_privacy_qa.parquet"),
        "benchmark": Path("data/benchmarks/privacy_qa.json"),
        "collection": "privacyqa_sac_v4",
        "dataset_name": "privacy_qa",
        "gt_module": "core.evaluation.ground_truth_privacyqa",
        "gt_class": "PrivacyQAGroundTruth",
        "routing_index": "data/routing_index_privacyqa_v4.npz",
        "corpus_dir": Path("data/corpus"),
    },
    "cuad": {
        "parquet": Path("data/corpus_cuad.parquet"),
        "benchmark": Path("data/benchmarks/cuad.json"),
        "collection": "cuad_sac_v4",
        "dataset_name": "cuad",
        "gt_module": "core.evaluation.ground_truth_cuad",
        "gt_class": "CUADGroundTruth",
        "routing_index": "data/routing_index_cuad_v4.npz",
        "corpus_dir": Path("data/corpus"),
    },
    "maud": {
        "parquet": Path("data/corpus_maud.parquet"),
        "benchmark": Path("data/benchmarks/maud.json"),
        "collection": "maud_sac_v4",
        "dataset_name": "maud",
        "gt_module": "core.evaluation.ground_truth_maud",
        "gt_class": "MAUDGroundTruth",
        "routing_index": "data/routing_index_maud_v4.npz",
        "corpus_dir": Path("data/corpus"),
    },
}

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


def main():
    p = argparse.ArgumentParser(description="Run eval on any corpus")
    p.add_argument("--corpus", required=True, choices=list(CORPUS_CONFIG.keys()))
    p.add_argument("--chunk-alpha", type=float, required=True)
    p.add_argument("--routing-topk", type=int, default=None)
    p.add_argument("--routing-alpha", type=float, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cfg = CORPUS_CONFIG[args.corpus]

    os.environ["MERIDIAN_EMBED_MODEL"] = "voyage-4"
    os.environ["MERIDIAN_NO_RERANK"] = "1"
    os.environ["MERIDIAN_CC_ALPHA"] = str(args.chunk_alpha)

    if args.routing_topk:
        os.environ["MERIDIAN_ROUTING_TOPK"] = str(args.routing_topk)
        os.environ["MERIDIAN_ROUTING_INDEX"] = cfg["routing_index"]
        if args.routing_alpha:
            os.environ["MERIDIAN_ROUTING_ALPHA"] = str(args.routing_alpha)
    elif "MERIDIAN_ROUTING_TOPK" in os.environ:
        del os.environ["MERIDIAN_ROUTING_TOPK"]

    import core.retrieval.routing as routing_mod
    routing_mod._router = None

    from core.measurement.metrics import compute_all_k
    from core.measurement.taxonomy import classify_with_confidence
    from core.supervisor.context import PipelineContext
    from core.supervisor.graph import compile_graph

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    context = PipelineContext.build(
        corpus_path=cfg["parquet"],
        qdrant_url=qdrant_url,
        collection_name=cfg["collection"],
        dataset_name=cfg["dataset_name"],
        top_k=50,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    compiled, _ = compile_graph(db_path="data/corpus_eval_pipeline.sqlite", context=context)

    mod = importlib.import_module(cfg["gt_module"])
    gt_cls = getattr(mod, cfg["gt_class"])
    ground_truth = gt_cls(cfg["benchmark"], cfg["corpus_dir"])

    data = json.loads(cfg["benchmark"].read_text(encoding="utf-8"))
    questions = [{"query_id": t["query_id"], "query": t["query"]} for t in data["tests"]]
    if args.limit:
        questions = questions[:args.limit]

    write_lock = threading.Lock()
    results = []

    def run_one(q):
        qid, question = q["query_id"], q["query"]
        state = {
            "raw_query": question, "iteration": 1, "max_iterations": 1,
            "loop_complete": False, "trace_id": "local",
        }
        try:
            final = _run_with_retry(
                lambda: compiled.invoke(state, config={"configurable": {"thread_id": str(uuid.uuid4())}})
            )
        except Exception as exc:
            with write_lock:
                print(f"  [{qid}] FAILED: {exc}", file=sys.stderr)
            return

        reranked = final.get("reranked_result", {})
        spans = [tuple(s) for s in reranked.get("spans", [])]
        ids = reranked.get("ids", [])
        gt_spans = ground_truth.get_spans(qid)
        gt_doc = ground_truth.get_doc_id(qid)
        metrics = compute_all_k(spans, gt_spans)

        retrieved_with = [(cid.split("#")[0], s[0], s[1]) for cid, s in zip(ids, spans)]
        gt_with = [(gt_doc, s[0], s[1]) for s in gt_spans]
        classification = classify_with_confidence(retrieved_with, gt_with)

        record = {
            "query_id": qid,
            "query": question,
            "answer": final.get("answer", ""),
            "claims": final.get("claims", []),
            "p_at_1": metrics.p_at_k.get(1, 0.0),
            "r_at_8": metrics.r_at_k.get(8, 0.0),
            "failure_type": classification.failure_type.name,
            "routing_hit": (gt_doc in final.get("routed_docs", [])) if final.get("routed_docs") else None,
        }

        with write_lock:
            results.append(record)
            ft = record["failure_type"]
            print(f"  [{qid}] P@1={record['p_at_1']:.2f} R@8={record['r_at_8']:.2f} {ft}")

    print(f"Running {len(questions)} queries on {args.corpus} "
          f"(alpha={args.chunk_alpha}, routing={'ON' if args.routing_topk else 'OFF'})...")

    with ThreadPoolExecutor(max_workers=min(args.workers, 12)) as executor:
        futures = {executor.submit(run_one, q): q for q in questions}
        for f in as_completed(futures):
            f.result()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda x: x["query_id"]):
            f.write(json.dumps(r) + "\n")

    n = len(results)
    from collections import Counter
    ft = Counter(r["failure_type"] for r in results)
    print(f"\n{n} results -> {args.output}")
    print(f"P@1={sum(r['p_at_1'] for r in results)/n:.4f} "
          f"R@8={sum(r['r_at_8'] for r in results)/n:.4f}")
    print(f"Failure: {dict(ft)}")


if __name__ == "__main__":
    main()
