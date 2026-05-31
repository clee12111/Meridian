"""Full-scale multi-query evaluation on any corpus.

Replaces Phase 3's single rewrite with multi-query fan-out:
  1. Keep the raw query
  2. Generate 2 reformulations (legal-specific + conceptual)
  3. Retrieve for each (3 total retrieval passes)
  4. Fuse results (RRF across the 3 result sets, then CC fusion of dense+sparse)
  5. Continue pipeline from Phase 5 onward as normal

Usage:
    python scripts/run_multiquery_eval.py --corpus maud --chunk-alpha 0.2 \\
        --routing-topk 3 --routing-alpha 0.7 --workers 14 \\
        --output data/eval_maud_multiquery.jsonl
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

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s")

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


def generate_reformulations(query: str) -> list[str]:
    """Generate 2 alternative query reformulations."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com")
    prompt = (
        "You are helping a legal retrieval system. Given the original query, "
        "produce exactly 2 alternative reformulations that might retrieve "
        "different relevant passages from a merger agreement.\n\n"
        "Reformulation 1: Use specific legal terminology and section references.\n"
        "Reformulation 2: Use broader conceptual language about the legal concept.\n\n"
        "Return ONLY the two reformulations, one per line, no numbering or labels.\n\n"
        f"Original query: {query}"
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": "You are a legal query reformulator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = (resp.choices[0].message.content or "").strip()
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            return lines[:2]
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return []


def main():
    p = argparse.ArgumentParser(description="Multi-query eval on any corpus")
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
    os.environ["MERIDIAN_NO_REWRITE"] = "1"  # We handle query expansion ourselves
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
    compiled, _ = compile_graph(db_path="data/corpus_eval_pipeline.sqlite",
                                context=context)

    mod = importlib.import_module(cfg["gt_module"])
    gt_cls = getattr(mod, cfg["gt_class"])
    ground_truth = gt_cls(cfg["benchmark"], cfg["corpus_dir"])

    data = json.loads(cfg["benchmark"].read_text(encoding="utf-8"))
    questions = [{"query_id": t["query_id"], "query": t["query"]}
                 for t in data["tests"]]
    if args.limit:
        questions = questions[:args.limit]

    write_lock = threading.Lock()
    results = []

    from core.supervisor.nodes import synthesis as _synthesis

    def _retrieve_via_pipeline(variant_query: str) -> dict | None:
        """Run the compiled pipeline for retrieval only (rewrite OFF, no rerank).

        Uses compiled.invoke() for thread safety — each call gets isolated state.
        Returns the reranked_result (retrieval output) without using the synthesis.
        """
        state = {
            "raw_query": variant_query,
            "iteration": 1, "max_iterations": 1,
            "loop_complete": False, "trace_id": "local",
        }
        try:
            final = _run_with_retry(
                lambda: compiled.invoke(
                    state,
                    config={"configurable": {"thread_id": str(uuid.uuid4())}}
                )
            )
            reranked = final.get("reranked_result", {})
            return {
                "ids": reranked.get("ids", []),
                "contents": reranked.get("contents", []),
                "scores": reranked.get("scores", []),
                "spans": reranked.get("spans", []),
                "routed_docs": final.get("routed_docs", []),
            }
        except Exception:
            return None

    def run_one(q):
        qid, question = q["query_id"], q["query"]

        # Multi-query: generate reformulations
        reformulations = generate_reformulations(question)
        all_queries = [question] + reformulations

        # Retrieve via pipeline for ALL variants (synthesis happens inside
        # but its output is discarded — we only use reranked_result)
        all_retrievals = []
        for variant_query in all_queries:
            result = _retrieve_via_pipeline(variant_query)
            if result is not None:
                all_retrievals.append(result)

        if not all_retrievals:
            with write_lock:
                print(f"  [{qid}] ALL variants FAILED", file=sys.stderr)
            return

        # Fuse: RRF across the variant retrieval sets
        chunk_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for retrieval in all_retrievals:
            ids = retrieval["ids"]
            contents = retrieval["contents"]
            spans = retrieval["spans"]
            for rank, (cid, content) in enumerate(zip(ids, contents)):
                rrf_score = 1.0 / (rank + 60)
                chunk_scores[cid] = chunk_scores.get(cid, 0) + rrf_score
                if cid not in chunk_data:
                    span = spans[rank] if rank < len(spans) else (0, 0)
                    chunk_data[cid] = {"content": content, "span": span}

        # Sort by fused score, take top-8
        sorted_cids = sorted(chunk_scores, key=lambda c: chunk_scores[c],
                             reverse=True)[:8]

        fused_contents = [chunk_data[cid]["content"] for cid in sorted_cids]
        fused_spans = [chunk_data[cid]["span"] for cid in sorted_cids]

        # Synthesize ONCE on the fused context
        synth_state = {
            "raw_query": question,
            "context_chunks": fused_contents,
            "context_ids": sorted_cids,
            "rewritten_query": question,
        }
        synth_result = _synthesis(synth_state, context)

        # Collect routed_docs from the primary (raw query) retrieval
        routed_docs = all_retrievals[0].get("routed_docs", [])

        # Measurement: use the fused spans for taxonomy
        reranked = {
            "ids": sorted_cids,
            "spans": fused_spans,
            "contents": fused_contents,
        }
        spans_for_tax = [tuple(s) for s in fused_spans]
        gt_spans = ground_truth.get_spans(qid)
        gt_doc = ground_truth.get_doc_id(qid)
        metrics = compute_all_k(spans_for_tax, gt_spans)

        retrieved_with = [(cid.split("#")[0], s[0], s[1])
                          for cid, s in zip(sorted_cids, spans_for_tax)]
        gt_with = [(gt_doc, s[0], s[1]) for s in gt_spans]
        classification = classify_with_confidence(retrieved_with, gt_with)

        ctx_chunks = [
            {"chunk_id": cid, "doc_id": cid.split("#")[0],
             "content": chunk_data[cid]["content"]}
            for cid in sorted_cids
        ]

        record = {
            "query_id": qid,
            "query": question,
            "answer": synth_result.get("answer", ""),
            "claims": synth_result.get("claims", []),
            "context_chunks": ctx_chunks,
            "p_at_1": metrics.p_at_k.get(1, 0.0),
            "r_at_8": metrics.r_at_k.get(8, 0.0),
            "failure_type": classification.failure_type.name,
            "routing_hit": (gt_doc in routed_docs) if routed_docs else None,
            "n_queries": len(all_queries),
        }

        with write_lock:
            results.append(record)
            ft = record["failure_type"]
            print(f"  [{qid}] P@1={record['p_at_1']:.2f} "
                  f"R@8={record['r_at_8']:.2f} {ft}")

    print(f"Running {len(questions)} queries on {args.corpus} "
          f"(multi-query, alpha={args.chunk_alpha}, "
          f"routing={'ON' if args.routing_topk else 'OFF'})...")

    with ThreadPoolExecutor(max_workers=min(args.workers, 16)) as executor:
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
