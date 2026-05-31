"""Diagnose access-miss residual: is GT evidence ranked 9-30 (recoverable)
or not retrieved at all (retrieval-upstream)?

For each INCORRECT+FAITHFUL CBF case, re-retrieves at top-50 within routed
docs and computes R@k at k=1,4,8,16,30,50. Cases with R@8=0 but R@30>0
are type-(a) recoverable by reorder. R@50=0 is type-(b) retrieval-upstream.

Usage:
    python scripts/diagnose_access_miss.py
    python scripts/diagnose_access_miss.py --corpus maud
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

CORPUS_PARAMS = {
    "maud": {
        "parquet": Path("data/corpus_maud.parquet"),
        "benchmark": Path("data/benchmarks/maud.json"),
        "collection": "maud_sac_v4",
        "dataset_name": "maud",
        "gt_module": "core.evaluation.ground_truth_maud",
        "gt_class": "MAUDGroundTruth",
        "routing_index": "data/routing_index_maud_v4.npz",
        "corpus_dir": Path("data/corpus"),
        "cc_alpha": 0.2,
        "routing_alpha": 0.7,
        "routing_topk": 3,
    },
    "contractnli": {
        "parquet": Path("data/corpus_contractnli.parquet"),
        "benchmark": Path("data/benchmarks/contractnli.json"),
        "collection": "contractnli_sac_v4",
        "dataset_name": "contractnli",
        "gt_module": "core.evaluation.ground_truth_contractnli",
        "gt_class": "ContractNLIGroundTruth",
        "routing_index": "data/routing_index_v4.npz",
        "corpus_dir": Path("data/corpus"),
        "cc_alpha": 0.2,
        "routing_alpha": 0.3,
        "routing_topk": 3,
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
        "cc_alpha": 0.1,
        "routing_alpha": 0.3,
        "routing_topk": 3,
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
        "cc_alpha": 0.1,
        "routing_alpha": 0.5,
        "routing_topk": 3,
    },
}

FAITH_THRESHOLD = 0.8
EVAL_FILES = {
    "maud": "data/eval_maud_baseline_ctx.jsonl",
    "contractnli": "data/eval_contractnli_baseline_ctx.jsonl",
    "cuad": "data/eval_cuad_baseline_ctx.jsonl",
    "privacyqa": "data/eval_privacyqa_baseline_ctx.jsonl",
}
JUDGE_FILES = {
    "maud": "data/judge_v2_maud_eval_maud_baseline_ctx.jsonl",
    "contractnli": "data/judge_v2_contractnli_eval_contractnli_baseline_ctx.jsonl",
    "cuad": "data/judge_v2_cuad_eval_cuad_baseline_ctx.jsonl",
    "privacyqa": "data/judge_v2_privacyqa_eval_privacyqa_baseline_ctx.jsonl",
}
FAITH_FILES = {
    "maud": "data/faith_maud_eval_maud_baseline_ctx.jsonl",
    "contractnli": "data/faith_contractnli_eval_contractnli_baseline_ctx.jsonl",
    "cuad": "data/faith_cuad_eval_cuad_baseline_ctx.jsonl",
    "privacyqa": "data/faith_privacyqa_eval_privacyqa_baseline_ctx.jsonl",
}


def load_jsonl(path):
    results = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                results[r["query_id"]] = r
    return results


def get_access_miss_population(corpus):
    """Get INCORRECT+FAITHFUL queries, filtered to CBF (within-doc access miss)."""
    judges = load_jsonl(JUDGE_FILES[corpus])
    faiths = load_jsonl(FAITH_FILES[corpus])
    evals = load_jsonl(EVAL_FILES[corpus])

    population = []
    for qid, j in judges.items():
        if j.get("verdict") != "INCORRECT":
            continue
        if faiths.get(qid, {}).get("faithfulness_score", 0) < FAITH_THRESHOLD:
            continue
        ft = j.get("failure_type", evals.get(qid, {}).get("failure_type", "?"))
        population.append({"query_id": qid, "failure_type": ft,
                          "query": evals.get(qid, {}).get("query", "")})
    return sorted(population, key=lambda r: r["query_id"])


def run_diagnostic(corpus, population, params):
    """Re-retrieve at top-50 and compute R@k for each query."""
    os.environ["MERIDIAN_EMBED_MODEL"] = "voyage-4"
    os.environ["MERIDIAN_NO_REWRITE"] = "1"
    os.environ["MERIDIAN_NO_RERANK"] = "1"
    os.environ["MERIDIAN_CC_ALPHA"] = str(params["cc_alpha"])
    os.environ["MERIDIAN_ROUTING_TOPK"] = str(params["routing_topk"])
    os.environ["MERIDIAN_ROUTING_INDEX"] = params["routing_index"]
    os.environ["MERIDIAN_ROUTING_ALPHA"] = str(params["routing_alpha"])

    import core.retrieval.routing as routing_mod
    routing_mod._router = None

    from core.measurement.metrics import compute_all_k
    from core.retrieval.fusion import cc_fusion
    from core.retrieval.routing import route_query
    from core.supervisor.context import PipelineContext

    context = PipelineContext.build(
        corpus_path=params["parquet"],
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection_name=params["collection"],
        dataset_name=params["dataset_name"],
        top_k=50,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )

    mod = importlib.import_module(params["gt_module"])
    gt_cls = getattr(mod, params["gt_class"])
    ground_truth = gt_cls(params["benchmark"], params["corpus_dir"])

    results = []
    ds = params["dataset_name"]
    alpha = params["cc_alpha"]

    for rec in population:
        qid = rec["query_id"]
        query = rec["query"]

        # Route
        routed_docs = route_query(query, top_k=params["routing_topk"])

        # Retrieve dense+sparse at top-50 within routed docs
        dense = context.qdrant_retriever.retrieve(query, dataset_name=ds, doc_ids=routed_docs)
        sparse = context.bm25_retriever.retrieve(query, dataset_name=ds, doc_ids=routed_docs)

        # CC fusion at top-50
        fused = cc_fusion(sparse, dense, alpha=alpha, top_n=50)

        # Compute R@k at multiple cutoffs using GT spans
        gt_spans = ground_truth.get_spans(qid)
        gt_doc = ground_truth.get_doc_id(qid)
        spans_50 = [tuple(s) for s in fused.spans]
        ids_50 = fused.ids

        metrics = compute_all_k(spans_50, gt_spans)

        # Find first rank where GT overlap appears
        first_gt_rank = None
        for rank, (cid, span) in enumerate(zip(ids_50, spans_50)):
            doc_id = cid.split("#")[0]
            if doc_id != gt_doc:
                continue
            # Check span overlap with any GT span
            for gt_start, gt_end in gt_spans:
                overlap = max(0, min(span[1], gt_end) - max(span[0], gt_start))
                if overlap > 0:
                    first_gt_rank = rank + 1  # 1-based
                    break
            if first_gt_rank is not None:
                break

        routing_hit = gt_doc in [str(d) for d in routed_docs]

        result = {
            "query_id": qid,
            "failure_type": rec["failure_type"],
            "routing_hit": routing_hit,
            "r_at_8": metrics.r_at_k.get(8, 0.0),
            "r_at_16": metrics.r_at_k.get(16, 0.0),
            "r_at_32": metrics.r_at_k.get(32, 0.0),
            "first_gt_rank": first_gt_rank,
            "n_candidates": len(ids_50),
            "type": "?",
        }

        # Classify
        if first_gt_rank is not None and first_gt_rank <= 30:
            result["type"] = "a_recoverable"
        elif first_gt_rank is not None and first_gt_rank > 30:
            result["type"] = "a_deep"
        else:
            result["type"] = "b_not_retrieved"

        results.append(result)
        tag = f"rank={first_gt_rank}" if first_gt_rank else "NOT_FOUND"
        print(f"  [{qid}] {rec['failure_type']} route={'HIT' if routing_hit else 'MISS'} "
              f"R@8={result['r_at_8']:.2f} R@16={result['r_at_16']:.2f} R@32={result['r_at_32']:.2f} "
              f"first_gt={tag} -> {result['type']}")

    return results


def main():
    p = argparse.ArgumentParser(description="Diagnose access-miss residual")
    p.add_argument("--corpus", nargs="*", default=list(CORPUS_PARAMS.keys()),
                   choices=list(CORPUS_PARAMS.keys()))
    args = p.parse_args()

    all_results = []

    for corpus in args.corpus:
        params = CORPUS_PARAMS[corpus]
        population = get_access_miss_population(corpus)

        print(f"\n{'=' * 70}")
        print(f"CORPUS: {corpus} | {len(population)} INCORRECT+FAITHFUL queries")
        print(f"{'=' * 70}")

        if not population:
            continue

        results = run_diagnostic(corpus, population, params)
        all_results.extend(results)

        # Summary
        types = Counter(r["type"] for r in results)
        ft_types = Counter(r["failure_type"] for r in results)
        print(f"\n  {corpus} summary:")
        print(f"    Total: {len(results)}")
        print(f"    Type (a) recoverable (GT in rank 9-30): {types.get('a_recoverable', 0)}")
        print(f"    Type (a) deep (GT in rank 31-50):       {types.get('a_deep', 0)}")
        print(f"    Type (b) not retrieved (GT not in 50):  {types.get('b_not_retrieved', 0)}")
        print(f"    Failure types: {dict(ft_types)}")

        # Routing miss breakdown
        route_miss = sum(1 for r in results if not r["routing_hit"])
        print(f"    Routing miss (GT doc not in top-3): {route_miss}")

    # Cross-corpus summary
    if all_results:
        print(f"\n{'=' * 70}")
        print("CROSS-CORPUS SPLIT")
        print(f"{'=' * 70}")
        types = Counter(r["type"] for r in all_results)
        total = len(all_results)
        print(f"Total: {total}")
        a_rec = types.get("a_recoverable", 0)
        a_deep = types.get("a_deep", 0)
        b_not = types.get("b_not_retrieved", 0)
        print(f"  (a) Recoverable (GT rank 9-30):     {a_rec:3d} ({a_rec/total*100:.0f}%)")
        print(f"  (a) Deep (GT rank 31-50):            {a_deep:3d} ({a_deep/total*100:.0f}%)")
        print(f"  (b) Not retrieved (GT not in top-50): {b_not:3d} ({b_not/total*100:.0f}%)")
        print()
        if a_rec > 0:
            print(f"VIABLE: {a_rec} queries have GT evidence ranked 9-30 (critic-promotion target).")
            # Show the recoverable cases
            print(f"Recoverable cases:")
            for r in sorted(all_results, key=lambda x: x.get("first_gt_rank") or 999):
                if r["type"] == "a_recoverable":
                    print(f"  {r['query_id']}: first_gt_rank={r['first_gt_rank']} "
                          f"R@8={r['r_at_8']:.2f} R@16={r['r_at_16']:.2f} "
                          f"ft={r['failure_type']}")
        else:
            print("NOT VIABLE: no queries have recoverable GT evidence in rank 9-30.")
            print("The access-miss residual is retrieval-upstream, not reorder-fixable.")


if __name__ == "__main__":
    main()
