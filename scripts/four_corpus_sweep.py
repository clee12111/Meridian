"""Four-corpus retrieval sweep on voyage-4.

Stage 1: Chunk-α sweep per corpus (routing OFF)
Stage 2: Routing-α sweep per corpus (chunk-α fixed)
Stage 3: Full validation with winning αs
Stage 4: ZeroEntropy baseline comparison

Usage:
    python scripts/four_corpus_sweep.py             # all stages
    python scripts/four_corpus_sweep.py --stage 1   # chunk-α only
    python scripts/four_corpus_sweep.py --stage 3   # full validation only
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

CORPORA = [
    {
        "name": "ContractNLI",
        "benchmark": Path("data/benchmarks/contractnli.json"),
        "corpus_parquet": Path("data/corpus_contractnli.parquet"),
        "corpus_dir": Path("data/corpus"),
        "collection": "contractnli_sac_v4",
        "dataset_name": "contractnli",
        "gt_class": "ContractNLIGroundTruth",
        "gt_module": "core.evaluation.ground_truth_contractnli",
        "routing_index": "data/routing_index_v4.npz",
    },
    {
        "name": "PrivacyQA",
        "benchmark": Path("data/benchmarks/privacy_qa.json"),
        "corpus_parquet": Path("data/corpus_privacy_qa.parquet"),
        "corpus_dir": Path("data/corpus"),
        "collection": "privacyqa_sac_v4",
        "dataset_name": "privacy_qa",
        "gt_class": "PrivacyQAGroundTruth",
        "gt_module": "core.evaluation.ground_truth_privacyqa",
        "routing_index": "data/routing_index_privacyqa_v4.npz",
    },
    {
        "name": "CUAD",
        "benchmark": Path("data/benchmarks/cuad.json"),
        "corpus_parquet": Path("data/corpus_cuad.parquet"),
        "corpus_dir": Path("data/corpus"),
        "collection": "cuad_sac_v4",
        "dataset_name": "cuad",
        "gt_class": "CUADGroundTruth",
        "gt_module": "core.evaluation.ground_truth_cuad",
        "routing_index": "data/routing_index_cuad_v4.npz",
    },
    {
        "name": "MAUD",
        "benchmark": Path("data/benchmarks/maud.json"),
        "corpus_parquet": Path("data/corpus_maud.parquet"),
        "corpus_dir": Path("data/corpus"),
        "collection": "maud_sac_v4",
        "dataset_name": "maud",
        "gt_class": "MAUDGroundTruth",
        "gt_module": "core.evaluation.ground_truth_maud",
        "routing_index": "data/routing_index_maud_v4.npz",
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


def run_config(
    corpus_cfg: dict,
    chunk_alpha: float | None,
    routing_topk: int | None = None,
    routing_alpha: float | None = None,
    n_queries: int | None = None,
) -> dict:
    """Run N queries on a corpus with given config. Returns metrics dict."""
    from core.measurement.metrics import compute_all_k
    from core.measurement.taxonomy import classify_with_confidence
    from core.supervisor.context import PipelineContext
    from core.supervisor.graph import compile_graph

    # Set env flags
    os.environ["MERIDIAN_NO_RERANK"] = "1"
    os.environ["MERIDIAN_EMBED_MODEL"] = "voyage-4"

    if chunk_alpha is not None:
        os.environ["MERIDIAN_CC_ALPHA"] = str(chunk_alpha)
    elif "MERIDIAN_CC_ALPHA" in os.environ:
        del os.environ["MERIDIAN_CC_ALPHA"]

    if routing_topk is not None:
        os.environ["MERIDIAN_ROUTING_TOPK"] = str(routing_topk)
        os.environ["MERIDIAN_ROUTING_INDEX"] = corpus_cfg["routing_index"]
        if routing_alpha is not None:
            os.environ["MERIDIAN_ROUTING_ALPHA"] = str(routing_alpha)
        elif "MERIDIAN_ROUTING_ALPHA" in os.environ:
            del os.environ["MERIDIAN_ROUTING_ALPHA"]
    else:
        if "MERIDIAN_ROUTING_TOPK" in os.environ:
            del os.environ["MERIDIAN_ROUTING_TOPK"]

    # Reset routing singleton so it reloads the correct index
    import core.retrieval.routing as routing_mod
    routing_mod._router = None

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
        db_path="data/sweep_pipeline.sqlite",
        context=context,
    )

    mod = importlib.import_module(corpus_cfg["gt_module"])
    gt_class = getattr(mod, corpus_cfg["gt_class"])
    ground_truth = gt_class(
        benchmark_path=corpus_cfg["benchmark"],
        corpus_dir=corpus_cfg["corpus_dir"],
    )

    data = json.loads(corpus_cfg["benchmark"].read_text(encoding="utf-8"))
    questions = [
        {"query_id": t["query_id"], "query": t["query"]}
        for t in data["tests"]
    ]
    if n_queries is not None:
        questions = questions[:n_queries]

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
                print(f"    [{query_id}] FAILED: {exc}", file=sys.stderr)
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

        # Routing recall
        routed_docs = final_state.get("routed_docs")
        routing_hit = None
        if routed_docs is not None:
            routing_hit = gt_doc_id in routed_docs

        record = {
            "query_id": query_id,
            "p_at_1": metric_result.p_at_k.get(1, 0.0),
            "p_at_4": metric_result.p_at_k.get(4, 0.0),
            "r_at_1": metric_result.r_at_k.get(1, 0.0),
            "r_at_8": metric_result.r_at_k.get(8, 0.0),
            "r_at_16": metric_result.r_at_k.get(16, 0.0),
            "r_at_64": metric_result.r_at_k.get(64, 0.0),
            "failure_type": classification.failure_type.name,
            "routing_hit": routing_hit,
        }

        with write_lock:
            results.append(record)

        return record

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_one, q): q for q in questions}
        for future in as_completed(futures):
            future.result()

    n = len(results)
    if n == 0:
        return {"n": 0}

    ft_counts = Counter(r["failure_type"] for r in results)
    routed = [r for r in results if r["routing_hit"] is not None]
    routing_recall = (
        sum(1 for r in routed if r["routing_hit"]) / len(routed) * 100
        if routed else None
    )

    return {
        "n": n,
        "p1": sum(r["p_at_1"] for r in results) / n,
        "p4": sum(r["p_at_4"] for r in results) / n,
        "r1": sum(r["r_at_1"] for r in results) / n,
        "r8": sum(r["r_at_8"] for r in results) / n,
        "r16": sum(r["r_at_16"] for r in results) / n,
        "r64": sum(r["r_at_64"] for r in results) / n,
        "drm": ft_counts.get("DRM", 0) / n * 100,
        "ok": ft_counts.get("OK", 0) / n * 100,
        "ft": dict(ft_counts),
        "routing_recall": routing_recall,
    }


# ── Stage 1: Chunk-α sweep ──────────────────────────────────────────────

CHUNK_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]


def stage1(corpora: list[dict]) -> dict[str, dict]:
    """Chunk-α sweep, routing OFF, 50 queries per corpus."""
    print("\n" + "=" * 80)
    print("STAGE 1: Chunk-alpha sweep (routing OFF, 50 queries)")
    print("=" * 80)

    results: dict[str, dict] = {}

    for cfg in corpora:
        name = cfg["name"]
        results[name] = {}
        for alpha in CHUNK_ALPHAS:
            label = f"{alpha:.1f}"
            print(f"  [{name}] chunk-alpha={label}...", end="", flush=True)
            m = run_config(cfg, chunk_alpha=alpha, n_queries=50)
            results[name][label] = m
            print(f" P@1={m['p1']:.3f} R@8={m['r8']:.3f} DRM={m['drm']:.0f}%")

    # Print table
    print(f"\n{'alpha':>5}", end="")
    for cfg in corpora:
        print(f" | {cfg['name']:>12} R@8 {cfg['name']:>6} DRM", end="")
    print()
    print("-" * (5 + len(corpora) * 28))

    best_alpha: dict[str, str] = {}
    for name in [c["name"] for c in corpora]:
        best_alpha[name] = max(results[name], key=lambda a: results[name][a]["r8"])

    for alpha in [f"{a:.1f}" for a in CHUNK_ALPHAS]:
        print(f"{alpha:>5}", end="")
        for cfg in corpora:
            name = cfg["name"]
            m = results[name].get(alpha, {})
            marker = " *" if alpha == best_alpha[name] else "  "
            print(f" | {m.get('r8', 0):>12.4f}  {m.get('drm', 0):>6.1f}%{marker}", end="")
        print()

    print("\nBest chunk-alpha per corpus:")
    for name, alpha in best_alpha.items():
        m = results[name][alpha]
        print(f"  {name}: alpha={alpha} (R@8={m['r8']:.4f}, DRM={m['drm']:.1f}%)")

    return {name: {"chunk_alpha": float(best_alpha[name]), "stage1": results[name]}
            for name in best_alpha}


# ── Stage 2: Routing-α sweep ────────────────────────────────────────────

ROUTING_ALPHAS = [0.3, 0.5, 0.7]


def stage2(corpora: list[dict], stage1_results: dict[str, dict]) -> dict[str, dict]:
    """Routing-α sweep, chunk-α fixed, 50 queries per corpus."""
    print("\n" + "=" * 80)
    print("STAGE 2: Routing-alpha sweep (chunk-alpha fixed, 50 queries)")
    print("=" * 80)

    results: dict[str, dict] = {}

    for cfg in corpora:
        name = cfg["name"]
        chunk_alpha = stage1_results[name]["chunk_alpha"]
        results[name] = {"chunk_alpha": chunk_alpha}

        # Also run with routing OFF for baseline comparison
        print(f"  [{name}] chunk-alpha={chunk_alpha}, routing=OFF...", end="", flush=True)
        m_off = run_config(cfg, chunk_alpha=chunk_alpha, n_queries=50)
        results[name]["OFF"] = m_off
        print(f" R@8={m_off['r8']:.3f} DRM={m_off['drm']:.0f}%")

        for ralpha in ROUTING_ALPHAS:
            label = f"{ralpha:.1f}"
            print(f"  [{name}] routing-alpha={label}...", end="", flush=True)
            m = run_config(cfg, chunk_alpha=chunk_alpha,
                          routing_topk=3, routing_alpha=ralpha, n_queries=50)
            results[name][label] = m
            rr = m.get("routing_recall")
            rr_str = f"{rr:.0f}%" if rr is not None else "n/a"
            print(f" R@8={m['r8']:.3f} DRM={m['drm']:.0f}% rr={rr_str}")

    # Print table
    print(f"\n{'config':>10}", end="")
    for cfg in corpora:
        print(f" | {cfg['name']:>8} R@8 {cfg['name']:>5} DRM {'rr':>4}", end="")
    print()
    print("-" * (10 + len(corpora) * 30))

    for label in ["OFF"] + [f"{a:.1f}" for a in ROUTING_ALPHAS]:
        disp = f"rt={label}" if label != "OFF" else "no-route"
        print(f"{disp:>10}", end="")
        for cfg in corpora:
            name = cfg["name"]
            m = results[name].get(label, {})
            rr = m.get("routing_recall")
            rr_str = f"{rr:>3.0f}%" if rr is not None else "   -"
            print(f" | {m.get('r8', 0):>8.4f}  {m.get('drm', 0):>5.1f}% {rr_str}", end="")
        print()

    # Determine best routing config per corpus
    print("\nRouting verdict per corpus:")
    routing_config: dict[str, dict] = {}
    for cfg in corpora:
        name = cfg["name"]
        chunk_alpha = results[name]["chunk_alpha"]

        # Compare routing ON (best α) vs OFF
        m_off = results[name]["OFF"]
        best_routing_label = None
        best_routing_m = None
        for ralpha in ROUTING_ALPHAS:
            label = f"{ralpha:.1f}"
            m = results[name].get(label, {})
            rr = m.get("routing_recall")
            if rr is not None and rr < 60:
                continue  # too low routing recall
            if best_routing_m is None or m.get("r8", 0) > best_routing_m.get("r8", 0):
                best_routing_label = label
                best_routing_m = m

        # Does routing help?
        if best_routing_m and best_routing_m.get("r8", 0) > m_off.get("r8", 0):
            rr = best_routing_m.get("routing_recall", 0)
            print(f"  {name}: ROUTING HELPS (alpha={best_routing_label}, "
                  f"R@8 {m_off['r8']:.3f}->{best_routing_m['r8']:.3f}, "
                  f"DRM {m_off['drm']:.0f}%->{best_routing_m['drm']:.0f}%, "
                  f"routing recall={rr:.0f}%)")
            routing_config[name] = {
                "chunk_alpha": chunk_alpha,
                "routing_topk": 3,
                "routing_alpha": float(best_routing_label),
            }
        else:
            print(f"  {name}: ROUTING OFF (no improvement or recall too low)")
            routing_config[name] = {
                "chunk_alpha": chunk_alpha,
                "routing_topk": None,
                "routing_alpha": None,
            }

    return routing_config


# ── Stage 3: Full validation ────────────────────────────────────────────


def stage3(corpora: list[dict], routing_config: dict[str, dict]) -> dict[str, dict]:
    """Full query set with winning αs."""
    print("\n" + "=" * 80)
    print("STAGE 3: Full validation (all queries, winning alphas)")
    print("=" * 80)

    results: dict[str, dict] = {}

    for cfg in corpora:
        name = cfg["name"]
        rc = routing_config[name]
        ca = rc["chunk_alpha"]
        rt = rc.get("routing_topk")
        ra = rc.get("routing_alpha")

        config_str = f"chunk-alpha={ca}"
        if rt:
            config_str += f", routing(top-{rt}, alpha={ra})"
        else:
            config_str += ", no routing"

        print(f"\n  [{name}] {config_str} (full)...", flush=True)
        m = run_config(cfg, chunk_alpha=ca, routing_topk=rt,
                      routing_alpha=ra, n_queries=None)
        results[name] = m

        rr = m.get("routing_recall")
        rr_str = f", routing recall={rr:.1f}%" if rr is not None else ""
        print(f"    n={m['n']} P@1={m['p1']:.4f} R@8={m['r8']:.4f} "
              f"DRM={m['drm']:.1f}%{rr_str}")
        ft = m.get("ft", {})
        print(f"    Failure dist: " + " ".join(
            f"{k}={v}" for k, v in sorted(ft.items())))

    # Print full table
    print(f"\n{'Metric':<12}", end="")
    for cfg in corpora:
        print(f" {cfg['name']:>14}", end="")
    print()
    print("-" * (12 + len(corpora) * 15))

    for metric, key in [("P@1", "p1"), ("P@4", "p4"), ("R@1", "r1"),
                        ("R@8", "r8"), ("R@16", "r16"), ("R@64", "r64"),
                        ("DRM%", "drm"), ("OK%", "ok")]:
        print(f"{metric:<12}", end="")
        for cfg in corpora:
            v = results[cfg["name"]].get(key, 0)
            if "%" in metric:
                print(f" {v:>13.1f}%", end="")
            else:
                print(f" {v:>14.4f}", end="")
        print()

    # Routing recall row
    print(f"{'Route rec%':<12}", end="")
    for cfg in corpora:
        rr = results[cfg["name"]].get("routing_recall")
        if rr is not None:
            print(f" {rr:>13.1f}%", end="")
        else:
            print(f" {'n/a':>14}", end="")
    print()

    # Failure taxonomy
    print(f"\n{'Failure':<8}", end="")
    for cfg in corpora:
        print(f" {cfg['name']:>14}", end="")
    print()
    print("-" * (8 + len(corpora) * 15))
    for ft_name in ["OK", "DRM", "CBF", "SGP", "ICR", "OVR"]:
        print(f"{ft_name:<8}", end="")
        for cfg in corpora:
            ft = results[cfg["name"]].get("ft", {})
            n = results[cfg["name"]]["n"]
            count = ft.get(ft_name, 0)
            print(f" {count:>5} ({count/n*100:>4.1f}%)", end="")
        print()

    return results


# ── Stage 4: ZeroEntropy comparison ─────────────────────────────────────

# LegalBench-RAG RCTS baselines (published in the paper)
ZEROENTROPY_BASELINES = {
    "ContractNLI": {"P@1": 0.0884, "R@8": 0.5029},
    "PrivacyQA": {"P@1": None, "R@8": None},  # not yet available
    "CUAD": {"P@1": None, "R@8": None},  # not yet available
    "MAUD": {"P@1": 0.0265, "R@8": 0.0618},
}


def stage4(results: dict[str, dict]) -> None:
    """Print ZeroEntropy baseline comparison."""
    print("\n" + "=" * 80)
    print("STAGE 4: ZeroEntropy LegalBench-RAG baseline comparison")
    print("=" * 80)

    print(f"\n{'Corpus':<14} {'Metric':<8} {'ZeroEntropy':>12} {'Meridian':>12} {'Delta':>10}")
    print("-" * 58)

    for name, baselines in ZEROENTROPY_BASELINES.items():
        m = results.get(name, {})
        for metric_name, key in [("P@1", "p1"), ("R@8", "r8")]:
            ze = baselines.get(metric_name)
            ours = m.get(key, 0)
            if ze is not None:
                delta = (ours - ze) * 100
                print(f"{name:<14} {metric_name:<8} {ze:>12.4f} {ours:>12.4f} {delta:>+9.1f}pp")
            else:
                print(f"{name:<14} {metric_name:<8} {'n/a':>12} {ours:>12.4f} {'n/a':>10}")

    print("\nNOTE: ZeroEntropy baselines are the RCTS (v1 locked) numbers where available.")
    print("Missing baselines flagged as n/a — need published per-dataset numbers.")


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Four-corpus retrieval sweep")
    p.add_argument("--stage", type=int, default=None,
                   help="Run only this stage (1-4)")
    args = p.parse_args()

    corpora = CORPORA

    if args.stage is None or args.stage == 1:
        s1 = stage1(corpora)
    else:
        # Use prior results or defaults
        s1 = {
            "ContractNLI": {"chunk_alpha": 0.3},
            "PrivacyQA": {"chunk_alpha": 0.1},
            "CUAD": {"chunk_alpha": 0.3},
            "MAUD": {"chunk_alpha": 0.3},
        }

    if args.stage is None or args.stage == 2:
        routing_cfg = stage2(corpora, s1)
    else:
        routing_cfg = {
            name: {"chunk_alpha": s1[name]["chunk_alpha"],
                   "routing_topk": 3, "routing_alpha": 0.5}
            for name in s1
        }

    if args.stage is None or args.stage == 3:
        full_results = stage3(corpora, routing_cfg)
    else:
        full_results = {}

    if (args.stage is None or args.stage == 4) and full_results:
        stage4(full_results)

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
