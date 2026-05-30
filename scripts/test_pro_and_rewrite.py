"""Experiment A (Pro synthesis) + Experiment B (Rewrite OFF) on 18 labeled MAUD failures.

Three arms on the same 18 queries, same evidence:
  1. Current baseline (flash, rewrite ON) — the control
  2. Experiment A: deepseek-pro on Phase 8, everything else identical
  3. Experiment B: rewrite OFF (different retrieval), flash on Phase 8

Usage:
    python scripts/test_pro_and_rewrite.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ── Synthesis call ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a legal document analyst. Answer the query using ONLY the "
    "provided document chunks.\n\n"
    "Rules:\n"
    "- Every factual claim in your answer must cite a specific chunk\n"
    "- cited_chunk_id must be one of the chunk IDs provided in context\n"
    "- cited_text must be copied verbatim from the cited chunk\n"
    "- One atomic fact per claim — do not bundle multiple facts\n"
    "- Do not invent information not present in the chunks\n"
    "- If the answer cannot be found in the chunks, say so explicitly\n"
    "- Produce at most 10 claims. If the answer needs more, merge "
    "closely related facts into single claims."
)


def call_synthesis(query: str, context_chunks: list[dict], model: str) -> dict:
    """Call synthesis with given model. Returns {answer, claims, tokens}."""
    import instructor
    from openai import OpenAI
    from core.supervisor.schemas_phase8 import StructuredAnswer

    context_string = "\n---\n".join(
        f"[{c['chunk_id']}]\n{c['content']}" for c in context_chunks
    )
    user_prompt = (
        f"Query: {query}\n\n"
        f"Document chunks:\n{context_string}\n\n"
        "Answer the query and provide claim-citation pairs for every "
        "factual assertion."
    )

    client = instructor.from_openai(
        OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
               base_url="https://api.deepseek.com")
    )

    for attempt in range(3):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                response_model=StructuredAnswer,
                max_retries=3,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
            elapsed = time.time() - t0
            return {
                "answer": response.answer,
                "claims": [c.model_dump() for c in response.claims],
                "elapsed_s": round(elapsed, 1),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"answer": f"[FAILED: {e}]", "claims": [], "elapsed_s": 0}


# ── Experiment B: re-run retrieval with rewrite OFF ────────────────────────

def run_retrieval_no_rewrite(qid: str, query: str) -> list[dict]:
    """Run the full pipeline with MERIDIAN_NO_REWRITE=1, return context_chunks."""
    os.environ["MERIDIAN_EMBED_MODEL"] = "voyage-4"
    os.environ["MERIDIAN_NO_RERANK"] = "1"
    os.environ["MERIDIAN_CC_ALPHA"] = "0.2"
    os.environ["MERIDIAN_ROUTING_TOPK"] = "3"
    os.environ["MERIDIAN_ROUTING_INDEX"] = "data/routing_index_maud_v4.npz"
    os.environ["MERIDIAN_ROUTING_ALPHA"] = "0.7"
    os.environ["MERIDIAN_NO_REWRITE"] = "1"

    import core.retrieval.routing as routing_mod
    routing_mod._router = None

    from core.supervisor.context import PipelineContext
    from core.supervisor.graph import compile_graph

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    context = PipelineContext.build(
        corpus_path=Path("data/corpus_maud.parquet"),
        qdrant_url=qdrant_url,
        collection_name="maud_sac_v4",
        dataset_name="maud",
        top_k=50,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    compiled, _ = compile_graph(
        db_path="data/corpus_eval_pipeline.sqlite", context=context
    )

    state = {
        "raw_query": query, "iteration": 1, "max_iterations": 1,
        "loop_complete": False, "trace_id": "local",
    }
    final = compiled.invoke(
        state, config={"configurable": {"thread_id": str(uuid.uuid4())}}
    )

    ctx_texts = final.get("context_chunks", [])
    ctx_ids = final.get("context_ids", [])
    return [
        {"chunk_id": cid, "doc_id": cid.split("#")[0], "content": text}
        for cid, text in zip(ctx_ids, ctx_texts)
    ]


def main():
    # Load the 18 labeled failures
    eval_data = {}
    with open("data/eval_maud_base_hier_ab.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            eval_data[r["query_id"]] = r

    corr_data = {}
    with open("data/judge_v2_maud_eval_maud_base_hier_ab.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corr_data[r["query_id"]] = r

    faith_data = {}
    with open("data/faith_maud_eval_maud_base_hier_ab.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            faith_data[r["query_id"]] = r

    target_qids = sorted([
        qid for qid in corr_data
        if corr_data[qid]["verdict"] == "INCORRECT"
        and faith_data[qid]["faithfulness_score"] == 1.0
    ])

    print(f"Target: {len(target_qids)} MAUD INCORRECT+FAITHFUL queries\n")

    # ── Run all three arms ────────────────────────────────────────────

    results_a = []   # Experiment A: pro synthesis
    results_b = []   # Experiment B: rewrite OFF
    chunk_diffs = []  # Track retrieval differences for B

    for i, qid in enumerate(target_qids):
        ev = eval_data[qid]
        query = ev["query"]
        ctx = ev.get("context_chunks", [])

        print(f"[{i+1}/{len(target_qids)}] {qid}")

        # Experiment A: pro on Phase 8, same evidence
        print(f"  A (pro)...", end=" ", flush=True)
        pro_result = call_synthesis(query, ctx, "deepseek-reasoner")
        print(f"{len(pro_result['claims'])} claims, {pro_result['elapsed_s']}s")

        results_a.append({
            "query_id": qid,
            "query": query,
            "answer": pro_result["answer"],
            "claims": pro_result["claims"],
            "context_chunks": ctx,
            "p_at_1": ev.get("p_at_1", 0),
            "r_at_8": ev.get("r_at_8", 0),
            "failure_type": ev["failure_type"],
            "routing_hit": ev.get("routing_hit"),
            "elapsed_s": pro_result["elapsed_s"],
        })

        # Experiment B: rewrite OFF, fresh retrieval
        print(f"  B (no-rewrite)...", end=" ", flush=True)
        norewrite_ctx = run_retrieval_no_rewrite(qid, query)
        norewrite_result = call_synthesis(query, norewrite_ctx, "deepseek-v4-flash")
        print(f"{len(norewrite_result['claims'])} claims, {norewrite_result['elapsed_s']}s")

        # Track chunk differences
        orig_ids = set(c["chunk_id"] for c in ctx)
        new_ids = set(c["chunk_id"] for c in norewrite_ctx)
        shared = orig_ids & new_ids
        only_orig = orig_ids - new_ids
        only_new = new_ids - orig_ids
        chunk_diffs.append({
            "query_id": qid,
            "shared": len(shared),
            "only_rewrite_on": len(only_orig),
            "only_rewrite_off": len(only_new),
            "total_orig": len(orig_ids),
            "total_new": len(new_ids),
        })

        results_b.append({
            "query_id": qid,
            "query": query,
            "answer": norewrite_result["answer"],
            "claims": norewrite_result["claims"],
            "context_chunks": norewrite_ctx,
            "p_at_1": ev.get("p_at_1", 0),
            "r_at_8": ev.get("r_at_8", 0),
            "failure_type": ev["failure_type"],
            "routing_hit": ev.get("routing_hit"),
            "elapsed_s": norewrite_result["elapsed_s"],
        })

    # ── Save outputs ──────────────────────────────────────────────────

    for out_path, data in [
        ("data/exp_a_pro.jsonl", results_a),
        ("data/exp_b_norewrite.jsonl", results_b),
    ]:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        print(f"\nSaved {len(data)} records to {out_path}")

    # ── Print chunk diff summary for Experiment B ─────────────────────

    print("\n\nEXPERIMENT B — RETRIEVAL DIFFERENCES (rewrite ON vs OFF):")
    print(f"{'QID':<12} {'Shared':>7} {'Only ON':>8} {'Only OFF':>9}")
    print("-" * 40)
    for d in chunk_diffs:
        print(f"{d['query_id']:<12} {d['shared']:>7} {d['only_rewrite_on']:>8} "
              f"{d['only_rewrite_off']:>9}")
    changed = sum(1 for d in chunk_diffs if d["only_rewrite_on"] > 0 or d["only_rewrite_off"] > 0)
    print(f"\nQueries with different chunks: {changed}/{len(chunk_diffs)}")

    print("\n\nNext: judge all with judge_answers_v2.py + judge_faithfulness.py")


if __name__ == "__main__":
    main()
