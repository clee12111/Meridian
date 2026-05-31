"""Strategy 1 (Multi-query retrieval) + Strategy 2 (Decomposition) on 18 MAUD failures.

Baseline for both: rewrite-OFF (the validated cheap win from Finding 34).

Strategy 1: fan-out 3 query reformulations, retrieve for each, fuse results.
Strategy 2: decompose query into sub-questions, answer each grounded, combine.

Usage:
    python scripts/test_multiquery_decomp.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# ── Shared ─────────────────────────────────────────────────────────────────

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


def call_llm(system: str, user: str, model: str = "deepseek-v4-flash",
             max_tokens: int = 1024) -> str:
    """Raw LLM call, returns text."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"[FAILED: {e}]"


def call_synthesis(query: str, context_chunks: list[dict]) -> dict:
    """Structured synthesis call."""
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
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                response_model=StructuredAnswer,
                max_retries=3,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
            return {
                "answer": response.answer,
                "claims": [c.model_dump() for c in response.claims],
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"answer": f"[FAILED: {e}]", "claims": []}


# ── Retrieval (rewrite-OFF baseline) ───────────────────────────────────────

def setup_pipeline():
    """Build pipeline context for MAUD baseline retrieval (rewrite OFF)."""
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

    context = PipelineContext.build(
        corpus_path=Path("data/corpus_maud.parquet"),
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection_name="maud_sac_v4",
        dataset_name="maud",
        top_k=50,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    compiled, _ = compile_graph(
        db_path="data/corpus_eval_pipeline.sqlite", context=context
    )
    return context, compiled


def run_retrieval(query: str, compiled) -> tuple[list[dict], list[str]]:
    """Run pipeline, return (context_chunks, context_ids)."""
    state = {
        "raw_query": query, "iteration": 1, "max_iterations": 1,
        "loop_complete": False, "trace_id": "local",
    }
    final = compiled.invoke(
        state, config={"configurable": {"thread_id": str(uuid.uuid4())}}
    )
    ctx_texts = final.get("context_chunks", [])
    ctx_ids = final.get("context_ids", [])
    chunks = [
        {"chunk_id": cid, "doc_id": cid.split("#")[0], "content": text}
        for cid, text in zip(ctx_ids, ctx_texts)
    ]
    return chunks, ctx_ids


# ── STRATEGY 1: Multi-query retrieval ──────────────────────────────────────

def generate_reformulations(query: str) -> list[str]:
    """Generate 2 alternative reformulations of the query."""
    prompt = (
        "You are helping a legal retrieval system. Given the original query, "
        "produce exactly 2 alternative reformulations that might retrieve "
        "different relevant passages from a merger agreement.\n\n"
        "Reformulation 1: Use specific legal terminology and section references.\n"
        "Reformulation 2: Use broader conceptual language about the legal concept.\n\n"
        "Return ONLY the two reformulations, one per line, no numbering or labels.\n\n"
        f"Original query: {query}"
    )
    raw = call_llm("You are a legal query reformulator.", prompt, max_tokens=300)
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    return lines[:2]


def run_multiquery(qid: str, query: str, compiled) -> dict:
    """Strategy 1: fan-out retrieval with 3 queries (raw + 2 reformulations)."""
    # Generate reformulations
    reformulations = generate_reformulations(query)

    # Retrieve for each query variant
    all_queries = [query] + reformulations
    all_chunks: dict[str, dict] = {}  # chunk_id -> chunk, dedup
    chunk_scores: dict[str, float] = {}  # best rank score

    for qi, q in enumerate(all_queries):
        chunks, _ = run_retrieval(q, compiled)
        for rank, c in enumerate(chunks):
            cid = c["chunk_id"]
            rank_score = 1.0 / (rank + 1)  # RRF-style score
            if cid not in all_chunks or rank_score > chunk_scores.get(cid, 0):
                all_chunks[cid] = c
                chunk_scores[cid] = chunk_scores.get(cid, 0) + rank_score

    # Sort by fused score, take top 8
    sorted_chunks = sorted(all_chunks.values(),
                           key=lambda c: chunk_scores[c["chunk_id"]], reverse=True)
    top_chunks = sorted_chunks[:8]

    # Track overlap with single-query
    single_ids = set(c["chunk_id"] for c, _ in zip(*run_retrieval(query, compiled)))

    result = call_synthesis(query, top_chunks)
    result["context_chunks"] = top_chunks
    result["n_queries"] = len(all_queries)
    result["reformulations"] = reformulations
    result["chunk_overlap_with_single"] = len(
        set(c["chunk_id"] for c in top_chunks) & single_ids
    )
    result["unique_from_fanout"] = len(
        set(c["chunk_id"] for c in top_chunks) - single_ids
    )
    return result


# ── STRATEGY 2: Decomposition ─────────────────────────────────────────────

def decompose_query(query: str) -> list[str]:
    """Break query into 2-4 atomic sub-questions."""
    prompt = (
        "You are analyzing a legal query about a merger agreement. "
        "Break this query into 2-4 atomic sub-questions that together "
        "answer the original query. Each sub-question should be answerable "
        "from a single section or clause.\n\n"
        "Return ONLY the sub-questions, one per line, no numbering.\n\n"
        f"Query: {query}"
    )
    raw = call_llm("You decompose complex legal queries into atomic sub-questions.",
                    prompt, max_tokens=400)
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    return lines[:4]


def run_decomposition(qid: str, query: str, compiled) -> dict:
    """Strategy 2: decompose, answer sub-questions, combine."""
    sub_questions = decompose_query(query)

    # Retrieve for the ORIGINAL query (rewrite-off) — same retrieval as baseline
    chunks, _ = run_retrieval(query, compiled)

    context_string = "\n---\n".join(
        f"[{c['chunk_id']}]\n{c['content']}" for c in chunks
    )

    # Answer each sub-question grounded in the same evidence
    sub_answers = []
    for sq in sub_questions:
        sa = call_llm(
            "You are a legal document analyst. Answer ONLY from the provided chunks. "
            "If the answer is not in the chunks, say 'Not found in provided chunks.'",
            f"Sub-question: {sq}\n\nDocument chunks:\n{context_string}\n\n"
            "Answer concisely, citing the chunk ID that supports your answer.",
            max_tokens=500,
        )
        sub_answers.append({"sub_question": sq, "sub_answer": sa})

    # Combine sub-answers into final structured answer
    sub_answer_text = "\n".join(
        f"Q: {sa['sub_question']}\nA: {sa['sub_answer']}"
        for sa in sub_answers
    )

    combine_prompt = (
        f"Original query: {query}\n\n"
        f"Sub-question answers (verified against the document):\n{sub_answer_text}\n\n"
        f"Document chunks:\n{context_string}\n\n"
        "Using ONLY the sub-answers above and the document chunks, produce a "
        "complete answer to the original query with claim-citation pairs."
    )

    import instructor
    from openai import OpenAI
    from core.supervisor.schemas_phase8 import StructuredAnswer

    client = instructor.from_openai(
        OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
               base_url="https://api.deepseek.com")
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                response_model=StructuredAnswer,
                max_retries=3,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": combine_prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
            return {
                "answer": response.answer,
                "claims": [c.model_dump() for c in response.claims],
                "context_chunks": chunks,
                "sub_questions": sub_questions,
                "sub_answers": sub_answers,
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {
                    "answer": f"[FAILED: {e}]", "claims": [],
                    "context_chunks": chunks,
                    "sub_questions": sub_questions,
                    "sub_answers": sub_answers,
                }


# ── Main ──────────────────────────────────────────────────────────────────

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

    print(f"Target: {len(target_qids)} MAUD INCORRECT+FAITHFUL queries")
    print(f"Baseline: rewrite-OFF (the Finding 34 cheap win)\n")

    # Setup pipeline once
    context, compiled = setup_pipeline()

    results_mq = []
    results_dc = []

    for i, qid in enumerate(target_qids):
        ev = eval_data[qid]
        query = ev["query"]

        print(f"[{i+1}/{len(target_qids)}] {qid}")

        # Strategy 1: Multi-query
        print(f"  S1 (multi-query)...", end=" ", flush=True)
        mq = run_multiquery(qid, query, compiled)
        print(f"{len(mq['claims'])} claims, "
              f"overlap={mq.get('chunk_overlap_with_single',0)}/8, "
              f"unique={mq.get('unique_from_fanout',0)}")

        results_mq.append({
            "query_id": qid, "query": query,
            "answer": mq["answer"], "claims": mq["claims"],
            "context_chunks": mq["context_chunks"],
            "p_at_1": ev.get("p_at_1", 0), "r_at_8": ev.get("r_at_8", 0),
            "failure_type": ev["failure_type"],
            "routing_hit": ev.get("routing_hit"),
        })

        # Strategy 2: Decomposition
        print(f"  S2 (decomposition)...", end=" ", flush=True)
        dc = run_decomposition(qid, query, compiled)
        print(f"{len(dc['claims'])} claims, "
              f"{len(dc.get('sub_questions',[]))} sub-Qs")

        results_dc.append({
            "query_id": qid, "query": query,
            "answer": dc["answer"], "claims": dc["claims"],
            "context_chunks": dc["context_chunks"],
            "p_at_1": ev.get("p_at_1", 0), "r_at_8": ev.get("r_at_8", 0),
            "failure_type": ev["failure_type"],
            "routing_hit": ev.get("routing_hit"),
        })

    # Save
    for path, data in [
        ("data/exp_s1_multiquery.jsonl", results_mq),
        ("data/exp_s2_decomp.jsonl", results_dc),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        print(f"\nSaved {len(data)} records to {path}")

    print("\nNext: judge with judge_answers_v2.py + judge_faithfulness.py")


if __name__ == "__main__":
    main()
