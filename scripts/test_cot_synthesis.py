"""Test chain-of-thought synthesis on labeled MAUD failure queries.

Runs both current prompt and CoT prompt on the same evidence (context_chunks
from eval file), judges both with the pinned correctness judge, reports flips.

Usage:
    python scripts/test_cot_synthesis.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ── Prompts ────────────────────────────────────────────────────────────────

CURRENT_SYSTEM = (
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

COT_SYSTEM = (
    "You are a legal document analyst. Answer the query using ONLY the "
    "provided document chunks.\n\n"
    "IMPORTANT: Before producing your answer, you MUST first reason through "
    "the evidence step by step in a <reasoning> block:\n"
    "1. For each chunk, state what it says that is relevant to the query.\n"
    "2. Identify which chunks directly address the query vs which are tangential.\n"
    "3. Note any cross-references between chunks (e.g. 'Section X refers to Y').\n"
    "4. State what the evidence supports and what it does NOT support.\n"
    "5. If the query asks about a specific provision and you find it, quote it.\n"
    "   If you do NOT find it, say 'Not found in provided chunks.'\n\n"
    "After your reasoning, produce the structured answer.\n\n"
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


def build_user_prompt(query: str, context_chunks: list[dict]) -> str:
    context_string = "\n---\n".join(
        f"[{c['chunk_id']}]\n{c['content']}" for c in context_chunks
    )
    return (
        f"Query: {query}\n\n"
        f"Document chunks:\n{context_string}\n\n"
        "Answer the query and provide claim-citation pairs for every "
        "factual assertion."
    )


def call_synthesis(system_prompt: str, user_prompt: str) -> dict:
    """Call DeepSeek synthesis and return structured answer."""
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
                    {"role": "system", "content": system_prompt},
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


def main():
    # Load labeled failure queries
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

    # INCORRECT+FAITHFUL = grounded-but-wrong
    target_qids = sorted([
        qid for qid in corr_data
        if corr_data[qid]["verdict"] == "INCORRECT"
        and faith_data[qid]["faithfulness_score"] == 1.0
    ])

    print(f"Target queries: {len(target_qids)} INCORRECT+FAITHFUL")
    print()

    # Run both prompts on each query
    results = []
    for i, qid in enumerate(target_qids):
        ev = eval_data[qid]
        query = ev["query"]
        ctx = ev.get("context_chunks", [])
        user_prompt = build_user_prompt(query, ctx)

        print(f"[{i+1}/{len(target_qids)}] {qid}...")

        # Current prompt
        current_result = call_synthesis(CURRENT_SYSTEM, user_prompt)

        # CoT prompt
        cot_result = call_synthesis(COT_SYSTEM, user_prompt)

        results.append({
            "query_id": qid,
            "query": query[:120],
            "failure_type": ev["failure_type"],
            "original_verdict": corr_data[qid]["verdict"],
            "original_reason": corr_data[qid].get("reason", "")[:200],
            "current_answer": current_result["answer"][:300],
            "current_claims": current_result["claims"],
            "cot_answer": cot_result["answer"][:300],
            "cot_claims": cot_result["claims"],
        })
        print(f"  current: {len(current_result['claims'])} claims")
        print(f"  cot: {len(cot_result['claims'])} claims")

    # Save for judging
    out_current = Path("data/cot_test_current.jsonl")
    out_cot = Path("data/cot_test_cot.jsonl")

    for out_path, key in [(out_current, "current"), (out_cot, "cot")]:
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                record = {
                    "query_id": r["query_id"],
                    "query": eval_data[r["query_id"]]["query"],
                    "answer": r[f"{key}_answer"],
                    "claims": r[f"{key}_claims"],
                    "context_chunks": eval_data[r["query_id"]].get("context_chunks", []),
                    "p_at_1": eval_data[r["query_id"]].get("p_at_1", 0),
                    "r_at_8": eval_data[r["query_id"]].get("r_at_8", 0),
                    "failure_type": r["failure_type"],
                    "routing_hit": eval_data[r["query_id"]].get("routing_hit"),
                }
                f.write(json.dumps(record) + "\n")
        print(f"\nSaved {len(results)} records to {out_path}")

    print("\nNext: judge both files with judge_answers_v2.py + judge_faithfulness.py")


if __name__ == "__main__":
    main()
