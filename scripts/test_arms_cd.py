"""ARM C (windowed parent) + ARM D (SAC visible) on 18 labeled MAUD failures.

ARM C: retrieve on hierarchy children, feed sibling-sections-under-same-Article
  as the synthesis context (the windowed variant). Window definition: all parent
  sections that fall within the same Article boundary as the retrieved child's
  parent. Fallback: parent + immediate neighbor parents if <2 Article markers.
  Cap: 20000 chars per window to prevent feeding entire documents.

ARM D: same retrieval as baseline, but feed sac_content (with document summary
  prefix) instead of raw content to Phase 8 synthesis.

Usage:
    python scripts/test_arms_cd.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# ── Shared synthesis call ──────────────────────────────────────────────────

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


def call_synthesis(query: str, context_chunks: list[dict]) -> dict:
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


# ── ARM C: Windowed parent retrieval + synthesis ───────────────────────────

ARTICLE_PAT = re.compile(r"^(?:ARTICLE|Article)\s+[IVXLCDM\d]+", re.MULTILINE)
WINDOW_CAP = 20000


def _find_article_boundaries(text: str) -> list[int]:
    """Return sorted offsets where Article markers occur."""
    return sorted(m.start() for m in ARTICLE_PAT.finditer(text))


def _get_windowed_context(
    child_chunk_ids: list[str],
    hier_df: pd.DataFrame,
    source_texts: dict[str, str],
) -> list[dict]:
    """For each retrieved child, find sibling parents under the same Article.

    Returns deduped parent-windowed context chunks, ordered by first-child rank.
    """
    children_df = hier_df[~hier_df["is_parent"]].reset_index(drop=True)
    parents_df = hier_df[hier_df["is_parent"]].reset_index(drop=True)
    child_to_parent = dict(zip(children_df["chunk_id"], children_df["parent_id"]))
    parent_lookup = {r["chunk_id"]: r for _, r in parents_df.iterrows()}

    # Group parents by doc
    parents_by_doc: dict[str, list] = {}
    for _, row in parents_df.iterrows():
        doc = row["doc_id"]
        if doc not in parents_by_doc:
            parents_by_doc[doc] = []
        parents_by_doc[doc].append(row)

    seen_windows: set[str] = set()  # dedup by Article key
    result_chunks: list[dict] = []

    for cid in child_chunk_ids:
        pid = child_to_parent.get(cid)
        if pid is None:
            # Leaf: use own content
            child_row = children_df[children_df["chunk_id"] == cid]
            if not child_row.empty:
                r = child_row.iloc[0]
                key = r["chunk_id"]
                if key not in seen_windows:
                    seen_windows.add(key)
                    result_chunks.append({
                        "chunk_id": r["chunk_id"],
                        "doc_id": r["doc_id"],
                        "content": r["content"],
                    })
            continue

        p_row = parent_lookup.get(pid)
        if p_row is None:
            continue

        doc_id = p_row["doc_id"]
        source = source_texts.get(doc_id, "")
        ps, pe = int(p_row["start_end_idx"][0]), int(p_row["start_end_idx"][1])

        # Find Article boundaries
        art_bounds = _find_article_boundaries(source)

        if len(art_bounds) < 2:
            # Fallback: parent + immediate neighbors
            doc_parents = sorted(
                parents_by_doc.get(doc_id, []),
                key=lambda r: int(r["start_end_idx"][0]),
            )
            parent_idx = next(
                (i for i, r in enumerate(doc_parents)
                 if r["chunk_id"] == pid), -1
            )
            if parent_idx == -1:
                continue

            window_parents = [doc_parents[parent_idx]]
            if parent_idx > 0:
                window_parents.insert(0, doc_parents[parent_idx - 1])
            if parent_idx < len(doc_parents) - 1:
                window_parents.append(doc_parents[parent_idx + 1])
            window_key = f"{doc_id}:neighbor:{pid}"
        else:
            # Find which Article this parent falls in
            article_start = 0
            article_end = len(source)
            for i, ab in enumerate(art_bounds):
                if ab <= ps:
                    article_start = ab
                    article_end = art_bounds[i + 1] if i + 1 < len(art_bounds) else len(source)

            # Gather all parents in this Article
            doc_parents = parents_by_doc.get(doc_id, [])
            window_parents = [
                r for r in doc_parents
                if int(r["start_end_idx"][0]) >= article_start
                and int(r["start_end_idx"][0]) < article_end
            ]
            window_parents.sort(key=lambda r: int(r["start_end_idx"][0]))
            window_key = f"{doc_id}:art:{article_start}"

        if window_key in seen_windows:
            continue
        seen_windows.add(window_key)

        # Build window content, capped
        window_text = ""
        window_id_parts = []
        for wp in window_parents:
            chunk_text = wp["content"]
            if len(window_text) + len(chunk_text) > WINDOW_CAP:
                break
            window_text += ("\n\n" if window_text else "") + chunk_text
            window_id_parts.append(wp["chunk_id"])

        if window_text:
            result_chunks.append({
                "chunk_id": f"window:{window_key}",
                "doc_id": doc_id,
                "content": window_text,
            })

    return result_chunks[:8]  # cap at 8 context windows


def run_arm_c(qid: str, query: str, hier_df: pd.DataFrame,
              source_texts: dict[str, str]) -> dict:
    """ARM C: retrieve on hierarchy children, feed windowed parent context."""
    os.environ["MERIDIAN_EMBED_MODEL"] = "voyage-4"
    os.environ["MERIDIAN_NO_RERANK"] = "1"
    os.environ["MERIDIAN_CC_ALPHA"] = "0.2"
    os.environ["MERIDIAN_ROUTING_TOPK"] = "3"
    os.environ["MERIDIAN_ROUTING_INDEX"] = "data/routing_index_maud_v4.npz"
    os.environ["MERIDIAN_ROUTING_ALPHA"] = "0.7"

    import core.retrieval.routing as routing_mod
    routing_mod._router = None

    from core.supervisor.context import PipelineContext
    from core.supervisor.graph import compile_graph

    children_df = hier_df[~hier_df["is_parent"]].reset_index(drop=True)

    context = PipelineContext.build(
        corpus_path=Path("data/corpus_maud.parquet"),
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection_name="maud_hier_sac_v4",
        dataset_name="maud",
        top_k=50,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    # Override with hierarchy children
    context.corpus_df = children_df
    context.qdrant_retriever._corpus_df = children_df
    context.qdrant_retriever._id_to_idx = {
        row["chunk_id"]: i for i, row in children_df.iterrows()
    }
    from core.retrieval.bm25_retriever import BM25Retriever
    context.bm25_retriever = BM25Retriever(children_df, top_k=50)
    # NO parent_df — we handle windowing ourselves

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

    # Get the child chunk_ids from retrieval (Phase 7 output without parent swap)
    child_ids = final.get("context_ids", [])

    # Build windowed context from these children
    windowed_ctx = _get_windowed_context(child_ids, hier_df, source_texts)

    # Re-run synthesis on windowed context
    result = call_synthesis(query, windowed_ctx)
    result["context_chunks"] = windowed_ctx
    return result


# ── ARM D: SAC visible ────────────────────────────────────────────────────

def run_arm_d(qid: str, query: str, eval_record: dict,
              sac_df: pd.DataFrame) -> dict:
    """ARM D: same retrieval, but feed sac_content instead of content."""
    ctx = eval_record.get("context_chunks", [])

    # Build sac lookup
    sac_lookup = dict(zip(sac_df["chunk_id"], sac_df["sac_content"]))

    # Replace content with sac_content where available
    sac_ctx = []
    for c in ctx:
        cid = c["chunk_id"]
        sac_text = sac_lookup.get(cid)
        sac_ctx.append({
            "chunk_id": cid,
            "doc_id": c["doc_id"],
            "content": sac_text if sac_text else c["content"],
        })

    result = call_synthesis(query, sac_ctx)
    result["context_chunks"] = sac_ctx
    return result


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

    print(f"Target: {len(target_qids)} MAUD INCORRECT+FAITHFUL queries\n")

    # Load hierarchy parquet and source texts for ARM C
    hier_df = pd.read_parquet("data/corpus_maud_hier_sac.parquet")
    source_texts = {}
    corpus_dir = Path("data/corpus/maud")
    for txt_file in corpus_dir.glob("*.txt"):
        doc_id = f"maud/{txt_file.name}"
        source_texts[doc_id] = txt_file.read_text(encoding="utf-8", errors="replace")

    # Load SAC parquet for ARM D
    sac_df = pd.read_parquet("data/corpus_maud_sac.parquet")

    results_c = []
    results_d = []

    for i, qid in enumerate(target_qids):
        ev = eval_data[qid]
        query = ev["query"]

        print(f"[{i+1}/{len(target_qids)}] {qid}")

        # ARM C: windowed parent
        print(f"  C (windowed)...", end=" ", flush=True)
        c_result = run_arm_c(qid, query, hier_df, source_texts)
        print(f"{len(c_result['claims'])} claims")

        results_c.append({
            "query_id": qid,
            "query": query,
            "answer": c_result["answer"],
            "claims": c_result["claims"],
            "context_chunks": c_result["context_chunks"],
            "p_at_1": ev.get("p_at_1", 0),
            "r_at_8": ev.get("r_at_8", 0),
            "failure_type": ev["failure_type"],
            "routing_hit": ev.get("routing_hit"),
        })

        # ARM D: SAC visible
        print(f"  D (SAC)...", end=" ", flush=True)
        d_result = run_arm_d(qid, query, ev, sac_df)
        print(f"{len(d_result['claims'])} claims")

        results_d.append({
            "query_id": qid,
            "query": query,
            "answer": d_result["answer"],
            "claims": d_result["claims"],
            "context_chunks": d_result["context_chunks"],
            "p_at_1": ev.get("p_at_1", 0),
            "r_at_8": ev.get("r_at_8", 0),
            "failure_type": ev["failure_type"],
            "routing_hit": ev.get("routing_hit"),
        })

    # Save
    for out_path, data in [
        ("data/exp_c_windowed.jsonl", results_c),
        ("data/exp_d_sac.jsonl", results_d),
    ]:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        print(f"\nSaved {len(data)} records to {out_path}")

    print("\nNext: judge all with judge_answers_v2.py + judge_faithfulness.py")


if __name__ == "__main__":
    main()
