"""Headline measurement on the COMBINED index (benchmark regime).

All 4 corpora pooled into ONE index (~11.5K SAC chunks from 72 mini-split
documents). This matches the paper's regime (cross-corpus distractors).

Arms:
  0 — BASELINE: RRF fusion, no routing, no selector, flash, single-shot.
  1 — FULL BEST CONFIG (flash): CC(per-corpus alpha) + routing(top-3) +
      selector(wider-pool top-30, re-synth-on-promotion) + no-rewrite +
      no-rerank.

Usage:
    python scripts/run_headline_combined.py
    python scripts/run_headline_combined.py --skip-index --corpus maud --limit 5
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sqlite3
import struct
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("headline_combined")
logger.setLevel(logging.INFO)

# ── Constants ────────────────────────────────────────────────────────────

FLASH_MODEL = "deepseek-v4-flash"
WIDER_POOL_K = 30
CONTEXT_K = 8
EMBED_MODEL = os.environ.get("MERIDIAN_EMBED_MODEL", "voyage-4")
EMBED_DIM = 1024  # voyage-4

DB_PATH = "data/headline_combined.db"

CORPUS_PARAMS = {
    "contractnli": {
        "parquet": Path("data/corpus_contractnli.parquet"),
        "benchmark": Path("data/benchmarks/contractnli.json"),
        "collection": "contractnli_sac_v4",
        "dataset_name": "contractnli",
        "gt_module": "core.evaluation.ground_truth_contractnli",
        "gt_class": "ContractNLIGroundTruth",
        "corpus_dir": Path("data/corpus"),
        "cc_alpha": 0.2,
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
        "corpus_dir": Path("data/corpus"),
        "cc_alpha": 0.1,
        "routing_alpha": 0.5,
        "routing_topk": 3,
    },
    "cuad": {
        "parquet": Path("data/corpus_cuad.parquet"),
        "benchmark": Path("data/benchmarks/cuad.json"),
        "collection": "cuad_sac_v4",
        "dataset_name": "cuad",
        "gt_module": "core.evaluation.ground_truth_cuad",
        "gt_class": "CUADGroundTruth",
        "corpus_dir": Path("data/corpus"),
        "cc_alpha": 0.1,
        "routing_alpha": 0.3,
        "routing_topk": 3,
    },
    "maud": {
        "parquet": Path("data/corpus_maud.parquet"),
        "benchmark": Path("data/benchmarks/maud.json"),
        "collection": "maud_sac_v4",
        "dataset_name": "maud",
        "gt_module": "core.evaluation.ground_truth_maud",
        "gt_class": "MAUDGroundTruth",
        "corpus_dir": Path("data/corpus"),
        "cc_alpha": 0.2,
        "routing_alpha": 0.7,
        "routing_topk": 3,
    },
}

_RETRY_WAITS = [5, 10, 20]


def _run_with_retry(fn):
    for wait in _RETRY_WAITS:
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


def _sf32(v):
    return struct.pack(f"{len(v)}f", *v)


# ── Step 1: Build combined sqlite-vec index ──────────────────────────────

def build_combined_index():
    """Pull vectors from all 4 Qdrant collections (mini-doc only) into sqlite-vec.

    Spans are resolved from the PARQUET files (Qdrant payloads don't store them).
    """
    import pandas as pd
    import sqlite_vec
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute(f"PRAGMA mmap_size = {3 * 1024 * 1024 * 1024}")
    db.execute(f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{EMBED_DIM}])")

    db.execute("""
        CREATE TABLE chunk_meta (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT,
            doc_id TEXT,
            content TEXT,
            start_idx INTEGER,
            end_idx INTEGER,
            corpus TEXT
        )
    """)

    global_idx = 0
    for corpus, params in CORPUS_PARAMS.items():
        coll = params["collection"]

        # Load parquet for span lookup (spans are NOT in Qdrant payloads)
        corpus_df = pd.read_parquet(params["parquet"])
        span_lookup = {}
        for _, row in corpus_df.iterrows():
            span = tuple(row["start_end_idx"])
            span_lookup[row["chunk_id"]] = (span[0], span[1])

        # Get mini-doc IDs
        bench = json.loads(params["benchmark"].read_text(encoding="utf-8"))
        mini_docs = set()
        for t in bench["tests"]:
            for s in t["snippets"]:
                mini_docs.add(unquote(s["file_path"]))

        logger.info("Pulling %s mini docs from %s (span lookup: %d chunks)...",
                     len(mini_docs), coll, len(span_lookup))

        filt = Filter(must=[
            FieldCondition(key="doc_id", match=MatchAny(any=list(mini_docs)))
        ])

        offset = None
        batch_vectors = []
        batch_meta = []
        missing_spans = 0
        while True:
            points, next_offset = client.scroll(
                collection_name=coll,
                scroll_filter=filt,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break

            for p in points:
                vec = p.vector
                payload = p.payload
                chunk_id = payload.get("chunk_id", "")
                span = span_lookup.get(chunk_id, (0, 0))
                if span == (0, 0):
                    missing_spans += 1

                batch_vectors.append((global_idx, _sf32(vec)))
                batch_meta.append((
                    global_idx,
                    chunk_id,
                    payload.get("doc_id", ""),
                    payload.get("content", ""),
                    span[0],
                    span[1],
                    corpus,
                ))
                global_idx += 1

            offset = next_offset
            if offset is None:
                break

        with db:
            db.executemany("INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)", batch_vectors)
            db.executemany(
                "INSERT INTO chunk_meta(rowid, chunk_id, doc_id, content, start_idx, end_idx, corpus) VALUES (?,?,?,?,?,?,?)",
                batch_meta,
            )
        logger.info("  %s: %d chunks pulled, %d missing spans", corpus, len(batch_vectors), missing_spans)

    db.close()
    logger.info("Combined index: %d total chunks in %s", global_idx, DB_PATH)
    return global_idx


# ── Step 2: Combined retrieval ───────────────────────────────────────────

_thread_local = threading.local()


def _get_db():
    if not hasattr(_thread_local, 'db') or _thread_local.db is None:
        import sqlite_vec
        _thread_local.db = sqlite3.connect(DB_PATH)
        _thread_local.db.enable_load_extension(True)
        sqlite_vec.load(_thread_local.db)
        _thread_local.db.enable_load_extension(False)
    return _thread_local.db


def _embed_query(query: str) -> list[float]:
    """Embed a query with voyage-4."""
    import voyageai
    vc = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    result = vc.embed([query], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]


def dense_retrieve_combined(query: str, top_k: int = 50,
                            doc_ids: list[str] | None = None) -> dict:
    """Dense retrieval from combined sqlite-vec index."""
    db = _get_db()
    query_emb = _embed_query(query)

    # Get top candidates (more than needed, filter by doc_id after)
    fetch_k = top_k * 3 if doc_ids else top_k
    rows = db.execute(
        "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance ASC LIMIT ?",
        [_sf32(query_emb), fetch_k],
    ).fetchall()

    results_ids = []
    results_contents = []
    results_spans = []
    results_scores = []

    for rid, dist in rows:
        meta = db.execute(
            "SELECT chunk_id, doc_id, content, start_idx, end_idx FROM chunk_meta WHERE rowid=?",
            [rid],
        ).fetchone()
        if meta is None:
            continue
        chunk_id, doc_id, content, start_idx, end_idx = meta

        # Filter by doc_ids if routing is active
        if doc_ids and doc_id not in doc_ids:
            continue

        results_ids.append(chunk_id)
        results_contents.append(content)
        results_spans.append((start_idx, end_idx))
        results_scores.append(1.0 - dist)  # cosine similarity = 1 - distance

        if len(results_ids) >= top_k:
            break

    # Return as a FusionResult-like object
    class _Result:
        def __init__(self, ids, contents, spans, scores):
            self.ids = ids
            self.contents = contents
            self.spans = spans
            self.scores = scores
    return _Result(results_ids, results_contents, results_spans, results_scores)


_combined_bm25 = None


def _get_combined_bm25():
    """Build ONE BM25 index from all 4 corpora's mini-doc chunks (channel-matched to dense)."""
    global _combined_bm25
    if _combined_bm25 is not None:
        return _combined_bm25

    import pandas as pd
    from core.retrieval.bm25_retriever import BM25Retriever

    all_dfs = []
    for corpus, params in CORPUS_PARAMS.items():
        df = pd.read_parquet(params["parquet"])

        # Filter to mini-doc chunks — MUST match the dense index's document set
        bench = json.loads(params["benchmark"].read_text(encoding="utf-8"))
        mini_doc_ids = set()
        for t in bench["tests"]:
            for s in t["snippets"]:
                mini_doc_ids.add(unquote(s["file_path"]))
        df = df[df["doc_id"].isin(mini_doc_ids)].reset_index(drop=True)
        logger.info("BM25 %s: %d mini-doc chunks", corpus, len(df))
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info("BM25 combined: %d total chunks (channel-matched to dense index)", len(combined_df))
    _combined_bm25 = BM25Retriever(corpus_df=combined_df)
    return _combined_bm25


def bm25_retrieve_combined(query: str, dataset_name: str,
                           doc_ids: list[str] | None = None) -> object:
    """BM25 retrieval from the COMBINED mini-doc index (all 4 corpora)."""
    bm25 = _get_combined_bm25()
    # Search the full combined BM25 — no dataset_name filter (cross-corpus pool)
    return bm25.retrieve(query, doc_ids=doc_ids)


# ── Synthesis (reuse from run_headline.py) ───────────────────────────────

def synthesize(query, context_ids, context_texts, model=FLASH_MODEL):
    """Structured synthesis call."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "", [], 0, 0, None

    context_string = "\n---\n".join(
        f"[{cid}]\n{text}" for cid, text in zip(context_ids, context_texts)
    )

    system_prompt = (
        "You are a legal document analyst. Answer the query using ONLY the "
        "provided document chunks.\n\n"
        "Rules:\n"
        "- Every factual claim in your answer must cite a specific chunk\n"
        "- cited_chunk_id must be one of the chunk IDs provided in context\n"
        "- cited_text must be copied verbatim from the cited chunk\n"
        "- One atomic fact per claim -- do not bundle multiple facts\n"
        "- Do not invent information not present in the chunks\n"
        "- If the answer cannot be found in the chunks, say so explicitly\n"
        "- Produce at most 10 claims."
    )

    user_prompt = (
        f"Query: {query}\n\n"
        f"Document chunks:\n{context_string}\n\n"
        "Answer the query and provide claim-citation pairs for every "
        "factual assertion."
    )

    try:
        import instructor
        from openai import OpenAI
        from core.supervisor.schemas_phase8 import StructuredAnswer

        client = instructor.from_openai(
            OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        )
        response = client.chat.completions.create(
            model=model,
            response_model=StructuredAnswer,
            max_retries=3,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"thinking": {"type": "disabled"}},
        )
        claims = [c.model_dump() for c in response.claims]
        in_tok = getattr(response, '_raw_response', None)
        input_tokens = 0
        output_tokens = 0
        served_model = None
        if in_tok:
            if hasattr(in_tok, 'usage'):
                input_tokens = getattr(in_tok.usage, 'prompt_tokens', 0) or 0
                output_tokens = getattr(in_tok.usage, 'completion_tokens', 0) or 0
            served_model = getattr(in_tok, 'model', None)
        return response.answer, claims, input_tokens, output_tokens, served_model
    except Exception as exc:
        logger.warning("Synthesis failed (%s): %s", model, exc)
        return "", [], 0, 0, None


# ── Selector (from run_headline.py) ──────────────────────────────────────

_DENIAL_PATTERNS = (
    "not found", "no provision", "no mention", "does not contain",
    "cannot determine", "cannot be determined", "no information",
    "not present", "not included", "no specific", "not specified",
    "does not address", "does not include", "not explicitly",
    "no evidence", "no clause", "not available", "does not specify",
    "there is no", "there are no",
)


def _presence_check(claims, chunk_ids, chunk_texts, answer=""):
    chunk_by_id = dict(zip(chunk_ids, chunk_texts))
    unsupported = []
    seen = set()
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                 "on", "at", "to", "for", "with", "that", "this", "it", "be",
                 "has", "have", "and", "or", "but", "not", "by", "from"}

    for i, claim in enumerate(claims):
        cid = claim.get("cited_chunk_id", "")
        cited_text = claim.get("cited_text", "").strip()
        if cid not in chunk_by_id:
            unsupported.append((i, claim))
            seen.add(i)
            continue
        chunk = chunk_by_id[cid]
        nc = " ".join(cited_text.lower().split())
        nk = " ".join(chunk.lower().split())
        if nc and nc in nk:
            continue
        ct = set(nc.split()) - stopwords
        kt = set(nk.split()) - stopwords
        if ct and len(ct & kt) / len(ct) >= 0.50:
            continue
        unsupported.append((i, claim))
        seen.add(i)

    if answer:
        na = answer.lower()
        if any(p in na for p in _DENIAL_PATTERNS):
            for i, claim in enumerate(claims):
                if i not in seen and any(p in claim.get("claim", "").lower() for p in _DENIAL_PATTERNS):
                    unsupported.append((i, claim))
                    seen.add(i)

    return unsupported


def run_selector_pass(query, answer, claims, top8_ids, top8_texts,
                      wider_ids, wider_texts, wider_spans, model=FLASH_MODEL):
    from pydantic import BaseModel, Field

    class PromotionCandidate(BaseModel):
        claim_index: int = Field(description="0-based index of the unsupported claim")
        promoted_chunk_id: str = Field(default="", description="Chunk to promote")
        reason: str = Field(description="Why this chunk supports the claim")
        promotes: bool = Field(description="True to promote")

    class SelectorOutput(BaseModel):
        promotions: list[PromotionCandidate]

    unsupported = _presence_check(claims, top8_ids, top8_texts, answer=answer)
    if not unsupported:
        return top8_ids, top8_texts, wider_spans[:CONTEXT_K], 0, [], answer, claims, 0

    top8_set = set(top8_ids)
    candidates_str = ""
    for cid, text in zip(wider_ids, wider_texts):
        if cid not in top8_set:
            candidates_str += f"\n[{cid}]\n{text}\n---\n"

    claims_str = ""
    for ci, claim in unsupported:
        claims_str += f"\n[Unsupported Claim {ci}]: {claim['claim']}\n"

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return top8_ids, top8_texts, wider_spans[:CONTEXT_K], 0, [], answer, claims, 0

    import instructor
    from openai import OpenAI

    client = instructor.from_openai(
        OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    )

    try:
        sel_resp = client.chat.completions.create(
            model=FLASH_MODEL,
            response_model=SelectorOutput,
            max_retries=3,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": "You are a legal document retrieval selector. Some claims lack evidence in the top-8 chunks. Check if any candidate chunk (ranks 9-30) supports them."},
                {"role": "user", "content": f"Query: {query}\n\nUnsupported claims:{claims_str}\n\nCandidate chunks:\n{candidates_str}\n\nFor each unsupported claim, promote a candidate or no-op."},
            ],
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception:
        return top8_ids, top8_texts, wider_spans[:CONTEXT_K], 0, [], answer, claims, 0

    new_ids = list(top8_ids)
    new_texts = list(top8_texts)
    new_spans = list(wider_spans[:CONTEXT_K])
    wider_lookup = {cid: (text, span) for cid, text, span in zip(wider_ids, wider_texts, wider_spans)}
    promoted_ids = []
    cited_chunks = {c.get("cited_chunk_id", "") for c in claims}

    for promo in sel_resp.promotions:
        if not promo.promotes or not promo.promoted_chunk_id:
            continue
        pcid = promo.promoted_chunk_id
        if pcid not in wider_lookup or pcid in set(new_ids):
            continue
        evict_idx = len(new_ids) - 1
        for j in range(len(new_ids) - 1, -1, -1):
            if new_ids[j] not in cited_chunks:
                evict_idx = j
                break
        ptext, pspan = wider_lookup[pcid]
        new_ids.pop(evict_idx)
        new_texts.pop(evict_idx)
        new_spans.pop(evict_idx)
        new_ids.insert(1, pcid)
        new_texts.insert(1, ptext)
        new_spans.insert(1, pspan)
        promoted_ids.append(pcid)

    if not promoted_ids:
        return top8_ids, top8_texts, wider_spans[:CONTEXT_K], 0, [], answer, claims, 0

    new_answer, new_claims, in_tok, out_tok, _ = _run_with_retry(
        lambda: synthesize(query, new_ids, new_texts, model=model)
    )
    if not new_answer:
        return top8_ids, top8_texts, wider_spans[:CONTEXT_K], 0, [], answer, claims, 0

    return new_ids, new_texts, new_spans, len(promoted_ids), promoted_ids, new_answer, new_claims, in_tok + out_tok


# ── Per-query runner ─────────────────────────────────────────────────────

def run_query(qid, query, corpus, params, ground_truth):
    """Run both arms for a single query on the combined index."""
    from core.measurement.metrics import compute_all_k
    from core.measurement.taxonomy import classify_with_confidence
    from core.retrieval.fusion import cc_fusion, rrf
    from core.retrieval.routing import route_query

    ds = params["dataset_name"]
    alpha = params["cc_alpha"]
    results = []

    # ── ARM 0: RRF, no routing, no selector ──────────────────────
    t0 = time.perf_counter()
    dense_r = _run_with_retry(lambda: dense_retrieve_combined(query, top_k=50))
    sparse_r = bm25_retrieve_combined(query, dataset_name=ds)
    fused = rrf(sparse_r, dense_r, top_n=CONTEXT_K)

    arm0_answer, arm0_claims, arm0_in, arm0_out, arm0_served = _run_with_retry(
        lambda: synthesize(query, fused.ids, fused.contents, model=FLASH_MODEL)
    )
    arm0_lat = int((time.perf_counter() - t0) * 1000)

    results.append(_build_record(
        qid, query, arm0_answer, arm0_claims, fused.ids, fused.contents, fused.spans,
        ground_truth, routed_docs=None, arm=0, latency_ms=arm0_lat,
        tokens_in=arm0_in, tokens_out=arm0_out, model=FLASH_MODEL,
        n_promoted=0, served_model=arm0_served,
    ))

    # ── ARM 1: CC + routing + selector ───────────────────────────
    t1 = time.perf_counter()
    routed_docs = route_query(query, top_k=params["routing_topk"])

    dense_routed = _run_with_retry(
        lambda: dense_retrieve_combined(query, top_k=50, doc_ids=routed_docs)
    )
    sparse_routed = bm25_retrieve_combined(query, dataset_name=ds, doc_ids=routed_docs)

    fused_cc = cc_fusion(sparse_routed, dense_routed, alpha=alpha, top_n=WIDER_POOL_K)
    wider_ids = fused_cc.ids
    wider_texts = fused_cc.contents
    wider_spans = fused_cc.spans

    top8_ids = wider_ids[:CONTEXT_K]
    top8_texts = wider_texts[:CONTEXT_K]

    arm1_answer, arm1_claims, arm1_in, arm1_out, arm1_served = _run_with_retry(
        lambda: synthesize(query, top8_ids, top8_texts, model=FLASH_MODEL)
    )

    sel_ids, sel_texts, sel_spans, n_prom, prom_ids, sel_answer, sel_claims, sel_tok = \
        run_selector_pass(
            query, arm1_answer, arm1_claims,
            top8_ids, top8_texts,
            wider_ids, wider_texts, wider_spans,
            model=FLASH_MODEL,
        )
    if n_prom > 0:
        arm1_answer = sel_answer
        arm1_claims = sel_claims
        arm1_in += sel_tok

    arm1_lat = int((time.perf_counter() - t1) * 1000)

    # Routing recall
    gt_doc = ground_truth.get_doc_id(qid)
    routing_hit = gt_doc in [str(d) for d in routed_docs] if routed_docs else None

    results.append(_build_record(
        qid, query, arm1_answer, arm1_claims, sel_ids, sel_texts, sel_spans,
        ground_truth, routed_docs=routed_docs, arm=1, latency_ms=arm1_lat,
        tokens_in=arm1_in, tokens_out=arm1_out, model=FLASH_MODEL,
        n_promoted=n_prom, served_model=arm1_served,
    ))

    return results


def _build_record(qid, query, answer, claims, ctx_ids, ctx_texts, ctx_spans,
                  ground_truth, routed_docs, arm, latency_ms,
                  tokens_in, tokens_out, model, n_promoted, served_model=None):
    from core.measurement.metrics import compute_all_k
    from core.measurement.taxonomy import classify_with_confidence

    gt_spans = ground_truth.get_spans(qid)
    gt_doc = ground_truth.get_doc_id(qid)
    spans = [tuple(s) for s in ctx_spans]
    metrics = compute_all_k(spans, gt_spans)

    retrieved_with = [(cid.split("#")[0], s[0], s[1]) for cid, s in zip(ctx_ids, spans)]
    gt_with = [(gt_doc, s[0], s[1]) for s in gt_spans]
    classification = classify_with_confidence(retrieved_with, gt_with)

    context_chunks = [
        {"chunk_id": cid, "doc_id": cid.split("#")[0], "content": text}
        for cid, text in zip(ctx_ids, ctx_texts)
    ]

    routing_hit = None
    if routed_docs is not None:
        routing_hit = gt_doc in [str(d) for d in routed_docs]

    return {
        "query_id": qid,
        "query": query,
        "answer": answer,
        "claims": claims,
        "context_chunks": context_chunks,
        "p_at_1": metrics.p_at_k.get(1, 0.0),
        "r_at_8": metrics.r_at_k.get(8, 0.0),
        "failure_type": classification.failure_type.name,
        "routing_hit": routing_hit,
        "arm": arm,
        "model": model,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "n_promoted": n_promoted,
        "served_model": served_model,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Headline on combined index")
    p.add_argument("--corpus", nargs="*", default=list(CORPUS_PARAMS.keys()),
                   choices=list(CORPUS_PARAMS.keys()))
    p.add_argument("--output-dir", type=Path, default=Path("data/headline_combined"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-index", action="store_true",
                   help="Skip building the combined sqlite-vec index")
    args = p.parse_args()

    print("HEADLINE MEASUREMENT — COMBINED INDEX (benchmark regime)")
    print(f"Embedding: {EMBED_MODEL}")
    print(f"Index: combined sqlite-vec ({DB_PATH})")
    print(f"Config: SAC + CC + routing(top-3) + selector + no-rerank + flash")
    print(f"Corpora: {args.corpus}")
    if args.limit:
        print(f"LIMIT: first {args.limit} per corpus")

    if not args.skip_index:
        n = build_combined_index()
        print(f"\nCombined index: {n} chunks from 4 corpora")

    for corpus in args.corpus:
        params = CORPUS_PARAMS[corpus]

        # Set env for routing
        os.environ["MERIDIAN_EMBED_MODEL"] = EMBED_MODEL
        os.environ["MERIDIAN_NO_REWRITE"] = "1"
        os.environ["MERIDIAN_NO_RERANK"] = "1"
        os.environ["MERIDIAN_CC_ALPHA"] = str(params["cc_alpha"])
        os.environ["MERIDIAN_ROUTING_TOPK"] = str(params["routing_topk"])
        os.environ["MERIDIAN_ROUTING_INDEX"] = str(
            params.get("routing_index", f"data/routing_index_{corpus}_v4.npz")
            if corpus != "contractnli" else "data/routing_index_v4.npz"
        )
        os.environ["MERIDIAN_ROUTING_ALPHA"] = str(params["routing_alpha"])

        # Reset routing cache
        import core.retrieval.routing as routing_mod
        routing_mod._router = None

        # Load ground truth
        mod = importlib.import_module(params["gt_module"])
        gt_cls = getattr(mod, params["gt_class"])
        ground_truth = gt_cls(params["benchmark"], params["corpus_dir"])

        bench = json.loads(params["benchmark"].read_text(encoding="utf-8"))
        questions = [{"query_id": t["query_id"], "query": t["query"]} for t in bench["tests"]]
        if args.limit:
            questions = questions[:args.limit]

        n = len(questions)
        print(f"\n{'=' * 70}")
        print(f"CORPUS: {corpus} | {n} queries x 2 arms on COMBINED index")
        print(f"{'=' * 70}")

        all_records = []
        errors = 0
        write_lock = threading.Lock()

        def _worker(q):
            nonlocal errors
            try:
                records = run_query(q["query_id"], q["query"], corpus, params, ground_truth)
                with write_lock:
                    all_records.extend(records)
                    r0 = records[0]
                    r1 = records[1]
                    print(f"  [{r0['query_id']}] A0:{r0['failure_type']} "
                          f"P@1={r0['p_at_1']:.2f} R@8={r0['r_at_8']:.2f} | "
                          f"A1:routing={'HIT' if r1.get('routing_hit') else 'MISS'} "
                          f"P@1={r1['p_at_1']:.2f} R@8={r1['r_at_8']:.2f}")
            except Exception as exc:
                with write_lock:
                    errors += 1
                    print(f"  [{q['query_id']}] FAILED: {exc}", file=sys.stderr)

        with ThreadPoolExecutor(max_workers=min(args.workers, 12)) as executor:
            futures = {executor.submit(_worker, q): q for q in questions}
            for f in as_completed(futures):
                f.result()

        # Write output
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"headline_{corpus}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in sorted(all_records, key=lambda x: (x["arm"], x["query_id"])):
                f.write(json.dumps(r) + "\n")

        print(f"\n{corpus}: {len(all_records)} records ({errors} errors) -> {out_path}")

        # Per-arm summary
        for arm in [0, 1]:
            arm_recs = [r for r in all_records if r["arm"] == arm]
            if not arm_recs:
                continue
            na = len(arm_recs)
            avg_p1 = sum(r["p_at_1"] for r in arm_recs) / na
            avg_r8 = sum(r["r_at_8"] for r in arm_recs) / na
            avg_lat = sum(r["latency_ms"] for r in arm_recs) / na
            avg_tok = sum(r["tokens_in"] + r["tokens_out"] for r in arm_recs) / na
            ft = Counter(r["failure_type"] for r in arm_recs)
            n_prom = sum(1 for r in arm_recs if r["n_promoted"] > 0)
            routing_hits = sum(1 for r in arm_recs if r.get("routing_hit") is True)
            routing_total = sum(1 for r in arm_recs if r.get("routing_hit") is not None)
            label = "BASELINE(RRF)" if arm == 0 else "BEST(flash)"
            routing_str = f" routing={routing_hits}/{routing_total}" if routing_total else ""
            print(f"  Arm {arm} ({label}): P@1={avg_p1:.3f} R@8={avg_r8:.3f} "
                  f"lat={avg_lat:.0f}ms tok={avg_tok:.0f} prom={n_prom}/{na}{routing_str}")
            print(f"    {dict(ft)}")

    print(f"\n{'=' * 70}")
    print("HEADLINE COMBINED-INDEX RUN COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
