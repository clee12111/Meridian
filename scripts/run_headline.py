"""Headline measurement run — 3 arms x 4 corpora x 194 queries, four axes.

Config-stack delta: CC + routing + selector over RRF baseline, SAC chunking
held constant. NOT the full-system v1->v2 delta.

Arms:
  0 — BASELINE: RRF fusion, no routing, no selector, flash, single-shot.
  1 — FULL BEST CONFIG (flash): CC(per-corpus alpha) + routing(top-3) +
      selector(wider-pool top-30, re-synth-on-promotion) + no-rewrite +
      no-rerank.
  2 — FULL BEST CONFIG (Pro): same as Arm 1 but synthesis on deepseek-chat.

Four axes:
  1. Grounded-correct (holistic judge) — VARIANCE-BAND
  2. Retrieval quality (P@1, R@8) — DETERMINISTIC
  3. Token cost per query — near-deterministic
  4. Latency per query — VARIANCE-BAND

Usage:
    python scripts/run_headline.py
    python scripts/run_headline.py --corpus maud --limit 5
    python scripts/run_headline.py --skip-arm2
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
logger = logging.getLogger("headline")
logger.setLevel(logging.INFO)

# ── Constants ─────────────────────────────────────────────────────────────

WIDER_POOL_K = 30
CONTEXT_K = 8
FLASH_MODEL = "deepseek-v4-flash"
PRO_MODEL = "deepseek-v4-pro"

CORPUS_PARAMS = {
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
}

_RETRY_WAITS = [5, 10, 20]

# Denial patterns for selector trigger
_DENIAL_PATTERNS = (
    "not found", "no provision", "no mention", "does not contain",
    "cannot determine", "cannot be determined", "no information",
    "not present", "not included", "no specific", "not specified",
    "does not address", "does not include", "not explicitly",
    "no evidence", "no clause", "not available", "does not specify",
    "there is no", "there are no",
)


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


# ── Synthesis ─────────────────────────────────────────────────────────────

def synthesize(query, context_ids, context_texts, model=FLASH_MODEL):
    """Structured synthesis call. Returns (answer, claims, input_tokens, output_tokens)."""
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
        # Token usage + served model from instructor's raw response
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


# ── Selector (presence-check trigger + LLM selector) ─────────────────────

def _presence_check(claims, chunk_ids, chunk_texts, answer=""):
    """Dual trigger: unsupported claim OR denial detection."""
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

    # Denial detection
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
    """Single selector pass: check claims, optionally promote from 9-30.

    Returns (new_ids, new_texts, new_spans, n_promoted, promoted_ids,
             new_answer, new_claims, selector_tokens).
    Re-synthesizes ONLY if a promotion happens.
    """
    from pydantic import BaseModel, Field

    class PromotionCandidate(BaseModel):
        claim_index: int = Field(description="0-based index of the unsupported claim")
        promoted_chunk_id: str = Field(default="", description="Chunk to promote, empty if none")
        reason: str = Field(description="Why this chunk supports the claim, or why no chunk does")
        promotes: bool = Field(description="True to promote, False if no chunk found")

    class SelectorOutput(BaseModel):
        promotions: list[PromotionCandidate]

    unsupported = _presence_check(claims, top8_ids, top8_texts, answer=answer)
    if not unsupported:
        return top8_ids, top8_texts, wider_spans[:CONTEXT_K], 0, [], answer, claims, 0

    # Build candidate pool (ranks 9-30, not in top-8)
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

    system_prompt = (
        "You are a legal document retrieval selector. Some claims lack evidence "
        "in the top-8 chunks. Check if any candidate chunk (ranks 9-30) supports "
        "them. If yes, promote it. If no chunk helps, set promotes=false."
    )
    user_prompt = (
        f"Query: {query}\n\nUnsupported claims:{claims_str}\n\n"
        f"Candidate chunks:\n{candidates_str}\n\n"
        "For each unsupported claim, promote a candidate or no-op."
    )

    try:
        sel_resp = client.chat.completions.create(
            model=FLASH_MODEL,
            response_model=SelectorOutput,
            max_retries=3,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception:
        return top8_ids, top8_texts, wider_spans[:CONTEXT_K], 0, [], answer, claims, 0

    # Apply promotions — position 1, evict last, smart eviction
    new_ids = list(top8_ids)
    new_texts = list(top8_texts)
    new_spans = list(wider_spans[:CONTEXT_K])
    wider_lookup = {cid: (text, span) for cid, text, span in
                    zip(wider_ids, wider_texts, wider_spans)}
    promoted_ids = []
    cited_chunks = {c.get("cited_chunk_id", "") for c in claims}

    for promo in sel_resp.promotions:
        if not promo.promotes or not promo.promoted_chunk_id:
            continue
        pcid = promo.promoted_chunk_id
        if pcid not in wider_lookup or pcid in set(new_ids):
            continue

        # Smart eviction: prefer evicting chunk NOT cited by any claim
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

    # Re-synthesize over new top-8 (ONLY on promotion)
    new_answer, new_claims, in_tok, out_tok, _ = _run_with_retry(
        lambda: synthesize(query, new_ids, new_texts, model=model)
    )
    if not new_answer:
        return top8_ids, top8_texts, wider_spans[:CONTEXT_K], 0, [], answer, claims, 0

    return new_ids, new_texts, new_spans, len(promoted_ids), promoted_ids, new_answer, new_claims, in_tok + out_tok


# ── Per-query runner ──────────────────────────────────────────────────────

def run_query_all_arms(qid, query, context, params, ground_truth, run_arm2=True):
    """Run all arms for a single query. Returns list of records."""
    from core.measurement.metrics import compute_all_k
    from core.measurement.taxonomy import classify_with_confidence
    from core.retrieval.fusion import cc_fusion, rrf
    from core.retrieval.routing import route_query

    ds = params["dataset_name"]
    alpha = params["cc_alpha"]
    results = []

    # ── Shared retrieval: dense + sparse (no routing filter) ──────────
    t0_arm0 = time.perf_counter()
    dense_noroute = _run_with_retry(lambda: context.qdrant_retriever.retrieve(
        query, dataset_name=ds))
    sparse_noroute = context.bm25_retriever.retrieve(query, dataset_name=ds)

    # ── ARM 0: RRF, no routing, no selector ───────────────────────────
    fused_rrf = rrf(sparse_noroute, dense_noroute, top_n=CONTEXT_K)
    arm0_ids = fused_rrf.ids
    arm0_texts = fused_rrf.contents
    arm0_spans = fused_rrf.spans

    arm0_answer, arm0_claims, arm0_in, arm0_out, arm0_served = _run_with_retry(
        lambda: synthesize(query, arm0_ids, arm0_texts, model=FLASH_MODEL)
    )
    arm0_latency = int((time.perf_counter() - t0_arm0) * 1000)

    results.append(_build_record(
        qid, query, arm0_answer, arm0_claims, arm0_ids, arm0_texts, arm0_spans,
        ground_truth, routed_docs=None, arm=0, latency_ms=arm0_latency,
        tokens_in=arm0_in, tokens_out=arm0_out, model=FLASH_MODEL,
        n_promoted=0, served_model=arm0_served,
    ))

    # ── Routed retrieval for Arms 1 & 2 ──────────────────────────────
    t0_arm1 = time.perf_counter()
    routed_docs = route_query(query, top_k=params["routing_topk"])
    dense_routed = _run_with_retry(lambda: context.qdrant_retriever.retrieve(
        query, dataset_name=ds, doc_ids=routed_docs))
    sparse_routed = context.bm25_retriever.retrieve(
        query, dataset_name=ds, doc_ids=routed_docs)

    # CC fusion -> wider pool (top-30)
    fused_cc = cc_fusion(sparse_routed, dense_routed, alpha=alpha, top_n=WIDER_POOL_K)
    wider_ids = fused_cc.ids
    wider_texts = fused_cc.contents
    wider_spans = fused_cc.spans

    top8_ids = wider_ids[:CONTEXT_K]
    top8_texts = wider_texts[:CONTEXT_K]

    # ── ARM 1: flash synthesis + selector ─────────────────────────────
    arm1_answer, arm1_claims, arm1_in, arm1_out, arm1_served = _run_with_retry(
        lambda: synthesize(query, top8_ids, top8_texts, model=FLASH_MODEL)
    )

    # Selector pass
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
    final_ids_1 = sel_ids
    final_texts_1 = sel_texts
    final_spans_1 = sel_spans

    arm1_latency = int((time.perf_counter() - t0_arm1) * 1000)

    results.append(_build_record(
        qid, query, arm1_answer, arm1_claims, final_ids_1, final_texts_1, final_spans_1,
        ground_truth, routed_docs=routed_docs, arm=1, latency_ms=arm1_latency,
        tokens_in=arm1_in, tokens_out=arm1_out, model=FLASH_MODEL,
        n_promoted=n_prom, served_model=arm1_served,
    ))

    # ── ARM 2: Pro synthesis + selector (same retrieval) ──────────────
    if run_arm2:
        t0_arm2 = time.perf_counter()
        arm2_answer, arm2_claims, arm2_in, arm2_out, arm2_served = _run_with_retry(
            lambda: synthesize(query, top8_ids, top8_texts, model=PRO_MODEL)
        )

        sel_ids2, sel_texts2, sel_spans2, n_prom2, prom_ids2, sel_answer2, sel_claims2, sel_tok2 = \
            run_selector_pass(
                query, arm2_answer, arm2_claims,
                top8_ids, top8_texts,
                wider_ids, wider_texts, wider_spans,
                model=PRO_MODEL,
            )
        if n_prom2 > 0:
            arm2_answer = sel_answer2
            arm2_claims = sel_claims2
            arm2_in += sel_tok2
        final_ids_2 = sel_ids2
        final_texts_2 = sel_texts2
        final_spans_2 = sel_spans2

        arm2_latency = int((time.perf_counter() - t0_arm2) * 1000)

        results.append(_build_record(
            qid, query, arm2_answer, arm2_claims, final_ids_2, final_texts_2, final_spans_2,
            ground_truth, routed_docs=routed_docs, arm=2, latency_ms=arm2_latency,
            tokens_in=arm2_in, tokens_out=arm2_out, model=PRO_MODEL,
            n_promoted=n_prom2, served_model=arm2_served,
        ))

    return results


def _build_record(qid, query, answer, claims, ctx_ids, ctx_texts, ctx_spans,
                  ground_truth, routed_docs, arm, latency_ms,
                  tokens_in, tokens_out, model, n_promoted, served_model=None):
    """Build judge-compatible output record."""
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


# ── Arm-2-only runner (Pro re-run — skips Arms 0/1 synthesis) ────────────

def run_query_arm2_only(qid, query, context, params, ground_truth):
    """Run only Arm 2 (Pro synthesis + selector) for a single query."""
    from core.retrieval.fusion import cc_fusion
    from core.retrieval.routing import route_query

    ds = params["dataset_name"]
    alpha = params["cc_alpha"]

    t0 = time.perf_counter()
    routed_docs = route_query(query, top_k=params["routing_topk"])
    dense_routed = _run_with_retry(lambda: context.qdrant_retriever.retrieve(
        query, dataset_name=ds, doc_ids=routed_docs))
    sparse_routed = context.bm25_retriever.retrieve(
        query, dataset_name=ds, doc_ids=routed_docs)

    fused_cc = cc_fusion(sparse_routed, dense_routed, alpha=alpha, top_n=WIDER_POOL_K)
    wider_ids = fused_cc.ids
    wider_texts = fused_cc.contents
    wider_spans = fused_cc.spans

    top8_ids = wider_ids[:CONTEXT_K]
    top8_texts = wider_texts[:CONTEXT_K]

    arm2_answer, arm2_claims, arm2_in, arm2_out, arm2_served = _run_with_retry(
        lambda: synthesize(query, top8_ids, top8_texts, model=PRO_MODEL)
    )

    sel_ids, sel_texts, sel_spans, n_prom, prom_ids, sel_answer, sel_claims, sel_tok = \
        run_selector_pass(
            query, arm2_answer, arm2_claims,
            top8_ids, top8_texts,
            wider_ids, wider_texts, wider_spans,
            model=PRO_MODEL,
        )
    if n_prom > 0:
        arm2_answer = sel_answer
        arm2_claims = sel_claims
        arm2_in += sel_tok
    final_ids = sel_ids
    final_texts = sel_texts
    final_spans = sel_spans

    arm2_latency = int((time.perf_counter() - t0) * 1000)

    return _build_record(
        qid, query, arm2_answer, arm2_claims, final_ids, final_texts, final_spans,
        ground_truth, routed_docs=routed_docs, arm=2, latency_ms=arm2_latency,
        tokens_in=arm2_in, tokens_out=arm2_out, model=PRO_MODEL,
        n_promoted=n_prom, served_model=arm2_served,
    )


# ── Per-corpus runner ─────────────────────────────────────────────────────

def run_corpus(corpus, output_dir, workers=8, limit=None, run_arm2=True, only_arm2=False):
    """Run all arms on all queries for one corpus."""
    params = CORPUS_PARAMS[corpus]

    os.environ["MERIDIAN_EMBED_MODEL"] = "voyage-4"
    os.environ["MERIDIAN_NO_REWRITE"] = "1"
    os.environ["MERIDIAN_NO_RERANK"] = "1"
    os.environ["MERIDIAN_CC_ALPHA"] = str(params["cc_alpha"])
    os.environ["MERIDIAN_ROUTING_TOPK"] = str(params["routing_topk"])
    os.environ["MERIDIAN_ROUTING_INDEX"] = params["routing_index"]
    os.environ["MERIDIAN_ROUTING_ALPHA"] = str(params["routing_alpha"])

    import core.retrieval.routing as routing_mod
    routing_mod._router = None

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

    data = json.loads(params["benchmark"].read_text(encoding="utf-8"))
    questions = [{"query_id": t["query_id"], "query": t["query"]} for t in data["tests"]]
    if limit:
        questions = questions[:limit]

    n = len(questions)
    if only_arm2:
        n_arms = 1
        print(f"\n{'=' * 70}")
        print(f"CORPUS: {corpus} | {n} queries x Arm 2 ONLY (Pro: {PRO_MODEL})")
        print(f"{'=' * 70}")
    else:
        n_arms = 3 if run_arm2 else 2
        print(f"\n{'=' * 70}")
        print(f"CORPUS: {corpus} | {n} queries x {n_arms} arms = {n * n_arms} runs")
        print(f"{'=' * 70}")

    all_records = []
    errors = 0
    write_lock = threading.Lock()

    def _worker(q):
        nonlocal errors
        try:
            if only_arm2:
                record = run_query_arm2_only(
                    q["query_id"], q["query"], context, params, ground_truth)
                records = [record]
            else:
                records = run_query_all_arms(
                    q["query_id"], q["query"], context, params, ground_truth,
                    run_arm2=run_arm2,
                )
            with write_lock:
                all_records.extend(records)
                r0 = records[0]
                tag = f"A2-Pro" if only_arm2 else f"A0:{r0['failure_type']}"
                print(f"  [{r0['query_id']}] {tag} "
                      f"P@1={r0['p_at_1']:.2f} R@8={r0['r_at_8']:.2f}"
                      f"{' served=' + r0.get('served_model', '?') if only_arm2 else ''}")
        except Exception as exc:
            with write_lock:
                errors += 1
                print(f"  [{q['query_id']}] FAILED: {exc}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=min(workers, 24)) as executor:
        futures = {executor.submit(_worker, q): q for q in questions}
        for f in as_completed(futures):
            f.result()

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"headline_{corpus}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in sorted(all_records, key=lambda x: (x["arm"], x["query_id"])):
            f.write(json.dumps(r) + "\n")

    print(f"\n{corpus}: {len(all_records)} records ({errors} errors) -> {out_path}")

    # Per-arm summary
    arms_to_show = [2] if only_arm2 else list(range(n_arms))
    for arm in arms_to_show:
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
        label = {0: "BASELINE(RRF)", 1: "BEST(flash)", 2: "BEST(Pro)"}[arm]
        print(f"  Arm {arm} ({label}): P@1={avg_p1:.3f} R@8={avg_r8:.3f} "
              f"lat={avg_lat:.0f}ms tok={avg_tok:.0f} prom={n_prom}/{na}")
        print(f"    {dict(ft)}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Headline measurement run")
    p.add_argument("--corpus", nargs="*", default=list(CORPUS_PARAMS.keys()),
                   choices=list(CORPUS_PARAMS.keys()))
    p.add_argument("--output-dir", type=Path, default=Path("data/headline"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-arm2", action="store_true",
                   help="Skip Arm 2 (Pro synthesis)")
    p.add_argument("--only-arm2", action="store_true",
                   help="Run ONLY Arm 2 (Pro synthesis), skip Arms 0/1")
    args = p.parse_args()

    run_arm2 = not args.skip_arm2
    only_arm2 = args.only_arm2
    if only_arm2:
        print("HEADLINE PRO RE-RUN (Arm 2 only)")
        print(f"Pro model: {PRO_MODEL}")
        print(f"Corpora: {args.corpus}")
        print(f"Workers: {args.workers}")
        print(f"Embedding: voyage-4 (validated, not voyage-4-large)")
    else:
        print("HEADLINE MEASUREMENT RUN")
        print(f"Config-stack delta: CC + routing + selector over RRF baseline")
        print(f"SAC chunking HELD CONSTANT (not a full-system delta)")
        print(f"Corpora: {args.corpus}")
        print(f"Arms: 0=RRF-baseline, 1=best-flash, {'2=best-Pro' if run_arm2 else '(arm2 skipped)'}")
        print(f"Embedding: voyage-4 (validated, not voyage-4-large)")
    if args.limit:
        print(f"LIMIT: first {args.limit} per corpus")

    for corpus in args.corpus:
        run_corpus(corpus, args.output_dir, workers=args.workers,
                   limit=args.limit, run_arm2=run_arm2, only_arm2=only_arm2)

    print(f"\n{'=' * 70}")
    print("HEADLINE RUN COMPLETE")
    print(f"{'=' * 70}")
    print("\nNext: run judges (answer + faithfulness) per arm per corpus,")
    print("then produce four-axis report with variance bands.")
    print("\nREPORTING DISCIPLINE:")
    print("  - This is the CONFIG-STACK delta (CC+routing+selector over RRF)")
    print("  - NOT the full-system v1->v2 delta")
    print("  - Do NOT quote alongside the 25.8%->75.3% full-system number")
    print("  - SAC contribution cited separately from prior findings")


if __name__ == "__main__":
    main()
