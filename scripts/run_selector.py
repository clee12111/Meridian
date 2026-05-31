"""LLM-guided chunk re-selection from wider pool — conditional, single-iteration.

Fires ONLY when synthesis claims are unsupported by the top-8. For the ~88%
that pass, zero added cost. For the ~10% that fail, one selector LLM call
over ranks 9-30 (already retrieved), then one re-synthesis.

NOT the rejected loop (Finding 37): no re-retrieval, scoped to routed docs,
one iteration max. The selector reasons about relevance the cross-encoder
couldn't (vocabulary mismatch: "regulatory approvals" vs "cooperation/
reasonable best efforts").

Flow:
  1. Retrieve wider pool (top-30) within routed docs -> take top-8 -> synthesize -> verify
  2. TRIGGER: presence check — is any claim unsupported by the top-8?
     (NOT Phase-9 contradiction detector — confounded on MAUD negation)
  3. IF triggered: LLM-selector over ranks 9-30 — "does any chunk better support
     the unsupported claim?" Promote or no-op ("none of these answer it").
  4. IF promoted: re-synthesize ONCE over the new top-8. Single iteration only.

Usage:
    python scripts/run_selector.py
    python scripts/run_selector.py --corpus maud --limit 3
    python scripts/run_selector.py --corpus cuad privacyqa maud
"""

from __future__ import annotations

import argparse
import copy
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
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("selector")
logger.setLevel(logging.INFO)

# ── Constants ─────────────────────────────────────────────────────────────

WIDER_POOL_K = 30         # retrieve top-30 candidates
CONTEXT_K = 8             # synthesize over top-8
FAITH_THRESHOLD = 0.8     # INCORRECT+FAITHFUL population filter

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


# ── Pydantic models ──────────────────────────────────────────────────────

class PromotionCandidate(BaseModel):
    claim_index: int = Field(description="0-based index of the unsupported claim")
    promoted_chunk_id: str = Field(
        default="",
        description="Chunk ID from ranks 9-30 to promote. Empty string if no chunk supports the claim."
    )
    reason: str = Field(description="Why this chunk supports the claim, or why no chunk does")
    promotes: bool = Field(description="True if a chunk should be promoted, False if none found")


class SelectorOutput(BaseModel):
    promotions: list[PromotionCandidate] = Field(
        description="One entry per unsupported claim. promotes=False means "
                    "'no chunk in the wider pool answers this' (the regression guard)."
    )


# ── Population loading ────────────────────────────────────────────────────

def _load_jsonl(path):
    results = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                results[r["query_id"]] = r
    return results


def _load_correct_sample(corpus, n):
    """Load a random sample of CORRECT queries for regression testing."""
    import hashlib
    judges = _load_jsonl(JUDGE_FILES[corpus])
    evals = _load_jsonl(EVAL_FILES[corpus])
    correct_ids = [qid for qid, j in judges.items()
                   if j.get("verdict") == "CORRECT" and qid in evals]
    # Deterministic sample via hash
    def sort_key(qid):
        return hashlib.sha256(f"correct-sample:{qid}".encode()).hexdigest()
    correct_ids = sorted(correct_ids, key=sort_key)[:n]
    return [evals[qid] for qid in correct_ids]


def load_population(corpus):
    """Load INCORRECT+FAITHFUL queries."""
    judges = _load_jsonl(JUDGE_FILES[corpus])
    faiths = _load_jsonl(FAITH_FILES[corpus])
    evals = _load_jsonl(EVAL_FILES[corpus])

    population = []
    for qid, j in judges.items():
        if j.get("verdict") != "INCORRECT":
            continue
        if faiths.get(qid, {}).get("faithfulness_score", 0) < FAITH_THRESHOLD:
            continue
        if qid not in evals:
            continue
        rec = evals[qid]
        rec["_judge_reason"] = j.get("reason", "")
        rec["_failure_type"] = j.get("failure_type", rec.get("failure_type", "?"))
        population.append(rec)

    return sorted(population, key=lambda r: r["query_id"])


# ── Presence-check trigger (NOT Phase-9 contradiction detector) ──────────

_DENIAL_PATTERNS = (
    "not found", "no provision", "no mention", "does not contain",
    "cannot determine", "cannot be determined", "no information",
    "not present", "not included", "no specific", "not specified",
    "does not address", "does not include", "not explicitly",
    "no evidence", "no clause", "not available", "does not specify",
    "there is no", "there are no",
)


def _presence_check(claims, chunk_ids, chunk_texts, answer=""):
    """Check which claims lack supporting evidence OR trigger on denial.

    Two trigger conditions (either fires):
    1. UNSUPPORTED CLAIM: cited_text has <50% token overlap with its cited
       chunk, or cited chunk_id not in context.
    2. DENIAL DETECTION: the answer contains denial language ("not found",
       "no provision", etc.) -- the signal that evidence might exist in
       ranks 9-30 below the top-8 cutoff. NOTE: this trigger is safe on
       AFFIRMATIVE-ONLY benchmarks (denial = error). On a corpus with
       legitimate "no" answers it would fire on correct denials -- it is
       a benchmark-specific trigger, scoped accordingly.

    Returns (unsupported_list, trigger_reasons) where trigger_reasons is
    a dict with counts of each trigger type.
    """
    chunk_by_id = dict(zip(chunk_ids, chunk_texts))
    unsupported = []
    unsupported_indices = set()
    n_citation_unsupported = 0
    n_denial_triggered = 0

    # Check 1: per-claim citation support
    for i, claim in enumerate(claims):
        cid = claim.get("cited_chunk_id", "")
        cited_text = claim.get("cited_text", "").strip()

        if cid not in chunk_by_id:
            unsupported.append((i, claim))
            unsupported_indices.add(i)
            n_citation_unsupported += 1
            continue

        chunk = chunk_by_id[cid]
        norm_cited = " ".join(cited_text.lower().split())
        norm_chunk = " ".join(chunk.lower().split())
        if norm_cited and norm_cited in norm_chunk:
            continue

        stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                     "on", "at", "to", "for", "with", "that", "this", "it", "be",
                     "has", "have", "and", "or", "but", "not", "by", "from"}
        cited_tokens = set(norm_cited.split()) - stopwords
        chunk_tokens = set(norm_chunk.split()) - stopwords
        if not cited_tokens:
            continue
        overlap = len(cited_tokens & chunk_tokens) / len(cited_tokens)
        if overlap >= 0.50:
            continue

        unsupported.append((i, claim))
        unsupported_indices.add(i)
        n_citation_unsupported += 1

    # Check 2: denial detection on the answer text
    if answer:
        norm_answer = answer.lower()
        has_denial = any(pattern in norm_answer for pattern in _DENIAL_PATTERNS)
        if has_denial:
            for i, claim in enumerate(claims):
                if i not in unsupported_indices:
                    claim_text = claim.get("claim", "").lower()
                    claim_denies = any(p in claim_text for p in _DENIAL_PATTERNS)
                    if claim_denies:
                        unsupported.append((i, claim))
                        unsupported_indices.add(i)
                        n_denial_triggered += 1

    trigger_reasons = {
        "citation_unsupported": n_citation_unsupported,
        "denial_triggered": n_denial_triggered,
    }
    return unsupported, trigger_reasons


# ── LLM selector ─────────────────────────────────────────────────────────

def run_selector(
    query: str,
    answer: str,
    unsupported_claims: list[tuple[int, dict]],
    wider_pool_ids: list[str],
    wider_pool_texts: list[str],
    current_top8_ids: set[str],
) -> SelectorOutput:
    """LLM-guided selection: for each unsupported claim, find the best
    supporting chunk from ranks 9-30 (the wider pool minus the current top-8).

    The selector MUST be allowed to return promotes=False ("no chunk in
    the wider pool supports this claim"). Forcing a pick over a pool that
    lacks the answer introduces errors.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    # Build the candidate pool: chunks NOT in the current top-8
    candidates_str = ""
    n_candidates = 0
    for cid, text in zip(wider_pool_ids, wider_pool_texts):
        if cid in current_top8_ids:
            continue
        candidates_str += f"\n[{cid}]\n{text}\n---\n"
        n_candidates += 1

    if n_candidates == 0:
        return SelectorOutput(promotions=[
            PromotionCandidate(claim_index=ci, promoted_chunk_id="",
                              reason="No candidates outside top-8", promotes=False)
            for ci, _ in unsupported_claims
        ])

    claims_str = ""
    for ci, claim in unsupported_claims:
        claims_str += (
            f"\n[Unsupported Claim {ci}]\n"
            f"  Assertion: {claim['claim']}\n"
            f"  Currently cites: {claim.get('cited_chunk_id', '')}\n"
        )

    system_prompt = (
        "You are a legal document retrieval selector. The system synthesized "
        "an answer but some claims lack supporting evidence in the top-8 chunks. "
        "You have access to additional chunks (ranks 9-30 from the same "
        "retrieval, same documents) that were retrieved but not used.\n\n"
        "For each unsupported claim, check: does ANY chunk in the candidate "
        "pool below contain evidence that supports or addresses the claim?\n\n"
        "CRITICAL RULES:\n"
        "- If a candidate chunk DOES support the claim, set promotes=true and "
        "provide its chunk_id.\n"
        "- If NO candidate chunk supports the claim, set promotes=false. Do NOT "
        "force a pick -- promoting a wrong chunk introduces errors.\n"
        "- Match on SUBSTANCE, not surface keywords. A chunk about 'reasonable "
        "best efforts to obtain regulatory approvals' supports a claim about "
        "'regulatory approval conditions' even if the exact phrase differs.\n"
        "- The chunk must contain actual evidence, not just related terminology.\n"
        "- One promotion per unsupported claim maximum."
    )

    user_prompt = (
        f"Query: {query}\n\n"
        f"Current answer: {answer[:500]}\n\n"
        f"Unsupported claims:{claims_str}\n\n"
        f"Candidate chunks (ranks 9-30, not in current top-8):\n{candidates_str}\n\n"
        "For each unsupported claim, decide: promote a candidate chunk or no-op."
    )

    import instructor
    from openai import OpenAI

    client = instructor.from_openai(
        OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        response_model=SelectorOutput,
        max_retries=3,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )

    return response


# ── Re-synthesis (constrained by evidence) ────────────────────────────────

def resynthesize(query, context_ids, context_texts):
    """Single synthesis call over the (potentially updated) top-8 context."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "", []

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

        claims = [c.model_dump() for c in response.claims]
        return response.answer, claims
    except Exception as exc:
        logger.warning("Re-synthesis failed: %s", exc)
        return "", []


# ── Per-query runner ──────────────────────────────────────────────────────

def process_query(query_text, qid, context, params, ground_truth, unconditional=False):
    """Full flow: wider retrieval -> top-8 synth -> trigger check ->
    conditional selector -> optional re-synth.

    If unconditional=True, skip the trigger and fire the selector on ALL
    claims (mechanism ceiling test — not deployable, uses post-hoc knowledge).
    """
    from core.measurement.metrics import compute_all_k
    from core.measurement.taxonomy import classify_with_confidence
    from core.retrieval.fusion import cc_fusion
    from core.retrieval.routing import route_query

    t0 = time.perf_counter()
    ds = params["dataset_name"]
    alpha = params["cc_alpha"]

    # ── Step 1: Retrieve wider pool (top-30) within routed docs ───────
    routed_docs = route_query(query_text, top_k=params["routing_topk"])
    dense = context.qdrant_retriever.retrieve(query_text, dataset_name=ds, doc_ids=routed_docs)
    sparse = context.bm25_retriever.retrieve(query_text, dataset_name=ds, doc_ids=routed_docs)
    fused = cc_fusion(sparse, dense, alpha=alpha, top_n=WIDER_POOL_K)

    wider_ids = fused.ids
    wider_texts = fused.contents
    wider_spans = fused.spans

    # Top-8 for initial synthesis
    top8_ids = wider_ids[:CONTEXT_K]
    top8_texts = wider_texts[:CONTEXT_K]
    top8_spans = wider_spans[:CONTEXT_K]

    # ── Step 2: Synthesize over top-8 ─────────────────────────────────
    answer, claims = _run_with_retry(lambda: resynthesize(query_text, top8_ids, top8_texts))
    if not answer:
        latency = int((time.perf_counter() - t0) * 1000)
        return _build_record(qid, query_text, answer, claims, top8_ids, top8_texts,
                            top8_spans, wider_ids, ground_truth, routed_docs,
                            triggered=False, n_promoted=0, latency_ms=latency,
                            original_answer=answer)

    # ── Step 3: Trigger check ────────────────────────────────────────────
    if unconditional:
        # Unconditional mode: treat ALL claims as candidates for selector
        unsupported = [(i, claim) for i, claim in enumerate(claims)]
        trigger_reasons = {"citation_unsupported": 0, "denial_triggered": 0, "unconditional": len(claims)}
    else:
        unsupported, trigger_reasons = _presence_check(claims, top8_ids, top8_texts, answer=answer)

    if not unsupported:
        # All claims supported, no denial -> SHIP, zero selector cost
        latency = int((time.perf_counter() - t0) * 1000)
        return _build_record(qid, query_text, answer, claims, top8_ids, top8_texts,
                            top8_spans, wider_ids, ground_truth, routed_docs,
                            triggered=False, n_promoted=0, latency_ms=latency,
                            original_answer=answer, trigger_reasons=trigger_reasons)

    logger.info("[%s] TRIGGERED: %d unsupported claims (citation=%d, denial=%d), invoking selector",
                qid, len(unsupported),
                trigger_reasons["citation_unsupported"], trigger_reasons["denial_triggered"])

    # ── Step 4: LLM selector over ranks 9-30 ─────────────────────────
    top8_set = set(top8_ids)
    selector_output = _run_with_retry(lambda: run_selector(
        query_text, answer, unsupported,
        wider_ids, wider_texts, top8_set,
    ))

    # Apply promotions — insert at POSITION 1 (top-weighted for synthesis
    # attention), evict the last chunk (lowest retrieval rank).
    promoted_ids = []
    evicted_ids = []
    n_promoted = 0
    new_top8_ids = list(top8_ids)
    new_top8_texts = list(top8_texts)
    new_top8_spans = list(top8_spans)

    wider_lookup = {cid: (text, span) for cid, text, span in
                    zip(wider_ids, wider_texts, wider_spans)}

    for promo in selector_output.promotions:
        if not promo.promotes or not promo.promoted_chunk_id:
            logger.info("[%s]   claim %d: no promotion (reason: %s)",
                       qid, promo.claim_index, promo.reason[:80])
            continue

        pcid = promo.promoted_chunk_id
        if pcid not in wider_lookup:
            logger.warning("[%s]   claim %d: promoted chunk %s not in wider pool, skipping",
                          qid, promo.claim_index, pcid)
            continue

        if pcid in set(new_top8_ids):
            logger.info("[%s]   claim %d: chunk %s already in top-8",
                       qid, promo.claim_index, pcid[-30:])
            continue

        # Evict last chunk (lowest retrieval rank), insert promoted at position 1
        ptext, pspan = wider_lookup[pcid]
        evicted = new_top8_ids.pop()  # remove last
        new_top8_texts.pop()
        new_top8_spans.pop()
        evicted_ids.append(evicted)
        # Insert at position 1 (after position 0, which is the highest-ranked original)
        new_top8_ids.insert(1, pcid)
        new_top8_texts.insert(1, ptext)
        new_top8_spans.insert(1, pspan)
        promoted_ids.append(pcid)
        n_promoted += 1
        logger.info("[%s]   claim %d: PROMOTED %s to pos 1 (evicted %s)",
                   qid, promo.claim_index, pcid[-40:], evicted[-40:])

    # ── Eviction regression check: did evicting a chunk break a
    #    previously-supported claim? ───────────────────────────────
    eviction_regressions = 0
    if evicted_ids and claims:
        evicted_set = set(evicted_ids)
        for claim in claims:
            cid = claim.get("cited_chunk_id", "")
            if cid in evicted_set and cid not in set(new_top8_ids):
                eviction_regressions += 1
                logger.warning("[%s]   EVICTION REGRESSION: claim cited %s which was evicted",
                             qid, cid[-40:])

    if n_promoted == 0:
        # Selector said "no chunk helps" for all unsupported claims -> no-op
        latency = int((time.perf_counter() - t0) * 1000)
        return _build_record(qid, query_text, answer, claims, top8_ids, top8_texts,
                            top8_spans, wider_ids, ground_truth, routed_docs,
                            triggered=True, n_promoted=0, latency_ms=latency,
                            original_answer=answer,
                            selector_details=[p.model_dump() for p in selector_output.promotions],
                            trigger_reasons=trigger_reasons)

    # ── Step 5: Re-synthesize over the new top-8 ─────────────────────
    logger.info("[%s]   Re-synthesizing over updated top-8 (%d promotions, "
                "%d eviction regressions)", qid, n_promoted, eviction_regressions)
    new_answer, new_claims = _run_with_retry(
        lambda: resynthesize(query_text, new_top8_ids, new_top8_texts)
    )
    if not new_answer:
        new_answer = answer
        new_claims = claims

    latency = int((time.perf_counter() - t0) * 1000)
    return _build_record(qid, query_text, new_answer, new_claims,
                        new_top8_ids, new_top8_texts, new_top8_spans,
                        wider_ids, ground_truth, routed_docs,
                        triggered=True, n_promoted=n_promoted, latency_ms=latency,
                        original_answer=answer, promoted_chunk_ids=promoted_ids,
                        evicted_chunk_ids=evicted_ids,
                        eviction_regressions=eviction_regressions,
                        selector_details=[p.model_dump() for p in selector_output.promotions],
                        trigger_reasons=trigger_reasons)


def _build_record(qid, query, answer, claims, ctx_ids, ctx_texts, ctx_spans,
                  wider_ids, ground_truth, routed_docs,
                  triggered, n_promoted, latency_ms, original_answer,
                  promoted_chunk_ids=None, evicted_chunk_ids=None,
                  eviction_regressions=0, selector_details=None,
                  trigger_reasons=None):
    """Build judge-compatible output record with diagnostics."""
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

    return {
        "query_id": qid,
        "query": query,
        "answer": answer,
        "claims": claims,
        "context_chunks": context_chunks,
        "p_at_1": metrics.p_at_k.get(1, 0.0),
        "r_at_8": metrics.r_at_k.get(8, 0.0),
        "failure_type": classification.failure_type.name,
        "routing_hit": (gt_doc in [str(d) for d in routed_docs]) if routed_docs else None,
        # diagnostics
        "triggered": triggered,
        "n_promoted": n_promoted,
        "promoted_chunk_ids": promoted_chunk_ids or [],
        "evicted_chunk_ids": evicted_chunk_ids or [],
        "eviction_regressions": eviction_regressions,
        "original_answer": original_answer,
        "latency_ms": latency_ms,
        "wider_pool_size": len(wider_ids),
        "selector_details": selector_details or [],
        "trigger_reasons": trigger_reasons or {"citation_unsupported": 0, "denial_triggered": 0},
    }


# ── Per-corpus runner ─────────────────────────────────────────────────────

def run_corpus(corpus, output_dir, workers=4, limit=None, unconditional=False,
               correct_sample=0):
    """Run selector experiment on INCORRECT+FAITHFUL queries for one corpus.

    If unconditional=True, skip the trigger and fire the selector on all claims.
    If correct_sample>0, also run on N randomly-sampled CORRECT queries to
    measure regression risk on the good population.
    """
    params = CORPUS_PARAMS[corpus]

    # Set env vars
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

    population = load_population(corpus)
    if limit:
        population = population[:limit]

    # Also load benchmark for query text (population has it from eval)
    n = len(population)
    print(f"\n{'=' * 60}")
    print(f"CORPUS: {corpus} | {n} INCORRECT+FAITHFUL queries")
    print(f"{'=' * 60}")

    if n == 0:
        print("  No qualifying queries found.")
        return

    results = []
    errors = 0
    write_lock = threading.Lock()

    def _worker(rec, pop_label="incorrect"):
        nonlocal errors
        try:
            result = process_query(
                rec["query"], rec["query_id"], context, params, ground_truth,
                unconditional=unconditional,
            )
            result["_population"] = pop_label
            with write_lock:
                results.append(result)
                tag = "PROMOTED" if result["n_promoted"] > 0 else (
                    "TRIGGERED" if result["triggered"] else "PASS")
                print(f"  [{result['query_id']}] {result['failure_type']} {tag} "
                      f"P@1={result['p_at_1']:.2f} R@8={result['r_at_8']:.2f} "
                      f"promoted={result['n_promoted']} [{pop_label}]")
        except Exception as exc:
            with write_lock:
                errors += 1
                print(f"  [{rec['query_id']}] FAILED: {exc}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
        futures = {executor.submit(_worker, rec, "incorrect"): rec for rec in population}
        for f in as_completed(futures):
            f.result()

    # Correct-sample regression check (if requested)
    if correct_sample > 0:
        correct_pop = _load_correct_sample(corpus, correct_sample)
        if correct_pop:
            print(f"\n  --- CORRECT-SAMPLE REGRESSION CHECK ({len(correct_pop)} queries) ---")
            with ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
                futures = {executor.submit(_worker, rec, "correct"): rec for rec in correct_pop}
                for f in as_completed(futures):
                    f.result()

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_unconditional" if unconditional else ""
    out_path = output_dir / f"{corpus}_selector{suffix}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda x: x["query_id"]):
            f.write(json.dumps(r) + "\n")

    # Split results by population
    incorrect_results = [r for r in results if r.get("_population") == "incorrect"]
    correct_results = [r for r in results if r.get("_population") == "correct"]

    print(f"\n{corpus}: {len(incorrect_results)} incorrect + {len(correct_results)} correct "
          f"({errors} errors) -> {out_path}")

    # Summary for incorrect population
    ir = incorrect_results
    if ir:
        n_triggered = sum(1 for r in ir if r["triggered"])
        n_promoted = sum(1 for r in ir if r["n_promoted"] > 0)
        n_pass = sum(1 for r in ir if not r["triggered"])
        ft = Counter(r["failure_type"] for r in ir)
        avg_r8 = sum(r["r_at_8"] for r in ir) / len(ir)

        print(f"\n  INCORRECT POPULATION ({len(ir)} queries):")
        print(f"    Triggered: {n_triggered}/{len(ir)} "
              f"({n_triggered/len(ir)*100:.0f}%)")
        print(f"    Promoted: {n_promoted}/{len(ir)}")
        print(f"    No-op: {n_pass}/{len(ir)}")
        print(f"    Mean R@8: {avg_r8:.3f}")
        print(f"    Failure types: {dict(ft)}")

        # Eviction regression audit
        total_eviction_reg = sum(r.get("eviction_regressions", 0) for r in ir)
        if n_promoted > 0:
            queries_with_reg = sum(1 for r in ir if r.get("eviction_regressions", 0) > 0)
            print(f"    Eviction regressions: {total_eviction_reg} across {queries_with_reg} queries")

    # Correct-sample regression report
    if correct_results:
        cr = correct_results
        n_cr_triggered = sum(1 for r in cr if r["triggered"])
        n_cr_promoted = sum(1 for r in cr if r["n_promoted"] > 0)
        print(f"\n  CORRECT-SAMPLE REGRESSION ({len(cr)} queries):")
        print(f"    Triggered: {n_cr_triggered}/{len(cr)}")
        print(f"    Promoted (risk of regression): {n_cr_promoted}/{len(cr)}")
        cr_evict = sum(r.get("eviction_regressions", 0) for r in cr)
        if cr_evict > 0:
            print(f"    Eviction regressions: {cr_evict}")
        if n_cr_promoted > 0:
            print(f"    PROMOTED CORRECT QUERIES (check for regression after judging):")
            for r in cr:
                if r["n_promoted"] > 0:
                    print(f"      {r['query_id']}: promoted={r['n_promoted']} "
                          f"R@8={r['r_at_8']:.2f} evict_reg={r.get('eviction_regressions', 0)}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="LLM-guided chunk re-selection experiment")
    p.add_argument("--corpus", nargs="*", default=list(CORPUS_PARAMS.keys()),
                   choices=list(CORPUS_PARAMS.keys()))
    p.add_argument("--output-dir", type=Path, default=Path("data/selector"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--unconditional", action="store_true",
                   help="Skip trigger, fire selector on ALL claims (ceiling test)")
    p.add_argument("--correct-sample", type=int, default=0,
                   help="Also run on N CORRECT queries to measure regression risk")
    args = p.parse_args()

    mode = "UNCONDITIONAL" if args.unconditional else "CONDITIONAL"
    print(f"LLM-Guided Chunk Re-Selection Experiment ({mode})")
    print(f"Corpora: {args.corpus}")
    print(f"Wider pool: top-{WIDER_POOL_K}, context: top-{CONTEXT_K}")
    if args.limit:
        print(f"LIMIT: first {args.limit} per corpus")
    if args.correct_sample > 0:
        print(f"CORRECT-SAMPLE: {args.correct_sample} per corpus (regression check)")

    for corpus in args.corpus:
        run_corpus(corpus, args.output_dir, workers=args.workers, limit=args.limit,
                   unconditional=args.unconditional, correct_sample=args.correct_sample)

    print("\nNext steps:")
    print("  1. Run judge_answers_v2.py on selector output per corpus")
    print("  2. Run judge_faithfulness.py (holistic) per corpus")
    print("  3. Compare vs baseline: by-name flips, per-corpus, faith guard")
    print("  4. Report CUAD/PQA/MAUD (selection residual) vs ContractNLI (routing)")


if __name__ == "__main__":
    main()
