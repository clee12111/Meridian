"""Mini E2E — 3 arms × 4 corpora × 50 fixed queries.

Tests whether the redesigned loop (Finding 35) beats single-shot, and
whether an LLM critic gate adds value over a deterministic gate.

Arms:
  0 — SINGLE-SHOT (control): current best config, no loop.
  1 — DETERMINISTIC GATE: freeze passed claims, templated re-query
      within routed top-3 docs, accumulate chunks, patch-synthesize
      failed claims, final re-synthesis.
  2 — LLM CRITIC GATE: same mechanism as Arm 1 but LLM reasons about
      what's missing and generates a targeted re-query.

Usage:
    python scripts/run_mini_e2e.py
    python scripts/run_mini_e2e.py --corpus maud --n-queries 5
    python scripts/run_mini_e2e.py --corpus contractnli cuad --workers 4
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
logger = logging.getLogger("mini_e2e")
logger.setLevel(logging.INFO)

# ── Constants ─────────────────────────────────────────────────────────────

GATE_THRESHOLD = 0.75
MAX_ITERATIONS = 3        # 1 initial + up to 2 loop iterations
N_QUERIES_DEFAULT = 50
QUERY_SEED = "mini-e2e-v1"
CONTEXT_TOP_N = 16        # accumulated context cap (generous; 8 from iter 1 + new)
MAUD_FORCE_IDS = ["maud-0684", "maud-1114", "maud-1452"]

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


def _run_with_retry(fn):
    """Retry on transient API errors (rate limit, server error)."""
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


# ── Query ID selection ────────────────────────────────────────────────────

def select_query_ids(
    all_ids: list[str],
    n: int = N_QUERIES_DEFAULT,
    force_include: list[str] | None = None,
    seed: str = QUERY_SEED,
) -> list[str]:
    """Deterministic, reproducible query selection via hash ordering.

    Force-includes specified IDs (e.g. MAUD access-miss), fills remaining
    slots from hash-ordered candidates.
    """
    forced = set(force_include or [])
    # Validate forced IDs exist
    all_set = set(all_ids)
    for fid in forced:
        if fid not in all_set:
            logger.warning("Forced ID %r not in corpus — skipping", fid)
            forced.discard(fid)

    def sort_key(qid):
        return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()

    ordered = sorted(all_ids, key=sort_key)
    selected = [q for q in ordered if q in forced]
    for q in ordered:
        if len(selected) >= n:
            break
        if q not in forced:
            selected.append(q)
    return sorted(selected)


# ── Loop primitives ───────────────────────────────────────────────────────

# Stopwords and helpers reused from nodes.py Phase 9
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were",
    "of", "in", "on", "at", "to", "for", "with",
    "that", "this", "it", "be", "has", "have",
    "and", "or", "but", "not", "by", "from",
})

_NEGATION_PATTERNS = (
    "does not", "is not", "are not", "was not", "were not",
    "shall not", "cannot", "never", "no ", "prohibited",
    "not permitted", "not allowed", "will not",
)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _detect_contradiction(claim_text: str, chunk_content: str) -> bool:
    """Conservative contradiction check (mirrors nodes.py)."""
    norm_claim = claim_text.lower()
    norm_chunk = chunk_content.lower()
    claim_tokens = set(norm_claim.split()) - _STOPWORDS
    if not claim_tokens:
        return False
    sentences = norm_chunk.replace("\n", " ").split(".")
    for sentence in sentences:
        has_negation = any(neg in sentence for neg in _NEGATION_PATTERNS)
        if not has_negation:
            continue
        sentence_tokens = set(sentence.split()) - _STOPWORDS
        overlap = claim_tokens & sentence_tokens
        if len(overlap) >= 2:
            return True
    return False


def split_claims(verification_result: dict, claims: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split claims into frozen (ENTAILED) and failed (not ENTAILED).

    Returns (frozen_claims, failed_claims) where each entry is the
    original claim dict augmented with 'verdict' and 'reason'.
    """
    details = verification_result.get("details", [])
    # Build verdict lookup by claim text (details parallel claims)
    frozen = []
    failed = []
    for claim, detail in zip(claims, details):
        augmented = {**claim, "verdict": detail.get("verdict", ""), "reason": detail.get("reason", "")}
        if detail.get("verdict") == "ENTAILED":
            frozen.append(augmented)
        else:
            failed.append(augmented)
    return frozen, failed


def run_verification(claims: list[dict], chunk_ids: list[str], chunk_texts: list[str]) -> dict:
    """Deterministic Phase 9 verification — mirrors nodes.py:565-715.

    Pure function, no LLM calls. Returns verification_result dict.
    """
    if not claims:
        return {
            "entailed": 0, "contradicted": 0, "baseless": 0,
            "total": 0, "score": 0.0, "has_contradiction": False,
            "details": [],
        }

    chunk_text_by_id = dict(zip(chunk_ids, chunk_texts))
    total = len(claims)
    entailed_count = 0
    contradicted_count = 0
    baseless_count = 0
    details: list[dict] = []

    for claim in claims:
        cited_chunk_id = claim.get("cited_chunk_id", "")
        cited_text = claim.get("cited_text", "").strip()
        claim_text = claim.get("claim", "").strip()

        # CHECK 1 — Chunk ID exists
        if cited_chunk_id not in chunk_text_by_id:
            baseless_count += 1
            details.append({"claim": claim_text, "cited_chunk_id": cited_chunk_id,
                            "verdict": "BASELESS", "reason": "chunk_id not in context"})
            continue

        chunk_content = chunk_text_by_id[cited_chunk_id]

        # CHECK 2 — Contradiction
        if _detect_contradiction(claim_text, chunk_content):
            contradicted_count += 1
            details.append({"claim": claim_text, "cited_chunk_id": cited_chunk_id,
                            "verdict": "CONTRADICTED", "reason": "chunk explicitly contradicts claim"})
            continue

        # CHECK 3 — Text presence
        norm_cited = _normalize_text(cited_text)
        norm_chunk = _normalize_text(chunk_content)

        # Test A — substring match
        if norm_cited and norm_cited in norm_chunk:
            entailed_count += 1
            details.append({"claim": claim_text, "cited_chunk_id": cited_chunk_id,
                            "verdict": "ENTAILED", "reason": "exact"})
            continue

        # Test B — token overlap
        cited_tokens = set(norm_cited.split()) - _STOPWORDS
        chunk_tokens = set(norm_chunk.split()) - _STOPWORDS
        overlap = cited_tokens & chunk_tokens
        denom = max(len(cited_tokens), 1)
        ratio = len(overlap) / denom

        if ratio >= 0.75:
            entailed_count += 1
            details.append({"claim": claim_text, "cited_chunk_id": cited_chunk_id,
                            "verdict": "ENTAILED", "reason": "token_overlap"})
        else:
            baseless_count += 1
            details.append({"claim": claim_text, "cited_chunk_id": cited_chunk_id,
                            "verdict": "BASELESS", "reason": "cited_text not found in chunk"})

    score = entailed_count / total if total > 0 else 0.0
    return {
        "entailed": entailed_count,
        "contradicted": contradicted_count,
        "baseless": baseless_count,
        "total": total,
        "score": score,
        "has_contradiction": contradicted_count > 0,
        "details": details,
    }


def build_deterministic_requery(failed_claims: list[dict]) -> str:
    """Arm 1: templated re-query from failed claim texts. No LLM."""
    return "; ".join(c["claim"] for c in failed_claims)[:256]


def build_llm_critic_requery(
    query: str,
    answer: str,
    failed_claims: list[dict],
    chunk_ids: list[str],
    chunk_texts: list[str],
) -> str:
    """Arm 2: LLM-steered re-query. DeepSeek reasons about the gap.

    The critic STEERS control flow — it must NOT feed into Layer-1
    taxonomy or Layer-2 metrics.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("Critic: no DEEPSEEK_API_KEY, falling back to deterministic")
        return build_deterministic_requery(failed_claims)

    failed_list = "\n".join(
        f"- {c['claim']} [verdict: {c.get('verdict', '?')}, reason: {c.get('reason', '?')}]"
        for c in failed_claims
    )
    # Summarize evidence briefly (first 200 chars per chunk, max 5 chunks)
    evidence_summary = "\n".join(
        f"[{cid}] {text[:200]}..."
        for cid, text in zip(chunk_ids[:5], chunk_texts[:5])
    )

    system_prompt = (
        "You are a retrieval gap analyst for a legal document retrieval system. "
        "Read the answer, the claims that failed verification, and the current "
        "evidence. Identify what specific information is missing and write a "
        "targeted search query (max 2 sentences) to find it within the same "
        "documents.\n\n"
        "Rules:\n"
        "- Return ONLY the search query as plain text\n"
        "- Focus on the specific missing evidence, not general topics\n"
        "- Do not repeat the original query verbatim\n"
        "- Maximum 2 sentences"
    )
    user_prompt = (
        f"Original query: {query}\n\n"
        f"Current answer: {answer[:500]}\n\n"
        f"Claims that failed verification:\n{failed_list}\n\n"
        f"Current evidence (summary):\n{evidence_summary}\n\n"
        "Write a targeted search query to find the missing evidence."
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=128,
            extra_body={"thinking": {"type": "disabled"}},
        )
        result = (response.choices[0].message.content or "").strip()
        if result:
            return result
    except Exception as exc:
        logger.warning("Critic LLM call failed (%s), falling back to deterministic", exc)

    return build_deterministic_requery(failed_claims)


def accumulate_chunks(
    existing_ids: list[str],
    existing_texts: list[str],
    new_fused_ids: list[str],
    new_fused_texts: list[str],
    top_n: int = CONTEXT_TOP_N,
) -> tuple[list[str], list[str]]:
    """Union existing chunks with newly-fused chunks (monotonic).

    Existing chunks kept in original order, new chunks appended in
    fused-score order. Dedup by chunk_id. Capped at top_n.
    """
    seen = set(existing_ids)
    union_ids = list(existing_ids)
    union_texts = list(existing_texts)
    for cid, text in zip(new_fused_ids, new_fused_texts):
        if cid not in seen:
            seen.add(cid)
            union_ids.append(cid)
            union_texts.append(text)
    result_ids = union_ids[:top_n]
    result_texts = union_texts[:top_n]
    # Assert monotonic: existing chunks all preserved AFTER truncation
    assert set(existing_ids).issubset(set(result_ids)), (
        f"Monotonic violation: {len(existing_ids)} existing chunks, "
        f"{len(result_ids)} after cap — existing chunks lost in truncation"
    )
    return result_ids, result_texts


def patch_synthesize(
    query: str,
    failed_claims: list[dict],
    chunk_ids: list[str],
    chunk_texts: list[str],
) -> list[dict]:
    """Modified Phase 8: generate claims ONLY for failed assertions.

    Returns list of claim dicts (same schema as Phase 8 output).
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("Patch-synthesis: no DEEPSEEK_API_KEY, returning empty")
        return []

    failed_list = "\n".join(
        f"{i+1}. {c['claim']} — reason: {c.get('reason', 'unverified')}"
        for i, c in enumerate(failed_claims)
    )
    context_string = "\n---\n".join(
        f"[{cid}]\n{text}" for cid, text in zip(chunk_ids, chunk_texts)
    )

    system_prompt = (
        "You are a legal document analyst. Some claims from a previous answer "
        "could not be verified against the evidence. New evidence has been "
        "retrieved. Your task is to produce replacement claims ONLY for the "
        "previously unverified assertions.\n\n"
        "Rules:\n"
        "- Produce claims ONLY for the failed assertions listed below\n"
        "- DO NOT reproduce claims that already passed verification\n"
        "- cited_chunk_id must be one of the chunk IDs provided in context\n"
        "- cited_text must be copied verbatim from the cited chunk\n"
        "- One atomic fact per claim — do not bundle multiple facts\n"
        "- If the evidence still does not support a failed claim, omit it\n"
        "- Do not invent information not present in the chunks"
    )
    user_prompt = (
        f"Query: {query}\n\n"
        f"Claims needing new evidence:\n{failed_list}\n\n"
        f"Available evidence chunks:\n{context_string}\n\n"
        "Produce claim-citation pairs for ONLY the failed claims above, "
        "using evidence from the chunks."
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
        # Validate chunk IDs
        valid_ids = set(chunk_ids)
        result_claims = []
        for c in response.claims:
            claim_dict = c.model_dump()
            if claim_dict["cited_chunk_id"] not in valid_ids:
                logger.warning("Patch-synthesis: claim cites unknown chunk %r", claim_dict["cited_chunk_id"])
            result_claims.append(claim_dict)
        return result_claims

    except Exception as exc:
        logger.warning("Patch-synthesis failed: %s", exc)
        return []


def final_resynthesize(
    query: str,
    all_claims: list[dict],
    chunk_ids: list[str],
    chunk_texts: list[str],
) -> str:
    """Final re-synthesis: produce coherent answer text incorporating
    ALL claims (frozen + patch) against accumulated context.

    Correction #1: frozen answer text would make correctness invisible
    to the judge. This call ensures patched claims reach the judged
    answer prose.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("Final re-synthesis: no DEEPSEEK_API_KEY")
        return ""

    claims_list = "\n".join(
        f"- {c['claim']} [source: {c.get('cited_chunk_id', '?')}]"
        for c in all_claims
    )
    context_string = "\n---\n".join(
        f"[{cid}]\n{text}" for cid, text in zip(chunk_ids, chunk_texts)
    )

    system_prompt = (
        "You are a legal document analyst. Write a coherent, complete answer "
        "to the query that incorporates ALL of the following verified claims. "
        "Do not omit any claim. Do not add information beyond what the claims "
        "and evidence support.\n\n"
        "Rules:\n"
        "- Every claim listed below must appear in your answer\n"
        "- Use the evidence chunks for additional context and phrasing\n"
        "- Produce a single coherent narrative, not a bulleted list of claims\n"
        "- Be specific and cite details from the evidence"
    )
    user_prompt = (
        f"Query: {query}\n\n"
        f"Verified claims to incorporate:\n{claims_list}\n\n"
        f"Evidence chunks:\n{context_string}\n\n"
        "Write a coherent answer incorporating ALL claims above."
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Final re-synthesis failed: %s", exc)
        return ""


# ── Per-query runner ──────────────────────────────────────────────────────

def _extract_record(
    final: dict,
    ground_truth,
    qid: str,
    query: str,
    arm: int,
    iterations: int,
    gate_triggered: bool,
    grounding_score: float,
    frozen_count: int = 0,
    patch_count: int = 0,       # claims successfully patched (failed -> entailed)
    residual_failed: int = 0,   # claims still unverified after loop
    requery_text: str = "",
    latency_ms: int = 0,
    override_answer: str | None = None,
    override_claims: list[dict] | None = None,
    override_ctx_ids: list[str] | None = None,
    override_ctx_texts: list[str] | None = None,
):
    """Build a judge-compatible JSONL record from pipeline state."""
    from core.measurement.metrics import compute_all_k
    from core.measurement.taxonomy import classify_with_confidence

    reranked = final.get("reranked_result", {})
    spans = [tuple(s) for s in reranked.get("spans", [])]
    ids = reranked.get("ids", [])
    gt_spans = ground_truth.get_spans(qid)
    gt_doc = ground_truth.get_doc_id(qid)
    metrics = compute_all_k(spans, gt_spans)

    retrieved_with = [(cid.split("#")[0], s[0], s[1]) for cid, s in zip(ids, spans)]
    gt_with = [(gt_doc, s[0], s[1]) for s in gt_spans]
    classification = classify_with_confidence(retrieved_with, gt_with)

    ctx_ids = override_ctx_ids if override_ctx_ids is not None else final.get("context_ids", [])
    ctx_texts = override_ctx_texts if override_ctx_texts is not None else final.get("context_chunks", [])
    context_chunks = [
        {"chunk_id": cid, "doc_id": cid.split("#")[0], "content": text}
        for cid, text in zip(ctx_ids, ctx_texts)
    ]

    answer = override_answer if override_answer is not None else final.get("answer", "")
    claims = override_claims if override_claims is not None else final.get("claims", [])

    return {
        "query_id": qid,
        "query": query,
        "answer": answer,
        "claims": claims,
        "context_chunks": context_chunks,
        "p_at_1": metrics.p_at_k.get(1, 0.0),
        "r_at_8": metrics.r_at_k.get(8, 0.0),
        "failure_type": classification.failure_type.name,
        "routing_hit": (gt_doc in final.get("routed_docs", [])) if final.get("routed_docs") else None,
        "arm": arm,
        "iterations_used": iterations,
        "grounding_score": round(grounding_score, 4),
        "gate_triggered": gate_triggered,
        "frozen_claim_count": frozen_count,
        "patch_claim_count": patch_count,
        "residual_failed_count": residual_failed,
        "requery_text": requery_text,
        "latency_ms": latency_ms,
    }


def run_loop(
    qid: str,
    query: str,
    original_answer: str,
    frozen_claims: list[dict],
    failed_claims: list[dict],
    kept_chunk_ids: list[str],
    kept_chunk_texts: list[str],
    routed_docs: list[str] | None,
    context,
    ground_truth,
    params: dict,
    gate: str,   # "deterministic" or "llm_critic"
    final: dict,
    arm: int,
) -> dict:
    """Run the freeze/accumulate/patch loop for Arms 1 or 2.

    Starts from iteration-1 state (shared with Arm 0). Runs up to
    MAX_ITERATIONS-1 additional iterations.
    """
    from core.retrieval.fusion import cc_fusion

    t0 = time.perf_counter()
    alpha = params["cc_alpha"]
    ds = params["dataset_name"]
    iterations_used = 1
    all_requery_texts = []

    initial_frozen_count = len(frozen_claims)
    cur_frozen = [copy.deepcopy(c) for c in frozen_claims]
    cur_failed = [copy.deepcopy(c) for c in failed_claims]
    cur_chunk_ids = list(kept_chunk_ids)
    cur_chunk_texts = list(kept_chunk_texts)

    for iteration in range(2, MAX_ITERATIONS + 1):
        if not cur_failed:
            break

        # ── Step 1: Generate delta re-query ───────────────────────────
        if gate == "deterministic":
            delta_query = build_deterministic_requery(cur_failed)
        else:
            delta_query = _run_with_retry(lambda: build_llm_critic_requery(
                query, original_answer, cur_failed,
                cur_chunk_ids, cur_chunk_texts,
            ))
        all_requery_texts.append(delta_query)
        logger.info("[%s] arm=%d iter=%d requery: %s", qid, arm, iteration, delta_query[:80])

        # ── Step 2: Delta retrieval WITHIN routed docs ────────────────
        dense_result = _run_with_retry(lambda: context.qdrant_retriever.retrieve(
            delta_query, dataset_name=ds, doc_ids=routed_docs,
        ))
        sparse_result = context.bm25_retriever.retrieve(
            delta_query, dataset_name=ds, doc_ids=routed_docs,
        )

        # ── Step 3: CC-fusion merge (NOT reranker) ────────────────────
        fused = cc_fusion(sparse_result, dense_result, alpha=alpha, top_n=50)

        # ── Step 4: Accumulate chunks (union, monotonic) ──────────────
        prev_count = len(cur_chunk_ids)
        cur_chunk_ids, cur_chunk_texts = accumulate_chunks(
            cur_chunk_ids, cur_chunk_texts,
            fused.ids, fused.contents,
            top_n=CONTEXT_TOP_N,
        )
        new_count = len(cur_chunk_ids) - prev_count
        logger.info("[%s] arm=%d iter=%d: +%d new chunks (%d total)",
                    qid, arm, iteration, new_count, len(cur_chunk_ids))

        # ── Step 5: Patch-synthesize failed claims ────────────────────
        patch_claims = _run_with_retry(lambda: patch_synthesize(
            query, cur_failed, cur_chunk_ids, cur_chunk_texts,
        ))

        # ── Step 6: Merge frozen + patch ──────────────────────────────
        # Strip verdict/reason augmentation from frozen claims for clean merge
        clean_frozen = [{k: v for k, v in c.items() if k in ("claim", "cited_chunk_id", "cited_text")}
                        for c in cur_frozen]
        all_claims = clean_frozen + patch_claims

        # ── Step 7: Re-verify ALL claims ──────────────────────────────
        verification = run_verification(all_claims, cur_chunk_ids, cur_chunk_texts)
        score = verification["score"]
        iterations_used = iteration
        logger.info("[%s] arm=%d iter=%d score=%.2f (%d/%d entailed)",
                    qid, arm, iteration, score, verification["entailed"], verification["total"])

        # ── Step 8: Update frozen/failed ──────────────────────────────
        cur_frozen, cur_failed = split_claims(verification, all_claims)

        # ── Gate check ────────────────────────────────────────────────
        if score >= GATE_THRESHOLD:
            break

    # ── Final re-synthesis: produce answer text reflecting ALL claims ──
    # Strip verdict/reason for clean claims list
    final_claims = [{k: v for k, v in c.items() if k in ("claim", "cited_chunk_id", "cited_text")}
                    for c in cur_frozen]
    # Add any remaining failed claims (they stay, just unverified)
    final_failed_clean = [{k: v for k, v in c.items() if k in ("claim", "cited_chunk_id", "cited_text")}
                          for c in cur_failed]
    final_all_claims = final_claims + final_failed_clean

    resynthesized_answer = _run_with_retry(lambda: final_resynthesize(
        query, final_all_claims, cur_chunk_ids, cur_chunk_texts,
    ))
    # Fall back to original answer if re-synthesis failed
    if not resynthesized_answer:
        resynthesized_answer = original_answer

    # Re-verify after re-synthesis (claims unchanged, answer text new)
    final_verification = run_verification(final_all_claims, cur_chunk_ids, cur_chunk_texts)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return _extract_record(
        final, ground_truth, qid, query,
        arm=arm,
        iterations=iterations_used,
        gate_triggered=True,
        grounding_score=final_verification["score"],
        frozen_count=len(final_claims),
        # patch_count = claims that were originally failed but are now entailed
        patch_count=max(0, len(final_claims) - initial_frozen_count),
        residual_failed=len(final_failed_clean),
        requery_text=" | ".join(all_requery_texts),
        latency_ms=latency_ms,
        override_answer=resynthesized_answer,
        override_claims=final_all_claims,
        override_ctx_ids=cur_chunk_ids,
        override_ctx_texts=cur_chunk_texts,
    )


def run_query(qid, query, compiled, context, ground_truth, params):
    """Run all 3 arms for a single query. Returns list of 3 records."""
    t0 = time.perf_counter()

    # ── Arm 0: single-shot (control) ──────────────────────────────────
    state = {
        "raw_query": query,
        "iteration": 1,
        "max_iterations": 1,
        "loop_complete": False,
        "trace_id": "local",
    }
    final = _run_with_retry(
        lambda: compiled.invoke(state, config={"configurable": {"thread_id": str(uuid.uuid4())}})
    )
    arm0_latency = int((time.perf_counter() - t0) * 1000)

    verification = final.get("verification_result", {})
    score = verification.get("score", 0.0)
    claims = final.get("claims", [])

    arm0 = _extract_record(
        final, ground_truth, qid, query,
        arm=0, iterations=1, gate_triggered=False,
        grounding_score=score, latency_ms=arm0_latency,
    )

    # ── Gate check: does this query enter the loop? ───────────────────
    # Identify failed claims
    frozen, failed = split_claims(verification, claims)
    gate_triggers = score < GATE_THRESHOLD and len(failed) > 0

    if not gate_triggers:
        # Gate NOT triggered — Arms 1 & 2 = Arm 0 (identical results)
        arm1 = {**arm0, "arm": 1, "gate_triggered": False}
        arm2 = {**arm0, "arm": 2, "gate_triggered": False}
        return [arm0, arm1, arm2]

    # ── Extract shared iteration-1 state for loop arms ────────────────
    routed_docs = final.get("routed_docs")
    ctx_ids = final.get("context_ids", [])
    ctx_texts = final.get("context_chunks", [])
    original_answer = final.get("answer", "")

    # ── Arm 1: deterministic gate loop ────────────────────────────────
    arm1 = run_loop(
        qid, query, original_answer, frozen, failed,
        ctx_ids, ctx_texts, routed_docs,
        context, ground_truth, params,
        gate="deterministic", final=final, arm=1,
    )

    # ── Arm 2: LLM critic gate loop ──────────────────────────────────
    arm2 = run_loop(
        qid, query, original_answer, frozen, failed,
        ctx_ids, ctx_texts, routed_docs,
        context, ground_truth, params,
        gate="llm_critic", final=final, arm=2,
    )

    return [arm0, arm1, arm2]


# ── Per-corpus runner ─────────────────────────────────────────────────────

def run_corpus(corpus_name: str, output_path: Path, workers: int = 8, n_queries: int = N_QUERIES_DEFAULT):
    """Run all 3 arms on n_queries from a single corpus."""
    params = CORPUS_PARAMS[corpus_name]

    # ── Set env vars (locked config) ──────────────────────────────────
    os.environ["MERIDIAN_EMBED_MODEL"] = "voyage-4"
    os.environ["MERIDIAN_NO_REWRITE"] = "1"    # Finding 36: rewrite OFF
    os.environ["MERIDIAN_NO_RERANK"] = "1"     # Finding 13/21: reranker OFF
    os.environ["MERIDIAN_CC_ALPHA"] = str(params["cc_alpha"])
    os.environ["MERIDIAN_ROUTING_TOPK"] = str(params["routing_topk"])
    os.environ["MERIDIAN_ROUTING_INDEX"] = params["routing_index"]
    os.environ["MERIDIAN_ROUTING_ALPHA"] = str(params["routing_alpha"])

    # Reset routing singleton (global state, must reset between corpora)
    import core.retrieval.routing as routing_mod
    routing_mod._router = None

    from core.supervisor.context import PipelineContext
    from core.supervisor.graph import compile_graph

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    context = PipelineContext.build(
        corpus_path=params["parquet"],
        qdrant_url=qdrant_url,
        collection_name=params["collection"],
        dataset_name=params["dataset_name"],
        top_k=50,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    compiled, _ = compile_graph(db_path="data/mini_e2e_pipeline.sqlite", context=context)

    # Load ground truth
    mod = importlib.import_module(params["gt_module"])
    gt_cls = getattr(mod, params["gt_class"])
    ground_truth = gt_cls(params["benchmark"], params["corpus_dir"])

    # Load benchmark and select fixed query IDs
    data = json.loads(params["benchmark"].read_text(encoding="utf-8"))
    all_questions = {t["query_id"]: t["query"] for t in data["tests"]}
    force = MAUD_FORCE_IDS if corpus_name == "maud" else None
    selected_ids = select_query_ids(list(all_questions.keys()), n=n_queries, force_include=force)

    print(f"\n{'='*60}")
    print(f"CORPUS: {corpus_name} | {len(selected_ids)} queries | "
          f"alpha={params['cc_alpha']} routing={params['routing_topk']}")
    print(f"{'='*60}")

    write_lock = threading.Lock()
    all_records: list[dict] = []
    error_count = 0

    def _worker(qid):
        nonlocal error_count
        try:
            records = run_query(
                qid, all_questions[qid], compiled, context, ground_truth, params,
            )
            with write_lock:
                all_records.extend(records)
                # Print progress for Arm 0
                r0 = records[0]
                gate = "LOOP" if r0.get("gate_triggered") or records[1].get("gate_triggered") else "PASS"
                print(f"  [{qid}] P@1={r0['p_at_1']:.2f} R@8={r0['r_at_8']:.2f} "
                      f"{r0['failure_type']} score={r0['grounding_score']:.2f} {gate}")
        except Exception as exc:
            with write_lock:
                error_count += 1
                print(f"  [{qid}] FAILED: {exc}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=min(workers, 12)) as executor:
        futures = {executor.submit(_worker, qid): qid for qid in selected_ids}
        for f in as_completed(futures):
            f.result()

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in sorted(all_records, key=lambda x: (x["arm"], x["query_id"])):
            f.write(json.dumps(r) + "\n")

    # Per-corpus summary
    print(f"\n{corpus_name}: {len(all_records)} records ({error_count} errors) -> {output_path}")
    _print_arm_summary(all_records, corpus_name)


def _print_arm_summary(records: list[dict], corpus_name: str):
    """Print per-arm summary table for a single corpus."""
    for arm in [0, 1, 2]:
        arm_recs = [r for r in records if r["arm"] == arm]
        if not arm_recs:
            continue
        n = len(arm_recs)
        avg_p1 = sum(r["p_at_1"] for r in arm_recs) / n
        avg_r8 = sum(r["r_at_8"] for r in arm_recs) / n
        avg_score = sum(r["grounding_score"] for r in arm_recs) / n
        ft = Counter(r["failure_type"] for r in arm_recs)
        triggered = sum(1 for r in arm_recs if r["gate_triggered"])
        avg_iters = sum(r["iterations_used"] for r in arm_recs) / n
        avg_latency = sum(r["latency_ms"] for r in arm_recs) / n

        arm_label = {0: "SINGLE-SHOT", 1: "DETERM-GATE", 2: "CRITIC-GATE"}[arm]
        print(f"  Arm {arm} ({arm_label}): P@1={avg_p1:.3f} R@8={avg_r8:.3f} "
              f"grounding={avg_score:.3f} iters={avg_iters:.1f} "
              f"triggered={triggered}/{n} latency={avg_latency:.0f}ms")
        print(f"    Taxonomy: {dict(ft)}")

    # Correction #2: per-access-miss gate trigger diagnostic (MAUD only)
    if corpus_name == "maud":
        for target_id in MAUD_FORCE_IDS:
            for arm in [0, 1, 2]:
                recs = [r for r in records if r["query_id"] == target_id and r["arm"] == arm]
                if recs:
                    r = recs[0]
                    print(f"  ACCESS-MISS {target_id} arm={arm}: gate={'TRIGGERED' if r['gate_triggered'] else 'PASS'} "
                          f"score={r['grounding_score']:.2f} iters={r['iterations_used']} "
                          f"ft={r['failure_type']}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Mini E2E: 3 arms × 4 corpora × N queries")
    p.add_argument("--corpus", nargs="*", default=list(CORPUS_PARAMS.keys()),
                   choices=list(CORPUS_PARAMS.keys()))
    p.add_argument("--output-dir", type=Path, default=Path("data/mini_e2e"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-queries", type=int, default=N_QUERIES_DEFAULT)
    args = p.parse_args()

    print(f"Mini E2E: {len(args.corpus)} corpora × {args.n_queries} queries × 3 arms")
    print(f"Config: NO_REWRITE=1, NO_RERANK=1, routing top-3, CC fusion (per-corpus alpha)")
    print(f"Gate: score<{GATE_THRESHOLD}, max {MAX_ITERATIONS} iterations")
    print(f"Output: {args.output_dir}/")

    for corpus in args.corpus:
        output = args.output_dir / f"mini_e2e_{corpus}.jsonl"
        run_corpus(corpus, output, workers=args.workers, n_queries=args.n_queries)

    # Cross-corpus summary
    print(f"\n{'='*60}")
    print("CROSS-CORPUS SUMMARY")
    print(f"{'='*60}")
    all_records = []
    for corpus in args.corpus:
        output = args.output_dir / f"mini_e2e_{corpus}.jsonl"
        if output.exists():
            with output.open() as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))

    if all_records:
        _print_arm_summary(all_records, "all")

    print("\nNext steps:")
    print("  1. Run judge_answers_v2.py on each arm's output")
    print("  2. Run judge_faithfulness.py on each arm's output (holistic)")
    print("  3. Join verdicts + faithfulness for CORRECT+FAITHFUL table")
    print("  4. Check regressions on CORRECT+FAITHFUL (Correction #3)")


if __name__ == "__main__":
    main()
