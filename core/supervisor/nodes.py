"""Supervisor graph nodes — v2 thin implementations.

Phases 3–9: minimum viable logic that passes real data through the pipeline.
Phase 10: stub only (agentic loop not yet implemented).

Phase map:
  3  query_understanding   — passthrough (raw_query → rewritten_query)
  4  retrieval             — dense + sparse channels via context retrievers
  5  fusion                — RRF via core.retrieval.fusion.rrf()
  6  reranking             — passthrough (no reranker yet)
  7  context_construction  — top 8 chunks from reranked result
  8  synthesis             — DeepSeek-pro structured answer + claim-citations
  9  verification          — stub (requires Phase 8 structured output)
  10 agentic_loop          — stub (not implemented)
"""

from __future__ import annotations

import logging
import os

from core.retrieval.base import RetrievalResult
from core.retrieval.fusion import cc_fusion, rrf, weighted_rrf
from core.supervisor.context import PipelineContext
from core.supervisor.state import ExperimentState
from core.supervisor.tracing import start_span, end_span

logger = logging.getLogger(__name__)


def _get_trace(state: ExperimentState, context: PipelineContext):
    """Get a Langfuse trace proxy for creating child spans.

    Returns the langfuse_client with trace_context set so
    start_observation() creates children under the correct trace.
    Returns None if tracing is unavailable.
    """
    if not context.langfuse_client:
        return None
    trace_id = state.get("trace_id")
    if not trace_id or trace_id == "local":
        return None
    try:
        from langfuse.types import TraceContext

        class _TraceProxy:
            """Thin proxy: routes start_observation to the client with trace_context."""
            def __init__(self, client, trace_id):
                self._client = client
                self._tc = TraceContext(trace_id=trace_id)

            def start_observation(self, **kwargs):
                return self._client.start_observation(trace_context=self._tc, **kwargs)

        return _TraceProxy(context.langfuse_client, trace_id)
    except Exception:
        return None


# ── Serialization helpers ─────────────────────────────────────────────────

def _serialize_result(r: RetrievalResult) -> dict:
    """Convert RetrievalResult to a JSON-serializable dict."""
    return {
        "contents": list(r.contents),
        "ids": list(r.ids),
        "scores": [float(s) for s in r.scores],
        "spans": [[int(s), int(e)] for s, e in r.spans],
    }


def _deserialize_result(d: dict) -> RetrievalResult:
    """Reconstruct RetrievalResult from a serialized dict."""
    return RetrievalResult(
        contents=d["contents"],
        ids=d["ids"],
        scores=d["scores"],
        spans=[tuple(s) for s in d["spans"]],
    )


# ── Phase nodes ───────────────────────────────────────────────────────────

def query_understanding(state: ExperimentState, context: PipelineContext) -> dict:
    """Phase 3 — DeepSeek-flash query rewriting for retrieval."""
    trace = _get_trace(state, context)
    span = start_span(trace, "phase_3_query_understanding", {"raw_query": state.get("raw_query", "")})
    raw = state.get("raw_query", "")
    if not raw:
        logger.warning("Phase 3 (query_understanding): empty raw_query, skipping")
        return {}

    # A/B toggle: skip rewriting when MERIDIAN_NO_REWRITE=1
    if os.environ.get("MERIDIAN_NO_REWRITE") == "1":
        logger.info("Phase 3: passthrough (MERIDIAN_NO_REWRITE=1)")
        result = {"rewritten_query": raw}
        end_span(span, result)
        return result

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("Phase 3 (query_understanding): DEEPSEEK_API_KEY not set, passthrough")
        return {"rewritten_query": raw}

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query rewriter for a legal document retrieval system.\n"
                        "Your job is to rewrite the user's query to improve retrieval quality.\n"
                        "Rules:\n"
                        "- Fix spelling and grammar errors\n"
                        "- Expand abbreviations and informal language to formal legal terminology\n"
                        "- Keep the same meaning and scope — do not add or remove information needs\n"
                        "- Return ONLY the rewritten query as plain text, no explanation, "
                        "no preamble, no punctuation changes beyond what is necessary"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Rewrite this query for retrieval: {raw}",
                },
            ],
            max_tokens=128,
            extra_body={"thinking": {"type": "disabled"}},
        )
        rewritten = (response.choices[0].message.content or "").strip()
        if not rewritten:
            rewritten = raw

        logger.info("Phase 3: '%s' -> '%s'", raw, rewritten)
        result = {"rewritten_query": rewritten}
        end_span(span, result)
        return result

    except Exception as exc:
        logger.warning("Phase 3 (query_understanding): rewrite failed (%s), passthrough", exc)
        result = {"rewritten_query": raw}
        end_span(span, result)
        return result


def retrieval(state: ExperimentState, context: PipelineContext) -> dict:
    """Phase 4 — call dense + sparse retrieval channels."""
    trace = _get_trace(state, context)
    query = state.get("rewritten_query") or state.get("raw_query", "")
    ds = context.dataset_name
    span = start_span(trace, "phase_4_retrieval", {"query": query, "dataset": ds})

    top_k_override = os.environ.get("MERIDIAN_TOP_K")
    top_k = int(top_k_override) if top_k_override else None

    logger.info("Phase 4 (retrieval): dense + sparse on dataset=%r", ds)
    dense_result = context.qdrant_retriever.retrieve(query, top_k=top_k, dataset_name=ds)
    sparse_result = context.bm25_retriever.retrieve(query, top_k=top_k, dataset_name=ds)

    result = {
        "retrieval_bundle": {
            "dense": _serialize_result(dense_result),
            "sparse": _serialize_result(sparse_result),
        },
    }
    end_span(span, {"dense_count": len(dense_result.ids), "sparse_count": len(sparse_result.ids)})
    return result


def fusion(state: ExperimentState, context: PipelineContext) -> dict:
    """Phase 5 — RRF fusion of dense + sparse results."""
    trace = _get_trace(state, context)
    span = start_span(trace, "phase_5_fusion", {})
    bundle = state.get("retrieval_bundle", {})
    dense_d = bundle.get("dense")
    sparse_d = bundle.get("sparse")

    if not dense_d or not sparse_d:
        logger.warning("Phase 5 (fusion): missing retrieval bundle, skipping")
        end_span(span, {"skipped": True})
        return {}

    dense_rr = _deserialize_result(dense_d)
    sparse_rr = _deserialize_result(sparse_d)

    top_n_override = os.environ.get("MERIDIAN_FUSION_TOP_N")
    top_n = int(top_n_override) if top_n_override else 50

    alpha_str = os.environ.get("MERIDIAN_CC_ALPHA")
    wrrf_str = os.environ.get("MERIDIAN_WRRF_SPARSE")

    if alpha_str is not None:
        alpha = float(alpha_str)
        fused = cc_fusion(sparse_rr, dense_rr, alpha=alpha, top_n=top_n)
        logger.info("Phase 5 (fusion): CC alpha=%.2f produced %d results", alpha, len(fused.ids))
    elif wrrf_str is not None:
        sw = float(wrrf_str)
        fused = weighted_rrf(sparse_rr, dense_rr, sparse_weight=sw, top_n=top_n)
        logger.info("Phase 5 (fusion): weighted RRF sparse=%.2f produced %d results", sw, len(fused.ids))
    else:
        fused = rrf(sparse_rr, dense_rr, top_n=top_n)
        logger.info("Phase 5 (fusion): RRF produced %d results", len(fused.ids))
    result = {"fused_result": _serialize_result(fused)}
    end_span(span, {"fused_count": len(fused.ids)})
    return result


def reranking(state: ExperimentState, context: PipelineContext) -> dict:
    """Phase 6 — Voyage rerank-2.5 over all fused candidates."""
    trace = _get_trace(state, context)
    fused = state.get("fused_result")
    if not fused:
        logger.warning("Phase 6 (reranking): no fused_result, skipping")
        return {}

    # A/B: skip reranker, pass RRF top-8 directly
    if os.environ.get("MERIDIAN_NO_RERANK") == "1":
        logger.info("Phase 6: passthrough (MERIDIAN_NO_RERANK=1)")
        return {
            "reranked_result": {
                "contents": fused.get("contents", [])[:8],
                "ids":      fused.get("ids", [])[:8],
                "scores":   fused.get("scores", [])[:8],
                "spans":    fused.get("spans", [])[:8],
            }
        }

    query = state.get("rewritten_query") or state.get("raw_query", "")

    contents = fused.get("contents", [])
    ids = fused.get("ids", [])
    scores = fused.get("scores", [])
    spans = fused.get("spans", [])
    n_in = len(contents)
    span = start_span(trace, "phase_6_reranking", {"n_candidates": n_in})

    if not contents:
        logger.warning("Phase 6 (reranking): fused_result empty, skipping")
        end_span(span, {"skipped": True})
        return {"reranked_result": fused}

    try:
        import voyageai

        vo = voyageai.Client()
        result = vo.rerank(
            query=query,
            documents=contents,
            model="rerank-2.5",
            top_k=8,
        )

        # Reconstruct reranked result using .index to map back to original positions
        reranked_contents = []
        reranked_ids = []
        reranked_scores = []
        reranked_spans = []

        for rr in result.results:
            idx = rr.index
            reranked_contents.append(contents[idx])
            reranked_ids.append(ids[idx])
            reranked_scores.append(float(rr.relevance_score))
            reranked_spans.append(spans[idx])

        n_out = len(reranked_ids)
        top_score = reranked_scores[0] if reranked_scores else 0.0
        logger.info(
            "Phase 6 (reranking): reranked %d -> %d, top score: %.4f",
            n_in, n_out, top_score,
        )

        # Log reranker span to Phoenix (no-op if PHOENIX_ENABLED!=1)
        from core.supervisor.phoenix_tracing import log_reranker_span

        log_reranker_span(
            query=query,
            candidates_in=[
                {"chunk_id": cid, "score": s, "text": t[:100]}
                for cid, s, t in zip(ids, scores, contents)
            ],
            candidates_out=[
                {"chunk_id": cid, "score": s, "text": t[:100]}
                for cid, s, t in zip(
                    reranked_ids, reranked_scores, reranked_contents
                )
            ],
            model="rerank-2.5",
        )

        result = {
            "reranked_result": {
                "contents": reranked_contents,
                "ids": reranked_ids,
                "scores": reranked_scores,
                "spans": reranked_spans,
            },
        }
        end_span(span, {"n_in": n_in, "n_out": n_out, "top_score": top_score})
        return result

    except Exception as exc:
        logger.warning(
            "Phase 6 (reranking): reranker failed (%s), falling back to fused top 8",
            exc,
        )
        result = {
            "reranked_result": {
                "contents": contents[:8],
                "ids": ids[:8],
                "scores": scores[:8],
                "spans": spans[:8],
            },
        }
        end_span(span, {"fallback": True, "error": str(exc)})
        return result


def context_construction(state: ExperimentState, context: PipelineContext) -> dict:
    """Phase 7 — take top 8 chunks from reranked result."""
    trace = _get_trace(state, context)
    span = start_span(trace, "phase_7_context_construction", {})
    reranked = state.get("reranked_result")
    if not reranked:
        logger.warning("Phase 7 (context_construction): no reranked_result, skipping")
        end_span(span, {"skipped": True})
        return {}

    top_n = 8
    contents = reranked.get("contents", [])[:top_n]
    ids = reranked.get("ids", [])[:top_n]

    logger.info("Phase 7 (context_construction): %d chunks selected", len(contents))
    result = {"context_chunks": contents, "context_ids": ids}
    end_span(span, {"n_chunks": len(contents), "chunk_ids": ids})
    return result


def synthesis(state: ExperimentState, context: PipelineContext) -> dict:
    """Phase 8 — structured DeepSeek call with claim-citation pairs."""
    trace = _get_trace(state, context)
    span = start_span(trace, "phase_8_synthesis", {"query": state.get("rewritten_query") or state.get("raw_query", "")})
    query = state.get("rewritten_query") or state.get("raw_query", "")
    chunks = state.get("context_chunks", [])
    chunk_ids = state.get("context_ids", [])

    if not chunks or not chunk_ids:
        logger.warning("Phase 8 (synthesis): no context chunks, skipping")
        return {"answer": "[no context]", "claims": []}

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("Phase 8 (synthesis): DEEPSEEK_API_KEY not set, skipping")
        return {"answer": "[Phase 8 skipped: no API key]", "claims": []}

    # Build context string with chunk IDs as explicit labels
    context_string = "\n---\n".join(
        f"[{cid}]\n{text}" for cid, text in zip(chunk_ids, chunks)
    )

    system_prompt = (
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

        logger.info("Phase 8 (synthesis): calling deepseek-v4-flash (structured)")
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

        # Validate cited_chunk_ids against context_ids
        valid_ids = set(chunk_ids)
        n_valid = sum(1 for c in response.claims if c.cited_chunk_id in valid_ids)
        for c in response.claims:
            if c.cited_chunk_id not in valid_ids:
                logger.warning(
                    "Phase 8: claim cites unknown chunk_id %r", c.cited_chunk_id
                )

        logger.info(
            "Phase 8: %d claims extracted, %d chunk IDs valid",
            len(response.claims), n_valid,
        )

        result = {
            "answer": response.answer,
            "claims": [c.model_dump() for c in response.claims],
        }
        end_span(span, {"n_claims": len(response.claims), "n_valid": n_valid})
        return result

    except Exception as exc:
        logger.error("Phase 8 (synthesis): structured call failed: %s", exc)
        result = {"answer": "[Phase 8 failed]", "claims": []}
        end_span(span, {"error": str(exc)})
        return result


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
    """Collapse whitespace and lowercase for comparison."""
    return " ".join(text.lower().split())


def _detect_contradiction(claim_text: str, chunk_content: str) -> bool:
    """Conservative contradiction check: negation proximate to claim terms.

    Only flags CONTRADICTED when a negation pattern appears in the chunk
    near (same sentence) key terms from the claim.
    """
    norm_claim = claim_text.lower()
    norm_chunk = chunk_content.lower()

    # Extract non-stopword terms from claim (key content words)
    claim_tokens = set(norm_claim.split()) - _STOPWORDS
    if not claim_tokens:
        return False

    # Find sentences in chunk that contain negation
    sentences = norm_chunk.replace("\n", " ").split(".")
    for sentence in sentences:
        has_negation = any(neg in sentence for neg in _NEGATION_PATTERNS)
        if not has_negation:
            continue
        # Check if claim's key terms appear in this negated sentence
        sentence_tokens = set(sentence.split()) - _STOPWORDS
        overlap = claim_tokens & sentence_tokens
        # Need at least 2 overlapping content words to be proximate
        if len(overlap) >= 2:
            return True

    return False


def verification(state: ExperimentState, context: PipelineContext) -> dict:
    """Phase 9 — three-way deterministic citation-traceability verification.

    No LLM calls. Produces ENTAILED / CONTRADICTED / BASELESS verdicts
    per claim using normalized text matching against chunk content.

    NOTE: cited_chunk_id is singular now. When NLI model lands in
    Stage 3, this becomes cited_chunk_ids (list) for multi-chunk support.
    """
    trace = _get_trace(state, context)
    span = start_span(trace, "phase_9_verification", {"n_claims": len(state.get("claims", []))})
    claims = state.get("claims", [])
    context_chunks = state.get("context_chunks", [])
    context_ids = state.get("context_ids", [])

    if not claims:
        logger.info("Phase 9 (verification): no claims to verify")
        return {
            "verification_result": {
                "entailed": 0,
                "contradicted": 0,
                "baseless": 0,
                "total": 0,
                "score": 0.0,
                "has_contradiction": False,
                "note": "no claims to verify",
                "details": [],
            },
        }

    chunk_text_by_id = dict(zip(context_ids, context_chunks))
    total = len(claims)
    entailed_count = 0
    contradicted_count = 0
    baseless_count = 0
    details: list[dict] = []

    for i, claim in enumerate(claims):
        cited_chunk_id = claim.get("cited_chunk_id", "")
        cited_text = claim.get("cited_text", "").strip()
        claim_text = claim.get("claim", "").strip()

        # CHECK 1 — Chunk ID exists?
        if cited_chunk_id not in chunk_text_by_id:
            baseless_count += 1
            reason = "chunk_id not in context"
            logger.info(
                "Phase 9: claim %d/%d [BASELESS] — %s (chunk: %s)",
                i + 1, total, reason, cited_chunk_id[:50],
            )
            details.append({
                "claim": claim_text,
                "cited_chunk_id": cited_chunk_id,
                "verdict": "BASELESS",
                "reason": reason,
            })
            continue

        chunk_content = chunk_text_by_id[cited_chunk_id]

        # CHECK 2 — Contradiction detection
        if _detect_contradiction(claim_text, chunk_content):
            contradicted_count += 1
            reason = "chunk explicitly contradicts claim"
            logger.info(
                "Phase 9: claim %d/%d [CONTRADICTED] — %s (chunk: %s)",
                i + 1, total, reason, cited_chunk_id[:50],
            )
            details.append({
                "claim": claim_text,
                "cited_chunk_id": cited_chunk_id,
                "verdict": "CONTRADICTED",
                "reason": reason,
            })
            continue

        # CHECK 3 — Text presence (entailment)
        norm_cited = _normalize_text(cited_text)
        norm_chunk = _normalize_text(chunk_content)

        # Test A — substring match
        if norm_cited and norm_cited in norm_chunk:
            entailed_count += 1
            reason = "exact"
            logger.info(
                "Phase 9: claim %d/%d [ENTAILED] — %s (chunk: %s)",
                i + 1, total, reason, cited_chunk_id[:50],
            )
            details.append({
                "claim": claim_text,
                "cited_chunk_id": cited_chunk_id,
                "verdict": "ENTAILED",
                "reason": reason,
            })
            continue

        # Test B — token overlap (after stopword removal)
        cited_tokens = set(norm_cited.split()) - _STOPWORDS
        chunk_tokens = set(norm_chunk.split()) - _STOPWORDS
        overlap = cited_tokens & chunk_tokens
        denom = max(len(cited_tokens), 1)
        ratio = len(overlap) / denom

        if ratio >= 0.75:
            entailed_count += 1
            reason = "token_overlap"
            logger.info(
                "Phase 9: claim %d/%d [ENTAILED] — %s (%.2f) (chunk: %s)",
                i + 1, total, reason, ratio, cited_chunk_id[:50],
            )
            details.append({
                "claim": claim_text,
                "cited_chunk_id": cited_chunk_id,
                "verdict": "ENTAILED",
                "reason": reason,
            })
        else:
            baseless_count += 1
            reason = "cited_text not found in chunk"
            logger.info(
                "Phase 9: claim %d/%d [BASELESS] — %s (overlap %.2f) (chunk: %s)",
                i + 1, total, reason, ratio, cited_chunk_id[:50],
            )
            details.append({
                "claim": claim_text,
                "cited_chunk_id": cited_chunk_id,
                "verdict": "BASELESS",
                "reason": reason,
            })

    score = entailed_count / total if total > 0 else 0.0
    has_contradiction = contradicted_count > 0

    logger.info(
        "Phase 9: %d/%d entailed, %d contradicted, %d baseless, score=%.2f, has_contradiction=%s",
        entailed_count, total, contradicted_count, baseless_count, score, has_contradiction,
    )

    result = {
        "verification_result": {
            "entailed": entailed_count,
            "contradicted": contradicted_count,
            "baseless": baseless_count,
            "total": total,
            "score": score,
            "has_contradiction": has_contradiction,
            "details": details,
        },
    }
    end_span(span, {"entailed": entailed_count, "contradicted": contradicted_count, "baseless": baseless_count, "score": score})
    return result


def agentic_loop(state: ExperimentState, context: PipelineContext) -> dict:
    """Phase 10 — per-query agentic loop.

    Checks Phase 9 verification score. If below threshold and iterations
    remain, refines the query via DeepSeek-flash and clears stale state
    so the graph loops back through phases 4-9.

    This is the INNER per-query loop only — it answers ONE question by
    retrieving multiple times. It does NOT decide what to test tomorrow.
    """
    trace = _get_trace(state, context)
    verification_result = state.get("verification_result", {})
    score = verification_result.get("score", 0.0)
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 3)
    raw_query = state.get("raw_query", "")
    answer = state.get("answer", "")
    span = start_span(trace, "phase_10_agentic_loop", {"iteration": iteration, "score": score})

    # ── Stopping conditions ───────────────────────────────────────────────
    if score >= 0.75:
        reason = f"score {score:.2f} >= 0.75"
        logger.info(
            "Phase 10: stopping — score=%.2f, iteration=%d/%d, reason=%s",
            score, iteration, max_iterations, reason,
        )
        end_span(span, {"loop_complete": True, "reason": reason, "iteration": iteration, "score": score})
        return {"loop_complete": True}

    if iteration >= max_iterations:
        reason = f"max iterations reached ({iteration}/{max_iterations})"
        logger.info(
            "Phase 10: stopping — score=%.2f, iteration=%d/%d, reason=%s",
            score, iteration, max_iterations, reason,
        )
        return {"loop_complete": True}

    if not answer or answer.startswith("[Phase 8"):
        reason = "error state — no valid answer"
        logger.info(
            "Phase 10: stopping — score=%.2f, iteration=%d/%d, reason=%s",
            score, iteration, max_iterations, reason,
        )
        return {"loop_complete": True}

    # ── Identify failed claims ────────────────────────────────────────────
    failed_claims = [
        d["claim"]
        for d in verification_result.get("details", [])
        if d.get("verdict") != "ENTAILED"
    ]

    if not failed_claims:
        logger.info(
            "Phase 10: stopping — no failed claims despite score %.2f", score,
        )
        return {"loop_complete": True}

    # ── Refine query via DeepSeek-flash ───────────────────────────────────
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("Phase 10: DEEPSEEK_API_KEY not set, stopping")
        return {"loop_complete": True}

    system_prompt = (
        "You are a query refinement assistant for a legal document "
        "retrieval system. A retrieval attempt failed to find evidence "
        "for some claims. Your job is to write a better search query "
        "that targets the missing evidence.\n\n"
        "Rules:\n"
        "- Return ONLY the refined query as plain text\n"
        "- Make it more specific than the original\n"
        "- Focus on the unverified claims\n"
        "- Do not add information not present in the original query "
        "or failed claims\n"
        "- Maximum 2 sentences"
    )

    failed_list = "\n".join(f"- {c}" for c in failed_claims)
    user_prompt = (
        f"Original query: {raw_query}\n\n"
        f"Claims that could not be verified:\n{failed_list}\n\n"
        "Write a refined search query to find evidence for these claims."
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
        refined_query = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Phase 10: query refinement failed (%s), stopping", exc)
        return {"loop_complete": True}

    if not refined_query:
        logger.warning("Phase 10: empty refined query, stopping")
        return {"loop_complete": True}

    logger.info(
        "Phase 10: iterating — score=%.2f, iteration=%d/%d, "
        "refined query: '%s'",
        score, iteration, max_iterations, refined_query[:100],
    )

    # ── Clear stale state and loop back to retrieval ──────────────────────
    result = {
        "rewritten_query": refined_query,
        "iteration": iteration + 1,
        "loop_complete": False,
        "retrieval_bundle": {},
        "fused_result": {},
        "reranked_result": {},
        "context_chunks": [],
        "context_ids": [],
        "claims": [],
        "verification_result": {},
    }
    end_span(span, {"loop_complete": False, "iteration": iteration, "score": score, "refined_query": refined_query})
    return result
