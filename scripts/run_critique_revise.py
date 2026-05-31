"""Surgical critique-revise -- single-pass reasoning-misalignment fix.

Tests whether an ALIGNRAG-style critic can detect and surgically fix
grounded-but-wrong cases (INCORRECT+FAITHFUL) where the model has correct
evidence but reasons to the wrong answer.

NOT a loop (Finding 37: loop dead even repaired). Same evidence in, better
reasoning out. Revision is SURGICAL: string-replace the misaligned
sentence(s), not full-answer regeneration (the Finding-37 leak).

Usage:
    python scripts/run_critique_revise.py
    python scripts/run_critique_revise.py --corpus maud --limit 3
    python scripts/run_critique_revise.py --corpus maud cuad --workers 4
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
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
logger = logging.getLogger("critique_revise")
logger.setLevel(logging.INFO)

# ── Constants ─────────────────────────────────────────────────────────────

FAITH_THRESHOLD = 0.8   # minimum faithfulness to qualify as "grounded-but-wrong"

CORPUS_FILES = {
    "maud": {
        "eval": Path("data/eval_maud_baseline_ctx.jsonl"),
        "judge": Path("data/judge_v2_maud_eval_maud_baseline_ctx.jsonl"),
        "faith": Path("data/faith_maud_eval_maud_baseline_ctx.jsonl"),
    },
    "contractnli": {
        "eval": Path("data/eval_contractnli_baseline_ctx.jsonl"),
        "judge": Path("data/judge_v2_contractnli_eval_contractnli_baseline_ctx.jsonl"),
        "faith": Path("data/faith_contractnli_eval_contractnli_baseline_ctx.jsonl"),
    },
    "cuad": {
        "eval": Path("data/eval_cuad_baseline_ctx.jsonl"),
        "judge": Path("data/judge_v2_cuad_eval_cuad_baseline_ctx.jsonl"),
        "faith": Path("data/faith_cuad_eval_cuad_baseline_ctx.jsonl"),
    },
    "privacyqa": {
        "eval": Path("data/eval_privacyqa_baseline_ctx.jsonl"),
        "judge": Path("data/judge_v2_privacyqa_eval_privacyqa_baseline_ctx.jsonl"),
        "faith": Path("data/faith_privacyqa_eval_privacyqa_baseline_ctx.jsonl"),
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


# ── Pydantic models for critic output ─────────────────────────────────────

class ClaimVerdict(BaseModel):
    claim_index: int = Field(description="0-based index of the claim")
    verdict: str = Field(description="AGREE, DISAGREE_CONTRADICTION, or DISAGREE_WRONG_CITATION")
    original_claim: str = Field(description="The original claim text")
    explanation: str = Field(
        description="Why the claim is correct (AGREE) or what's wrong (DISAGREE). "
                    "For DISAGREE_CONTRADICTION: which chunk contradicts the claim and how. "
                    "For DISAGREE_WRONG_CITATION: which chunk better supports a different conclusion."
    )
    better_chunk_id: str = Field(
        default="",
        description="For DISAGREE: the chunk_id that contradicts or better supports. Empty for AGREE."
    )
    corrected_claim: str = Field(
        default="",
        description="For DISAGREE: the corrected claim text aligned with the evidence. Empty for AGREE."
    )
    corrected_cited_text: str = Field(
        default="",
        description="For DISAGREE: verbatim excerpt from better_chunk_id supporting the corrected claim. Empty for AGREE."
    )


class CriticOutput(BaseModel):
    verdicts: list[ClaimVerdict] = Field(
        description="One verdict per claim. AGREE if the claim is the best-supported "
                    "conclusion given all retrieved evidence. DISAGREE_CONTRADICTION if "
                    "a retrieved chunk contradicts the claim. DISAGREE_WRONG_CITATION if "
                    "a different retrieved chunk better supports a different conclusion."
    )


# ── Verification (reused from run_mini_e2e.py) ───────────────────────────

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


def run_verification(claims, chunk_ids, chunk_texts):
    """Deterministic Phase 9 verification -- mirrors nodes.py."""
    if not claims:
        return {"entailed": 0, "contradicted": 0, "baseless": 0,
                "total": 0, "score": 0.0, "has_contradiction": False, "details": []}

    chunk_text_by_id = dict(zip(chunk_ids, chunk_texts))
    total = len(claims)
    entailed_count = 0
    contradicted_count = 0
    baseless_count = 0
    details = []

    for claim in claims:
        cited_chunk_id = claim.get("cited_chunk_id", "")
        cited_text = claim.get("cited_text", "").strip()
        claim_text = claim.get("claim", "").strip()

        if cited_chunk_id not in chunk_text_by_id:
            baseless_count += 1
            details.append({"claim": claim_text, "cited_chunk_id": cited_chunk_id,
                            "verdict": "BASELESS", "reason": "chunk_id not in context"})
            continue

        chunk_content = chunk_text_by_id[cited_chunk_id]

        if _detect_contradiction(claim_text, chunk_content):
            contradicted_count += 1
            details.append({"claim": claim_text, "cited_chunk_id": cited_chunk_id,
                            "verdict": "CONTRADICTED", "reason": "chunk explicitly contradicts claim"})
            continue

        norm_cited = _normalize_text(cited_text)
        norm_chunk = _normalize_text(chunk_content)

        if norm_cited and norm_cited in norm_chunk:
            entailed_count += 1
            details.append({"claim": claim_text, "cited_chunk_id": cited_chunk_id,
                            "verdict": "ENTAILED", "reason": "exact"})
            continue

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
        "entailed": entailed_count, "contradicted": contradicted_count,
        "baseless": baseless_count, "total": total, "score": score,
        "has_contradiction": contradicted_count > 0, "details": details,
    }


# ── Population loading ────────────────────────────────────────────────────

def load_population(corpus: str) -> list[dict]:
    """Load INCORRECT+FAITHFUL queries from existing eval + judge + faith results."""
    files = CORPUS_FILES[corpus]
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {key} file: {path}")

    # Load eval results
    with files["eval"].open(encoding="utf-8") as f:
        evals = {json.loads(line)["query_id"]: json.loads(line) for line in f if line.strip()}

    # Load judge verdicts
    with files["judge"].open(encoding="utf-8") as f:
        judges = {json.loads(line)["query_id"]: json.loads(line) for line in f if line.strip()}

    # Load faithfulness scores
    with files["faith"].open(encoding="utf-8") as f:
        faiths = {json.loads(line)["query_id"]: json.loads(line) for line in f if line.strip()}

    # Filter: INCORRECT + FAITHFUL (>= threshold)
    population = []
    for qid, judge in judges.items():
        if judge.get("verdict") != "INCORRECT":
            continue
        faith = faiths.get(qid, {})
        if faith.get("faithfulness_score", 0.0) < FAITH_THRESHOLD:
            continue
        if qid not in evals:
            continue
        rec = evals[qid]
        rec["_judge_reason"] = judge.get("reason", "")
        rec["_faithfulness_score"] = faith.get("faithfulness_score", 0.0)
        population.append(rec)

    return sorted(population, key=lambda r: r["query_id"])


# ── Critic detection ──────────────────────────────────────────────────────

def run_critic(record: dict) -> CriticOutput:
    """Detect reasoning misalignments by checking each claim against ALL
    retrieved chunks (not just the cited chunk).

    Catches two error types:
    - DISAGREE_CONTRADICTION: a retrieved chunk contradicts the claim
    - DISAGREE_WRONG_CITATION: a different retrieved chunk better supports
      a different conclusion than the one the claim draws

    Scope: can only detect misalignment among RETRIEVED evidence. True
    access misses (right evidence never retrieved) are out of reach.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    query = record["query"]
    answer = record["answer"]
    claims = record["claims"]
    context_chunks = record["context_chunks"]

    # Build claims list with cited info
    claims_str = ""
    for i, claim in enumerate(claims):
        claims_str += (
            f"\n[Claim {i}]\n"
            f"  Assertion: {claim['claim']}\n"
            f"  Currently cited chunk: {claim.get('cited_chunk_id', '')}\n"
            f"  Currently cited text: {claim.get('cited_text', '')[:300]}\n"
        )

    # Build full context — ALL chunks, the critic's evidence base
    context_str = "\n---\n".join(
        f"[{c['chunk_id']}]\n{c['content']}" for c in context_chunks
    )

    system_prompt = (
        "You are a legal reasoning auditor. For each claim below, check it "
        "against the ENTIRE set of retrieved document chunks (not just the "
        "chunk the claim cites). Your job is to detect REASONING MISALIGNMENT "
        "where the claim's conclusion diverges from the best available evidence "
        "in the retrieved chunks.\n\n"
        "For EACH claim, output one of three verdicts:\n\n"
        "AGREE: The claim is the best-supported conclusion given ALL retrieved "
        "evidence, and the citation is correct. Leave corrected fields empty.\n\n"
        "DISAGREE_CONTRADICTION: A retrieved chunk contains evidence that "
        "CONTRADICTS or UNDERMINES the claim. Flag this when:\n"
        "  a) The claim says 'not found / not present / no provision' but a "
        "chunk DOES contain the relevant term, definition, or provision -- even "
        "if the claim argues it means something different. If the query asks "
        "'is there a tail provision?' and a chunk defines 'Tail Period', that "
        "IS a tail provision regardless of how the claim interprets it.\n"
        "  b) The claim makes an UNSUPPORTED NEGATIVE INFERENCE: concluding "
        "'X does not relate to Y' or 'X is not a type of Y' when the evidence "
        "only defines X without explicitly excluding Y. The burden of proof is "
        "on the negative claim -- if the evidence does not explicitly say "
        "'X is not Y', the claim should not conclude that.\n"
        "  c) The claim draws a conclusion that the evidence does not support, "
        "even if the claim and its cited chunk are internally consistent.\n"
        "Provide the corrected claim aligned with the evidence.\n\n"
        "DISAGREE_WRONG_CITATION: The claim cites the wrong chunk. A DIFFERENT "
        "retrieved chunk better supports a different (more accurate) conclusion. "
        "The claim and its cited chunk may agree with each other, but the claim "
        "is wrong because better evidence exists in another retrieved chunk. "
        "Common pattern: claim cites chunk describing Section A, but the query "
        "asks about Section B which is in a different chunk. Provide the "
        "corrected claim re-pointed to the better chunk.\n\n"
        "IMPORTANT:\n"
        "- You MUST output exactly one verdict per claim (same count as input)\n"
        "- 'corrected_cited_text' must be VERBATIM from the chunk (copy exactly)\n"
        "- 'better_chunk_id' must be one of the chunk IDs listed in context\n"
        "- Be AGGRESSIVE about detecting negative claims that dismiss present "
        "evidence. When a query asks 'is there X?' and the evidence contains "
        "something called X (or a close variant), the answer should be yes.\n"
        "- If the right evidence is simply NOT in any retrieved chunk, verdict "
        "is AGREE (the claim is correct given available evidence -- the problem "
        "is retrieval, not reasoning)"
    )

    user_prompt = (
        f"Query: {query}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Claims to audit:{claims_str}\n\n"
        f"Full retrieved context (ALL chunks):\n{context_str}\n\n"
        "Audit each claim against the full context. Output one verdict per claim."
    )

    import instructor
    from openai import OpenAI

    client = instructor.from_openai(
        OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        response_model=CriticOutput,
        max_retries=3,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )

    return response


# ── Claim correction + constrained re-synthesis ───────────────────────────

def apply_corrections(record: dict, critic_output: CriticOutput) -> dict:
    """Apply claim-level corrections from the critic, then re-synthesize
    the answer from corrected claims (one LLM call).

    Corrections at the CLAIM level:
    - For DISAGREE verdicts: replace claim text + re-point citation
    - For AGREE verdicts: keep claim unchanged

    Then: single re-synthesis constrained by corrected claims.
    NOT free-form regeneration over evidence (the Finding-37 leak).
    """
    disagrees = [v for v in critic_output.verdicts if v.verdict.startswith("DISAGREE")]
    if not disagrees:
        return record

    revised = copy.deepcopy(record)
    claims = revised["claims"]
    valid_chunk_ids = {c["chunk_id"] for c in revised["context_chunks"]}

    n_corrected = 0
    n_skipped = 0
    correction_details = []

    for v in critic_output.verdicts:
        detail = {
            "claim_index": v.claim_index,
            "verdict": v.verdict,
            "original_claim": v.original_claim,
            "explanation": v.explanation,
            "applied": False,
        }

        if v.verdict == "AGREE":
            correction_details.append(detail)
            continue

        # Validate claim_index
        if v.claim_index < 0 or v.claim_index >= len(claims):
            logger.warning("[%s] Invalid claim_index %d (n_claims=%d), skipping",
                          record["query_id"], v.claim_index, len(claims))
            n_skipped += 1
            correction_details.append(detail)
            continue

        # Validate better_chunk_id
        if v.better_chunk_id and v.better_chunk_id not in valid_chunk_ids:
            logger.warning("[%s] Corrected chunk_id %r not in context, skipping",
                          record["query_id"], v.better_chunk_id)
            n_skipped += 1
            correction_details.append(detail)
            continue

        # Validate corrected claim is non-empty
        if not v.corrected_claim:
            logger.warning("[%s] Empty corrected_claim for claim %d, skipping",
                          record["query_id"], v.claim_index)
            n_skipped += 1
            correction_details.append(detail)
            continue

        # Apply claim-level correction
        claims[v.claim_index] = {
            "claim": v.corrected_claim,
            "cited_chunk_id": v.better_chunk_id or claims[v.claim_index]["cited_chunk_id"],
            "cited_text": v.corrected_cited_text or claims[v.claim_index].get("cited_text", ""),
        }

        detail["applied"] = True
        detail["corrected_claim"] = v.corrected_claim
        detail["better_chunk_id"] = v.better_chunk_id
        n_corrected += 1
        logger.info("[%s] Corrected claim %d (%s): %s -> %s",
                    record["query_id"], v.claim_index, v.verdict,
                    v.original_claim[:50], v.corrected_claim[:50])

        correction_details.append(detail)

    if n_corrected == 0:
        # All DISAGREE claims were skipped due to validation failures
        revised["n_disagree"] = len(disagrees)
        revised["n_corrected"] = 0
        revised["n_skipped"] = n_skipped
        revised["corrections"] = correction_details
        revised["original_answer"] = record["answer"]
        return revised

    revised["claims"] = claims

    # ── Constrained re-synthesis from corrected claims ────────────────
    # One LLM call: write answer prose incorporating the corrected claims.
    # The claims CONSTRAIN the synthesis — not free-form evidence reasoning.
    resynthesized = _resynthesize_from_claims(
        record["query"], claims, record["context_chunks"],
    )
    if resynthesized:
        revised["answer"] = resynthesized
    # else: keep original answer (synthesis failed)

    revised["original_answer"] = record["answer"]
    revised["n_disagree"] = len(disagrees)
    revised["n_corrected"] = n_corrected
    revised["n_skipped"] = n_skipped
    revised["corrections"] = correction_details

    return revised


def _resynthesize_from_claims(
    query: str,
    corrected_claims: list[dict],
    context_chunks: list[dict],
) -> str:
    """Single re-synthesis constrained by corrected claims.

    The claims dictate the facts; the synthesis produces coherent prose
    that incorporates them. NOT free-form reasoning over evidence (the
    Finding-37 leak). The claims are the guardrail.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return ""

    claims_list = "\n".join(
        f"- {c['claim']} [source: {c.get('cited_chunk_id', '?')}]"
        for c in corrected_claims
    )
    context_str = "\n---\n".join(
        f"[{c['chunk_id']}]\n{c['content']}" for c in context_chunks
    )

    system_prompt = (
        "You are a legal document analyst. Write a coherent answer to the "
        "query that incorporates ALL of the verified claims below. The claims "
        "are authoritative -- do not contradict, omit, or reinterpret them. "
        "Use the evidence chunks for context and phrasing.\n\n"
        "Rules:\n"
        "- Every claim listed below must appear in your answer\n"
        "- Do not add conclusions beyond what the claims state\n"
        "- Be specific, cite details from the evidence\n"
        "- Produce a coherent narrative, not a bulleted list"
    )
    user_prompt = (
        f"Query: {query}\n\n"
        f"Verified claims to incorporate:\n{claims_list}\n\n"
        f"Evidence chunks:\n{context_str}\n\n"
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
        logger.warning("Re-synthesis failed: %s", exc)
        return ""


# ── Contradiction false-positive audit ────────────────────────────────────

def _audit_contradictions(
    verification_details: list[dict],
    claims: list[dict],
    ctx_ids: list[str],
    ctx_texts: list[str],
) -> tuple[list[dict], int]:
    """Audit CONTRADICTED verdicts for Phase-9 negation false positives.

    A false positive is: the cited_text IS a substring (or high token overlap)
    of the cited chunk, but the contradiction detector fired on incidental
    negation+term-overlap in an unrelated sentence of the same chunk.

    Returns (audit_details, n_false_positives).
    """
    chunk_by_id = dict(zip(ctx_ids, ctx_texts))
    n_fp = 0
    audit_details = []

    for i, detail in enumerate(verification_details):
        if detail["verdict"] != "CONTRADICTED":
            continue

        claim = claims[i] if i < len(claims) else {}
        cited_chunk_id = claim.get("cited_chunk_id", "")
        cited_text = claim.get("cited_text", "").strip()
        chunk_content = chunk_by_id.get(cited_chunk_id, "")

        # Check if the cited_text IS actually supported by the chunk
        # (substring or high token overlap) despite the contradiction flag
        norm_cited = _normalize_text(cited_text)
        norm_chunk = _normalize_text(chunk_content)

        is_substring = bool(norm_cited) and norm_cited in norm_chunk

        cited_tokens = set(norm_cited.split()) - _STOPWORDS if norm_cited else set()
        chunk_tokens = set(norm_chunk.split()) - _STOPWORDS
        overlap = cited_tokens & chunk_tokens
        token_ratio = len(overlap) / max(len(cited_tokens), 1)
        semantically_supported = is_substring or token_ratio >= 0.75

        is_false_positive = semantically_supported

        audit = {
            "claim_index": i,
            "claim_text": detail["claim"][:100],
            "cited_chunk_id": cited_chunk_id,
            "verdict": "CONTRADICTED",
            "cited_text_is_substring": is_substring,
            "token_overlap_ratio": round(token_ratio, 2),
            "semantically_supported": semantically_supported,
            "is_false_positive": is_false_positive,
        }
        audit_details.append(audit)

        if is_false_positive:
            n_fp += 1
            logger.info("  FP audit: claim %d CONTRADICTED but cited_text IS supported "
                       "(substring=%s, overlap=%.2f) -- negation false positive",
                       i, is_substring, token_ratio)

    return audit_details, n_fp


# ── Per-query runner ──────────────────────────────────────────────────────

def process_query(record: dict) -> tuple[dict, dict]:
    """Run critic + correction + re-synthesis on a single query.

    Returns (baseline_record, revised_record).
    """
    qid = record["query_id"]
    t0 = time.perf_counter()

    # Run critic — check each claim against ALL retrieved chunks
    try:
        critic_output = _run_with_retry(lambda: run_critic(record))
    except Exception as exc:
        logger.error("[%s] Critic failed: %s", qid, exc)
        critic_output = CriticOutput(verdicts=[])

    n_disagree = sum(1 for v in critic_output.verdicts if v.verdict.startswith("DISAGREE"))

    # Apply corrections + re-synthesis
    if n_disagree > 0:
        revised = _run_with_retry(lambda: apply_corrections(record, critic_output))
    else:
        revised = copy.deepcopy(record)
        revised["n_disagree"] = 0
        revised["n_corrected"] = 0
        revised["n_skipped"] = 0
        revised["corrections"] = [
            {"claim_index": v.claim_index, "verdict": v.verdict,
             "original_claim": v.original_claim, "explanation": v.explanation,
             "applied": False}
            for v in critic_output.verdicts
        ]
        revised["original_answer"] = record["answer"]

    # Re-verify revised claims + audit CONTRADICTED for negation false positives
    ctx_ids = [c["chunk_id"] for c in revised["context_chunks"]]
    ctx_texts = [c["content"] for c in revised["context_chunks"]]
    verification = run_verification(revised["claims"], ctx_ids, ctx_texts)
    revised["grounding_score_raw"] = round(verification["score"], 4)

    # Audit every CONTRADICTED verdict: is it a Phase-9 negation false positive?
    audited_details, n_fp = _audit_contradictions(
        verification["details"], revised["claims"], ctx_ids, ctx_texts,
    )
    revised["grounding_audit"] = audited_details
    revised["n_contradiction_false_positives"] = n_fp

    # Corrected grounding: re-score treating confirmed false positives as ENTAILED
    total = verification["total"]
    entailed_corrected = verification["entailed"] + n_fp
    revised["grounding_score_corrected"] = round(entailed_corrected / total, 4) if total > 0 else 0.0
    # Raw (confounded) grounding for the record
    revised["grounding_score"] = revised["grounding_score_raw"]

    latency_ms = int((time.perf_counter() - t0) * 1000)
    revised["latency_ms"] = latency_ms

    # Build baseline record (original, for comparison)
    baseline = copy.deepcopy(record)
    baseline_vr = run_verification(record["claims"], ctx_ids, ctx_texts)
    baseline["grounding_score"] = round(baseline_vr["score"], 4)
    baseline["grounding_score_raw"] = baseline["grounding_score"]
    baseline["grounding_score_corrected"] = baseline["grounding_score"]
    baseline["n_contradiction_false_positives"] = 0
    baseline["n_disagree"] = 0
    baseline["n_corrected"] = 0

    fp = revised.get("n_contradiction_false_positives", 0)
    fp_tag = f" [{fp} FP]" if fp > 0 else ""
    logger.info("[%s] disagree=%d corrected=%d grounding: %.2f -> %.2f (corrected: %.2f)%s (%dms)",
                qid, n_disagree, revised.get("n_corrected", 0),
                baseline["grounding_score"], revised["grounding_score_raw"],
                revised["grounding_score_corrected"], fp_tag, latency_ms)

    return baseline, revised


# ── Per-corpus runner ─────────────────────────────────────────────────────

def run_corpus(corpus: str, output_dir: Path, workers: int = 4, limit: int | None = None):
    """Run critique-revise on all INCORRECT+FAITHFUL queries for a corpus."""
    population = load_population(corpus)
    if limit:
        population = population[:limit]

    n = len(population)
    print(f"\n{'=' * 60}")
    print(f"CORPUS: {corpus} | {n} INCORRECT+FAITHFUL queries")
    print(f"{'=' * 60}")

    if n == 0:
        print("  No qualifying queries found.")
        return

    baselines = []
    reviseds = []
    errors = 0

    with ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
        futures = {executor.submit(process_query, rec): rec for rec in population}
        for f in as_completed(futures):
            try:
                baseline, revised = f.result()
                baselines.append(baseline)
                reviseds.append(revised)

                qid = revised["query_id"]
                nd = revised.get("n_disagree", 0)
                nc = revised.get("n_corrected", 0)
                ft = revised.get("failure_type", "?")
                tag = f"CORRECTED({nc})" if nc > 0 else (f"DISAGREE({nd})" if nd > 0 else "NO-OP")
                fp = revised.get("n_contradiction_false_positives", 0)
                fp_tag = f" [FP={fp}]" if fp > 0 else ""
                print(f"  [{qid}] {ft} {tag} "
                      f"grounding={revised['grounding_score_raw']:.2f} "
                      f"(corrected={revised['grounding_score_corrected']:.2f}){fp_tag}")
            except Exception as exc:
                errors += 1
                rec = futures[f]
                print(f"  [{rec['query_id']}] FAILED: {exc}", file=sys.stderr)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / f"{corpus}_baseline.jsonl"
    revised_path = output_dir / f"{corpus}_revised.jsonl"

    with baseline_path.open("w", encoding="utf-8") as f:
        for r in sorted(baselines, key=lambda x: x["query_id"]):
            f.write(json.dumps(r) + "\n")
    with revised_path.open("w", encoding="utf-8") as f:
        for r in sorted(reviseds, key=lambda x: x["query_id"]):
            f.write(json.dumps(r) + "\n")

    print(f"\n{corpus}: {len(reviseds)} records ({errors} errors)")
    print(f"  Baseline -> {baseline_path}")
    print(f"  Revised  -> {revised_path}")

    # Summary
    n_disagree = sum(1 for r in reviseds if r.get("n_disagree", 0) > 0)
    n_corrected = sum(1 for r in reviseds if r.get("n_corrected", 0) > 0)
    n_noop = sum(1 for r in reviseds if r.get("n_disagree", 0) == 0)
    ft_dist = Counter(r.get("failure_type", "?") for r in reviseds)

    print(f"  Critic found DISAGREE: {n_disagree}/{n}")
    print(f"  Corrections applied: {n_corrected}/{n}")
    print(f"  All-AGREE (no-op): {n_noop}/{n}")
    print(f"  Failure types: {dict(ft_dist)}")

    # DRM vs non-DRM breakdown for no-ops
    noop_drm = sum(1 for r in reviseds
                   if r.get("n_disagree", 0) == 0 and r.get("failure_type") == "DRM")
    noop_nondrm = n_noop - noop_drm
    print(f"  No-op breakdown: DRM={noop_drm} non-DRM={noop_nondrm}")

    # Grounding confound audit
    total_fp = sum(r.get("n_contradiction_false_positives", 0) for r in reviseds)
    corrected_recs = [r for r in reviseds if r.get("n_corrected", 0) > 0]
    if corrected_recs:
        avg_raw = sum(r["grounding_score_raw"] for r in corrected_recs) / len(corrected_recs)
        avg_corr = sum(r["grounding_score_corrected"] for r in corrected_recs) / len(corrected_recs)
        print(f"  GROUNDING CONFOUND (corrected queries only):")
        print(f"    Raw Phase-9 grounding: {avg_raw:.3f} (confounded by negation FP)")
        print(f"    Audited grounding:     {avg_corr:.3f} (FP removed)")
        print(f"    Confound size:         {avg_corr - avg_raw:+.3f} ({total_fp} false positives)")
        print(f"    NOTE: Holistic faithfulness judge (Layer-2) is the trustworthy signal.")
        print(f"          Phase-9 grounding is known-confounded on corrected claims.")

    # Scope reminder
    print(f"  SCOPE: critic arbitrates among RETRIEVED evidence only. "
          f"True access misses (DRM/CBF where evidence never retrieved) are out of reach.")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Surgical critique-revise experiment")
    p.add_argument("--corpus", nargs="*", default=list(CORPUS_FILES.keys()),
                   choices=list(CORPUS_FILES.keys()))
    p.add_argument("--output-dir", type=Path, default=Path("data/critique_revise"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit to first N queries per corpus (for dry run)")
    args = p.parse_args()

    print("Surgical Critique-Revise Experiment")
    print(f"Corpora: {args.corpus}")
    print(f"Output: {args.output_dir}/")
    if args.limit:
        print(f"LIMIT: first {args.limit} per corpus (dry run)")

    for corpus in args.corpus:
        run_corpus(corpus, args.output_dir, workers=args.workers, limit=args.limit)

    # Cross-corpus summary
    print(f"\n{'=' * 60}")
    print("CROSS-CORPUS SUMMARY")
    print(f"{'=' * 60}")

    all_revised = []
    for corpus in args.corpus:
        path = args.output_dir / f"{corpus}_revised.jsonl"
        if path.exists():
            with path.open() as f:
                for line in f:
                    if line.strip():
                        all_revised.append(json.loads(line))

    if all_revised:
        total = len(all_revised)
        disagree = sum(1 for r in all_revised if r.get("n_disagree", 0) > 0)
        corrected = sum(1 for r in all_revised if r.get("n_corrected", 0) > 0)
        noop = total - disagree
        print(f"Total: {total} queries")
        print(f"Critic DISAGREE: {disagree}/{total} ({disagree/total*100:.0f}%)")
        print(f"Corrections applied: {corrected}/{total} ({corrected/total*100:.0f}%)")
        print(f"All-AGREE (no-op): {noop}/{total} ({noop/total*100:.0f}%)")

    print("\nNext steps:")
    print("  1. Run judge_answers_v2.py on baseline and revised outputs")
    print("  2. Run judge_faithfulness.py on both (holistic)")
    print("  3. Compare: by-name flips, CORRECT+FAITHFUL, regressions, drift")


if __name__ == "__main__":
    main()
