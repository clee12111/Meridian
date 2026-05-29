"""Faithfulness judge — RAGAS definition: fraction of answer claims entailed
by the FULL retrieved context.

Layer-2 LLM-judged metric, reported alongside correctness. Does NOT read,
write, or modify any taxonomy/failure_type field — emits its own output file.

Requires eval JSONL files with context_chunks (added to harness Part 1).

Usage:
    python scripts/judge_faithfulness.py --corpus cuad --input data/eval_cuad.jsonl
    python scripts/judge_faithfulness.py --corpus contractnli --input data/f1.jsonl data/f2.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

# Same pinned model + settings as the correctness judge (judge_answers_v2.py)
JUDGE_MODEL = "deepseek-v4-flash"

ENTAILMENT_PROMPT = """\
You are a factual entailment judge. Your task is to determine whether a \
specific claim is ENTAILED by the provided context.

CONTEXT (retrieved chunks the system had access to):
{context}

CLAIM:
{claim}

Is this claim entailed (supported/implied) by the context above?

A claim is ENTAILED if the context contains information that directly \
supports or logically implies the claim. Paraphrasing counts — the claim \
does not need to use the exact same words.

A claim is NOT_ENTAILED if the context does not contain sufficient \
information to support the claim, or if the claim contradicts the context, \
or if the claim introduces information not present in the context.

Return ONLY a JSON object (no markdown, no commentary):
{{"verdict": "ENTAILED" or "NOT_ENTAILED", "reason": "one sentence"}}"""

_RETRY_WAITS = [2, 5, 10]


def _parse_verdict(raw: str) -> dict:
    """Parse judge response — same logic as judge_answers_v2.py."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "PARSE_ERROR").upper()
        reason = parsed.get("reason", "")
        if verdict not in ("ENTAILED", "NOT_ENTAILED"):
            return {"verdict": "PARSE_ERROR", "reason": f"Unknown: {verdict}"}
        return {"verdict": verdict, "reason": reason}
    except json.JSONDecodeError:
        m = re.search(r'"verdict"\s*:\s*"(ENTAILED|NOT_ENTAILED)"', raw, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()
            m2 = re.search(r'"reason"\s*:\s*"([^"]*)', raw)
            reason = m2.group(1) if m2 else "(truncated)"
            return {"verdict": verdict, "reason": reason}
        return {"verdict": "PARSE_ERROR", "reason": "Failed to parse JSON"}


def _judge_claim(claim_text: str, context_str: str) -> dict:
    """Judge one claim against context via DeepSeek — same retry as correctness judge."""
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    prompt = ENTAILMENT_PROMPT.format(
        context=context_str[:6000],
        claim=claim_text,
    )

    for attempt, wait in enumerate(_RETRY_WAITS, start=1):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = (resp.choices[0].message.content or "").strip()
            if not raw:
                raise RuntimeError("Empty response from DeepSeek")
            return _parse_verdict(raw)
        except Exception as exc:
            cls = type(exc).__name__
            msg = str(exc)
            retryable = any(
                kw in cls for kw in
                ("RateLimit", "ServiceUnavailable", "ServerError", "Overloaded")
            ) or "529" in msg or "Empty response" in msg
            if not retryable:
                raise
            if attempt <= len(_RETRY_WAITS):
                time.sleep(wait)

    # Final attempt
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_verdict(raw) if raw else {"verdict": "PARSE_ERROR", "reason": "Empty after retries"}


def judge_query(record: dict) -> dict:
    """Judge all claims in one query record. Returns faithfulness result."""
    qid = record["query_id"]
    claims = record.get("claims", [])
    context_chunks = record.get("context_chunks", [])

    if not claims:
        return {
            "query_id": qid,
            "faithfulness_score": 0.0,
            "n_claims": 0,
            "n_entailed": 0,
            "per_claim_verdicts": [],
        }

    if not context_chunks:
        return {
            "query_id": qid,
            "faithfulness_score": 0.0,
            "n_claims": len(claims),
            "n_entailed": 0,
            "per_claim_verdicts": [
                {"claim": c.get("claim", ""), "verdict": "NOT_ENTAILED",
                 "reason": "no context_chunks in record"}
                for c in claims
            ],
        }

    # Build full context string (all chunks the model saw)
    context_str = "\n---\n".join(
        f"[{c['chunk_id']}]\n{c['content']}"
        for c in context_chunks
    )

    per_claim: list[dict] = []
    n_entailed = 0

    for claim in claims:
        claim_text = claim.get("claim", "")
        if not claim_text.strip():
            per_claim.append({"claim": "", "verdict": "NOT_ENTAILED", "reason": "empty claim"})
            continue

        verdict = _judge_claim(claim_text, context_str)
        per_claim.append({
            "claim": claim_text,
            "cited_chunk_id": claim.get("cited_chunk_id", ""),
            **verdict,
        })
        if verdict["verdict"] == "ENTAILED":
            n_entailed += 1

    n_claims = len(claims)
    score = n_entailed / n_claims if n_claims > 0 else 0.0

    return {
        "query_id": qid,
        "faithfulness_score": round(score, 4),
        "n_claims": n_claims,
        "n_entailed": n_entailed,
        "per_claim_verdicts": per_claim,
    }


def judge_file(filepath: Path, workers: int = 14) -> list[dict]:
    """Judge all records in a file."""
    records = []
    with filepath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Check that context_chunks is present
    missing_ctx = sum(1 for r in records if not r.get("context_chunks"))
    if missing_ctx:
        print(f"  WARNING: {missing_ctx}/{len(records)} records lack context_chunks")

    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge_query, r): r for r in records}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            qid = result["query_id"]
            score = result["faithfulness_score"]
            n = result["n_claims"]
            ne = result["n_entailed"]
            sys.stdout.write(f"  [{qid}] faith={score:.2f} ({ne}/{n} claims)\n")
            sys.stdout.flush()

    return results


def main():
    p = argparse.ArgumentParser(description="Faithfulness judge (RAGAS definition)")
    p.add_argument("--corpus", required=True,
                   choices=["contractnli", "privacyqa", "cuad", "maud"],
                   help="Corpus name (for output naming only)")
    p.add_argument("--input", nargs="+", type=Path, required=True)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    all_summaries = []

    for filepath in args.input:
        if not filepath.exists():
            print(f"Skipping {filepath} (not found)")
            continue

        n_records = sum(1 for line in filepath.open(encoding="utf-8") if line.strip())
        print(f"\nJudging faithfulness: {filepath.name} ({n_records} records)...")
        results = judge_file(filepath, workers=args.workers)

        n = len(results)
        scores = [r["faithfulness_score"] for r in results]
        mean_faith = sum(scores) / n if n > 0 else 0.0
        perfect = sum(1 for s in scores if s == 1.0)
        zero = sum(1 for s in scores if s == 0.0)

        print(f"\n  {filepath.name}:")
        print(f"    Mean faithfulness: {mean_faith:.3f}")
        print(f"    Perfect (1.0):     {perfect}/{n} ({perfect/n*100:.1f}%)")
        print(f"    Zero (0.0):        {zero}/{n} ({zero/n*100:.1f}%)")
        total_claims = sum(r["n_claims"] for r in results)
        total_entailed = sum(r["n_entailed"] for r in results)
        print(f"    Claims: {total_entailed}/{total_claims} entailed "
              f"({total_entailed/total_claims*100:.1f}%)" if total_claims > 0 else "")

        all_summaries.append({
            "file": filepath.name, "n": n,
            "mean_faithfulness": round(mean_faith, 4),
            "perfect": perfect, "zero": zero,
        })

        # Save per-query results
        out_path = args.output_dir / f"faith_{args.corpus}_{filepath.stem}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in sorted(results, key=lambda x: x["query_id"]):
                f.write(json.dumps(r) + "\n")
        print(f"    Saved: {out_path}")

    if len(all_summaries) > 1:
        print(f"\n{'File':<50} {'Mean faith':>10} {'Perfect':>8} {'Zero':>6}")
        print("-" * 76)
        for s in all_summaries:
            print(f"  {s['file']:<48} {s['mean_faithfulness']:>9.3f} "
                  f"{s['perfect']:>7} {s['zero']:>5}")


if __name__ == "__main__":
    main()
