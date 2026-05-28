"""Answer correctness judge for ContractNLI eval results.

Reads saved JSONL results (no pipeline re-runs), judges each answer
with DeepSeek-flash, reports end-to-end pass rate.

This is an LLM-judged metric, separate from the deterministic span
taxonomy. It answers: did the system reach the correct conclusion?

Ground truth: all 194 ContractNLI queries are "Does the document [X]?"
with spans present — the correct answer is always YES. The judge checks
whether the system answered affirmatively about the correct document.

Usage:
    python scripts/judge_answers.py data/eval_results_v2_sac_cc03_singleshot.jsonl
    python scripts/judge_answers.py *.jsonl   # judge multiple files
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are a legal NDA analysis judge. Your task is to evaluate whether a \
system's answer correctly addresses a yes/no question about a specific \
Non-Disclosure Agreement.

GROUND TRUTH: For this benchmark, the correct answer is always YES — \
the document DOES contain the clause or provision asked about. The \
relevant evidence exists in the contract.

QUERY:
{query}

SYSTEM ANSWER:
{answer}

SYSTEM CITED TEXT:
{cited_text}

CORRECT DOCUMENT: {correct_doc}
DOCUMENT(S) THE SYSTEM CITED: {cited_docs}

Judge the system's answer on these criteria:
1. Did the system reach the correct conclusion (YES, the document does \
contain this provision)?
2. Did the system cite the correct document (not a different NDA)?
3. Is the reasoning coherent and consistent with the conclusion?

Return ONLY a JSON object (no markdown, no commentary):
{{"verdict": "CORRECT" or "INCORRECT" or "PARTIAL", "reason": "one sentence"}}

Verdict guide:
- CORRECT: System answered YES about the correct document with coherent reasoning
- INCORRECT: System answered NO, or answered about the wrong document, \
or gave a contradictory/incoherent answer
- PARTIAL: System hedged significantly, or answered YES but with major \
caveats that undermine the conclusion, or cited the wrong document but \
reached the right conclusion anyway"""

_RETRY_WAITS = [2, 5, 10]


def _call_judge(query: str, answer: str, claims: list[dict],
                correct_doc: str) -> dict:
    """Call DeepSeek-flash to judge one answer. Returns parsed verdict."""
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Build cited text and cited docs from claims
    cited_texts = []
    cited_doc_ids = set()
    for c in claims:
        if c.get("cited_text"):
            cited_texts.append(c["cited_text"][:200])
        if c.get("cited_chunk_id"):
            cited_doc_ids.add(c["cited_chunk_id"].split("#")[0])

    cited_text_str = "\n---\n".join(cited_texts) if cited_texts else "(none)"
    cited_docs_str = ", ".join(sorted(cited_doc_ids)) if cited_doc_ids else "(none)"

    prompt = JUDGE_PROMPT.format(
        query=query,
        answer=answer[:1500],
        cited_text=cited_text_str[:1000],
        correct_doc=correct_doc,
        cited_docs=cited_docs_str,
    )

    # Retry on transient errors
    for attempt, wait in enumerate(_RETRY_WAITS, start=1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_verdict(raw)
        except Exception as exc:
            cls = type(exc).__name__
            msg = str(exc)
            retryable = any(
                kw in cls for kw in
                ("RateLimit", "ServiceUnavailable", "ServerError", "Overloaded")
            ) or "529" in msg
            if not retryable:
                raise
            if attempt <= len(_RETRY_WAITS):
                time.sleep(wait)

    # Final attempt
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    raw = resp.choices[0].message.content.strip()
    return _parse_verdict(raw)


def _parse_verdict(raw: str) -> dict:
    """Parse judge response into {verdict, reason}."""
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "PARSE_ERROR").upper()
        reason = parsed.get("reason", "")
        if verdict not in ("CORRECT", "INCORRECT", "PARTIAL"):
            return {"verdict": "PARSE_ERROR", "reason": f"Unknown verdict: {verdict}", "raw": raw}
        return {"verdict": verdict, "reason": reason}
    except json.JSONDecodeError:
        # Try to extract verdict from truncated/malformed JSON
        m = re.search(r'"verdict"\s*:\s*"(CORRECT|INCORRECT|PARTIAL)"', raw, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()
            m2 = re.search(r'"reason"\s*:\s*"([^"]*)', raw)
            reason = m2.group(1) if m2 else "(truncated)"
            return {"verdict": verdict, "reason": reason}
        return {"verdict": "PARSE_ERROR", "reason": "Failed to parse JSON", "raw": raw}


def judge_file(filepath: Path, correct_docs: dict[str, str],
               workers: int = 8) -> list[dict]:
    """Judge all answers in a JSONL file. Returns list of result dicts."""
    records = []
    with filepath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"  No records in {filepath}")
        return []

    results = []
    errors = 0

    def judge_one(r: dict) -> dict:
        query_id = r["query_id"]
        correct_doc = correct_docs.get(query_id, "unknown")
        try:
            verdict = _call_judge(
                query=r["query"],
                answer=r.get("answer", ""),
                claims=r.get("claims", []),
                correct_doc=correct_doc,
            )
        except Exception as exc:
            verdict = {"verdict": "ERROR", "reason": str(exc)}

        return {
            "query_id": query_id,
            "failure_type": r.get("failure_type", "?"),
            "p_at_1": r.get("p_at_1", 0),
            "r_at_8": r.get("r_at_8", 0),
            **verdict,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge_one, r): r for r in records}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["verdict"] in ("PARSE_ERROR", "ERROR"):
                errors += 1
            v = result["verdict"]
            qid = result["query_id"]
            ft = result["failure_type"]
            sys.stdout.write(f"  [{qid}] {ft} -> {v}\n")
            sys.stdout.flush()

    return results


def print_summary(filepath: Path, results: list[dict]) -> dict:
    """Print and return summary stats for one file."""
    n = len(results)
    if n == 0:
        return {}

    correct = sum(1 for r in results if r["verdict"] == "CORRECT")
    incorrect = sum(1 for r in results if r["verdict"] == "INCORRECT")
    partial = sum(1 for r in results if r["verdict"] == "PARTIAL")
    parse_errors = sum(1 for r in results if r["verdict"] in ("PARSE_ERROR", "ERROR"))

    pass_rate = correct / n * 100
    weighted_rate = (correct + 0.5 * partial) / n * 100

    # Failure type breakdown
    from collections import Counter
    ft_counts = Counter(r["failure_type"] for r in results)
    ok_pct = ft_counts.get("OK", 0) / n * 100
    mean_r8 = sum(r.get("r_at_8", 0) for r in results) / n

    print()
    print(f"  {filepath.name}")
    print(f"    CORRECT:   {correct:>3} ({pass_rate:.1f}%)")
    print(f"    PARTIAL:   {partial:>3} ({partial/n*100:.1f}%)")
    print(f"    INCORRECT: {incorrect:>3} ({incorrect/n*100:.1f}%)")
    if parse_errors:
        print(f"    ERRORS:    {parse_errors:>3} ({parse_errors/n*100:.1f}%)")
    print(f"    Weighted:  {weighted_rate:.1f}%")
    print(f"    OK%: {ok_pct:.1f}%  R@8: {mean_r8:.3f}")

    return {
        "file": filepath.name,
        "n": n,
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "pass_rate": pass_rate,
        "weighted_rate": weighted_rate,
        "ok_pct": ok_pct,
        "mean_r8": mean_r8,
        "parse_errors": parse_errors,
    }


def main():
    import argparse
    from urllib.parse import unquote

    p = argparse.ArgumentParser(description="Judge answer correctness")
    p.add_argument("files", nargs="+", type=Path,
                   help="JSONL result files to judge")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=Path("data"),
                   help="Directory for judge output files")
    args = p.parse_args()

    # Load ground truth doc IDs
    benchmark = json.loads(
        Path("data/benchmarks/contractnli.json").read_text(encoding="utf-8")
    )
    correct_docs = {}
    for t in benchmark["tests"]:
        correct_docs[t["query_id"]] = unquote(t["snippets"][0]["file_path"])

    all_summaries = []

    for filepath in args.files:
        if not filepath.exists():
            print(f"Skipping {filepath} (not found)")
            continue

        print(f"\nJudging {filepath.name} ({sum(1 for _ in filepath.open())} records)...")
        results = judge_file(filepath, correct_docs, workers=args.workers)

        summary = print_summary(filepath, results)
        all_summaries.append(summary)

        # Save detailed results
        out_path = args.output_dir / f"judge_{filepath.stem}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in sorted(results, key=lambda x: x["query_id"]):
                f.write(json.dumps(r) + "\n")
        print(f"    Detailed results: {out_path}")

    # Cross-file comparison table
    if len(all_summaries) > 1:
        print("\n" + "=" * 80)
        print("CROSS-CONFIG COMPARISON (LLM-judged, NOT deterministic)")
        print("=" * 80)
        print(f"  {'Config':<45} {'OK%':>6} {'R@8':>6} {'CORRECT%':>9} {'PARTIAL%':>9}")
        print("  " + "-" * 75)
        for s in all_summaries:
            print(
                f"  {s['file']:<45} "
                f"{s['ok_pct']:>5.1f}% "
                f"{s['mean_r8']:>5.3f} "
                f"{s['pass_rate']:>8.1f}% "
                f"{s['partial']/s['n']*100:>8.1f}%"
            )
        print("=" * 80)

    # DRM + CORRECT spot check
    print("\n--- DRM queries judged CORRECT (wrong doc, right answer) ---")
    for filepath in args.files:
        out_path = args.output_dir / f"judge_{filepath.stem}.jsonl"
        if not out_path.exists():
            continue
        drm_correct = []
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                if r["failure_type"] == "DRM" and r["verdict"] == "CORRECT":
                    drm_correct.append(r)
        if drm_correct:
            print(f"\n  {filepath.name}: {len(drm_correct)} DRM+CORRECT")
            for r in drm_correct[:3]:
                print(f"    {r['query_id']}: {r.get('reason', '')[:100]}")


if __name__ == "__main__":
    main()
