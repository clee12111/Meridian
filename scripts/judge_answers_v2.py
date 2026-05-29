"""Span-informed answer correctness judge for all LegalBench-RAG corpora.

Reads saved JSONL results, judges each answer against the golden span text
with DeepSeek-flash, reports affirmative-case answer correctness per corpus.

This is an LLM-judged metric, separate from the deterministic span taxonomy.
All four corpora are affirmative-only — these numbers measure correctness on
provisions that EXIST (recall of evidence), NOT false-positive rate.

Usage:
    python scripts/judge_answers_v2.py --corpus contractnli --input data/file.jsonl
    python scripts/judge_answers_v2.py --corpus cuad --input data/file1.jsonl data/file2.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are a legal document analysis judge. Your task is to evaluate whether \
a system's answer correctly addresses a query about a specific document.

GROUND TRUTH: The query asks about a provision that EXISTS in the document. \
The correct evidence is the golden span text below. The system should \
convey the same information as this evidence.

QUERY:
{query}

GOLDEN EVIDENCE (the correct answer from the document):
{golden_text}

CORRECT DOCUMENT: {correct_doc}

SYSTEM ANSWER:
{answer}

SYSTEM CITED TEXT:
{cited_text}

DOCUMENT(S) THE SYSTEM CITED: {cited_docs}

Judge the system's answer on these criteria:
1. Does the answer correctly convey the same information as the golden \
evidence? Score SEMANTIC match, not string match — a correctly paraphrased \
answer is CORRECT (e.g. "expires March 2025" matches "shall terminate on \
the 15th day of March, 2025").
2. Did the system cite the correct document?
3. Is the answer coherent and responsive to the query?

Return ONLY a JSON object (no markdown, no commentary):
{{"verdict": "CORRECT" or "INCORRECT" or "PARTIAL", "reason": "one sentence"}}

Verdict guide:
- CORRECT: Answer conveys the same information as the golden evidence, \
cites the correct document, and is coherent. Paraphrasing is fine.
- INCORRECT: Answer contradicts the golden evidence, says the information \
is absent/not found, cites the wrong document, or is incoherent.
- PARTIAL: Answer captures some but not all of the golden evidence, or \
hedges significantly, or cites the wrong document but reaches a partly \
correct conclusion."""

CORPUS_CONFIG = {
    "contractnli": {
        "benchmark": Path("data/benchmarks/contractnli.json"),
        "corpus_dir": Path("data/corpus"),
        "gt_module": "core.evaluation.ground_truth_contractnli",
        "gt_class": "ContractNLIGroundTruth",
    },
    "privacyqa": {
        "benchmark": Path("data/benchmarks/privacy_qa.json"),
        "corpus_dir": Path("data/corpus"),
        "gt_module": "core.evaluation.ground_truth_privacyqa",
        "gt_class": "PrivacyQAGroundTruth",
    },
    "cuad": {
        "benchmark": Path("data/benchmarks/cuad.json"),
        "corpus_dir": Path("data/corpus"),
        "gt_module": "core.evaluation.ground_truth_cuad",
        "gt_class": "CUADGroundTruth",
    },
    "maud": {
        "benchmark": Path("data/benchmarks/maud.json"),
        "corpus_dir": Path("data/corpus"),
        "gt_module": "core.evaluation.ground_truth_maud",
        "gt_class": "MAUDGroundTruth",
    },
}

_RETRY_WAITS = [2, 5, 10]


def _parse_verdict(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "PARSE_ERROR").upper()
        reason = parsed.get("reason", "")
        if verdict not in ("CORRECT", "INCORRECT", "PARTIAL"):
            return {"verdict": "PARSE_ERROR", "reason": f"Unknown: {verdict}", "raw": raw}
        return {"verdict": verdict, "reason": reason}
    except json.JSONDecodeError:
        m = re.search(r'"verdict"\s*:\s*"(CORRECT|INCORRECT|PARTIAL)"', raw, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()
            m2 = re.search(r'"reason"\s*:\s*"([^"]*)', raw)
            reason = m2.group(1) if m2 else "(truncated)"
            return {"verdict": verdict, "reason": reason}
        return {"verdict": "PARSE_ERROR", "reason": "Failed to parse JSON", "raw": raw}


def _call_judge(query: str, answer: str, claims: list[dict],
                correct_doc: str, golden_text: str) -> dict:
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

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
        golden_text=golden_text[:2000],
    )

    for attempt, wait in enumerate(_RETRY_WAITS, start=1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
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

    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_verdict(raw) if raw else {"verdict": "PARSE_ERROR", "reason": "Empty response after retries"}


def judge_file(filepath: Path, ground_truth, workers: int = 8) -> list[dict]:
    records = []
    with filepath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return []

    results = []

    def judge_one(r: dict) -> dict:
        query_id = r["query_id"]
        try:
            correct_doc = ground_truth.get_doc_id(query_id)
            gt_spans = ground_truth.get_spans(query_id)
            doc_text = ground_truth.doc_text(correct_doc)
            golden_text = "\n...\n".join(
                doc_text[s:e] for s, e in gt_spans
                if s < len(doc_text) and e <= len(doc_text)
            )[:2000]
        except Exception:
            correct_doc = "unknown"
            golden_text = "(could not load golden text)"

        try:
            verdict = _call_judge(
                query=r["query"],
                answer=r.get("answer", ""),
                claims=r.get("claims", []),
                correct_doc=correct_doc,
                golden_text=golden_text,
            )
        except Exception as exc:
            verdict = {"verdict": "ERROR", "reason": str(exc)}

        return {
            "query_id": query_id,
            "failure_type": r.get("failure_type", "?"),
            "r_at_8": r.get("r_at_8", 0),
            **verdict,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge_one, r): r for r in records}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            sys.stdout.write(
                f"  [{result['query_id']}] {result['failure_type']} -> {result['verdict']}\n"
            )
            sys.stdout.flush()

    return results


def main():
    import importlib

    p = argparse.ArgumentParser(description="Span-informed answer judge")
    p.add_argument("--corpus", required=True,
                   choices=list(CORPUS_CONFIG.keys()),
                   help="Which corpus ground truth to use")
    p.add_argument("--input", nargs="+", type=Path, required=True,
                   help="JSONL result files to judge")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    cfg = CORPUS_CONFIG[args.corpus]
    mod = importlib.import_module(cfg["gt_module"])
    gt_cls = getattr(mod, cfg["gt_class"])
    ground_truth = gt_cls(cfg["benchmark"], cfg["corpus_dir"])

    all_summaries = []

    for filepath in args.input:
        if not filepath.exists():
            print(f"Skipping {filepath} (not found)")
            continue

        n_records = sum(1 for line in filepath.open(encoding="utf-8") if line.strip())
        print(f"\nJudging {filepath.name} ({n_records} records, corpus={args.corpus})...")
        results = judge_file(filepath, ground_truth, workers=args.workers)

        n = len(results)
        correct = sum(1 for r in results if r["verdict"] == "CORRECT")
        partial = sum(1 for r in results if r["verdict"] == "PARTIAL")
        incorrect = sum(1 for r in results if r["verdict"] == "INCORRECT")
        errors = sum(1 for r in results if r["verdict"] in ("PARSE_ERROR", "ERROR"))

        print(f"\n  {filepath.name} ({args.corpus}):")
        print(f"    CORRECT:   {correct:>3} ({correct/n*100:.1f}%)")
        print(f"    PARTIAL:   {partial:>3} ({partial/n*100:.1f}%)")
        print(f"    INCORRECT: {incorrect:>3} ({incorrect/n*100:.1f}%)")
        if errors:
            print(f"    ERRORS:    {errors:>3} ({errors/n*100:.1f}%)")

        all_summaries.append({
            "file": filepath.name, "corpus": args.corpus, "n": n,
            "correct": correct, "partial": partial, "incorrect": incorrect,
            "errors": errors, "pass_rate": correct / n * 100,
        })

        out_path = args.output_dir / f"judge_v2_{args.corpus}_{filepath.stem}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in sorted(results, key=lambda x: x["query_id"]):
                f.write(json.dumps(r) + "\n")
        print(f"    Saved: {out_path}")

    if len(all_summaries) > 1:
        print(f"\n{'File':<50} {'CORRECT%':>9} {'PARTIAL%':>9} {'Errors':>6}")
        print("-" * 76)
        for s in all_summaries:
            print(f"  {s['file']:<48} {s['pass_rate']:>8.1f}% "
                  f"{s['partial']/s['n']*100:>8.1f}% {s['errors']:>5}")


if __name__ == "__main__":
    main()
