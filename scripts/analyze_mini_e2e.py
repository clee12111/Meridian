"""Analyze Mini E2E results — per-corpus per-arm report table.

Reads the raw JSONL output from run_mini_e2e.py, runs judge_answers_v2
and judge_faithfulness on each arm's records, and produces the report.

INTERPRETATION FRAME (locked):
  Not-improved queries split by gate_triggered:
    (a) GATE FIRED, loop ran, didn't improve -> real loop-mechanism failure
    (b) GATE NEVER FIRED (high grounding) -> out of loop's reach by gate design

  This experiment tests the loop on the low-grounding/access-miss population.
  It does NOT test the loop on the grounded-but-wrong comprehension cluster
  (which the grounding gate structurally excludes).

Usage:
    python scripts/analyze_mini_e2e.py
    python scripts/analyze_mini_e2e.py --input-dir data/mini_e2e
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


CORPUS_FROM_QID = {
    "contractnli": "contractnli",
    "privacy_qa": "privacyqa",
    "cuad": "cuad",
    "maud": "maud",
}


def _get_corpus(qid: str) -> str:
    """Extract corpus name from query_id."""
    prefix = qid.rsplit("-", 1)[0]
    return CORPUS_FROM_QID.get(prefix, prefix)


def _split_by_corpus_and_arm(records: list[dict], output_dir: Path) -> dict[tuple[str, int], Path]:
    """Split records by (corpus, arm) into separate JSONL files for judging."""
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        corpus = _get_corpus(r["query_id"])
        groups[(corpus, r["arm"])].append(r)

    file_map = {}
    for (corpus, arm), recs in groups.items():
        path = output_dir / f"{corpus}_arm{arm}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in sorted(recs, key=lambda x: x["query_id"]):
                f.write(json.dumps(r) + "\n")
        file_map[(corpus, arm)] = path
    return file_map


def _run_judge_answers(corpus: str, input_path: Path, output_dir: Path) -> list[dict]:
    """Run judge_answers_v2.py for a single corpus+arm file."""
    cmd = [
        sys.executable, "scripts/judge_answers_v2.py",
        "--corpus", corpus,
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        "--workers", "8",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  Answer judge failed for {input_path.name}: {result.stderr[:500]}", file=sys.stderr)
        return []
    # Output file is: output_dir/judge_v2_{corpus}_{input_stem}.jsonl
    out_path = output_dir / f"judge_v2_{corpus}_{input_path.stem}.jsonl"
    if not out_path.exists():
        return []
    with out_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_judge_faith(corpus: str, input_path: Path, output_dir: Path) -> list[dict]:
    """Run judge_faithfulness.py (holistic) for a single corpus+arm file."""
    cmd = [
        sys.executable, "scripts/judge_faithfulness.py",
        "--corpus", corpus,
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        "--workers", "8",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  Faith judge failed for {input_path.name}: {result.stderr[:500]}", file=sys.stderr)
        return []
    out_path = output_dir / f"faith_{corpus}_{input_path.stem}.jsonl"
    if not out_path.exists():
        return []
    with out_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_records(input_dir: Path) -> list[dict]:
    """Load all JSONL records from input directory."""
    records = []
    for path in sorted(input_dir.glob("mini_e2e_*.jsonl")):
        with path.open() as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def analyze(records: list[dict], answer_verdicts: dict, faith_scores: dict):
    """Produce the per-corpus per-arm report table."""
    corpora = sorted(set(_get_corpus(r["query_id"]) for r in records))

    print("\n" + "=" * 100)
    print("MINI E2E REPORT -- 3 Arms x N Corpora x 50 Queries")
    print("=" * 100)

    # Build lookup: (arm, query_id) -> record
    rec_lookup = {(r["arm"], r["query_id"]): r for r in records}

    for corpus in corpora:
        corpus_recs = [r for r in records if _get_corpus(r["query_id"]) == corpus]
        n_queries = len([r for r in corpus_recs if r["arm"] == 0])

        print(f"\n{'-' * 80}")
        print(f"CORPUS: {corpus.upper()} ({n_queries} queries)")
        print(f"{'-' * 80}")

        # Per-arm metrics
        print(f"\n{'Metric':<35} {'Arm 0 (Single)':<18} {'Arm 1 (Determ)':<18} {'Arm 2 (Critic)':<18}")
        print(f"{'-' * 35} {'-' * 18} {'-' * 18} {'-' * 18}")

        arm_data = {}
        for arm in [0, 1, 2]:
            arm_recs = [r for r in corpus_recs if r["arm"] == arm]
            if not arm_recs:
                continue
            n = len(arm_recs)

            # Answer verdicts
            correct = sum(1 for r in arm_recs
                          if answer_verdicts.get((arm, r["query_id"]), {}).get("verdict") == "CORRECT")
            # Faithfulness
            faith_vals = [faith_scores.get((arm, r["query_id"]), {}).get("faithfulness_score", 0.0)
                          for r in arm_recs]
            avg_faith = sum(faith_vals) / n if n else 0.0

            # CORRECT+FAITHFUL and CORRECT+UNFAITHFUL
            correct_faithful = 0
            correct_unfaithful = 0
            for r in arm_recs:
                v = answer_verdicts.get((arm, r["query_id"]), {})
                fs = faith_scores.get((arm, r["query_id"]), {})
                is_correct = v.get("verdict") == "CORRECT"
                is_faithful = fs.get("faithfulness_score", 0.0) >= 0.8
                if is_correct and is_faithful:
                    correct_faithful += 1
                elif is_correct and not is_faithful:
                    correct_unfaithful += 1

            gate_triggered = sum(1 for r in arm_recs if r["gate_triggered"])
            avg_iters = sum(r["iterations_used"] for r in arm_recs) / n
            avg_latency = sum(r["latency_ms"] for r in arm_recs) / n
            avg_grounding = sum(r["grounding_score"] for r in arm_recs) / n
            ft = Counter(r["failure_type"] for r in arm_recs)

            arm_data[arm] = {
                "n": n, "correct": correct, "avg_faith": avg_faith,
                "correct_faithful": correct_faithful,
                "correct_unfaithful": correct_unfaithful,
                "gate_triggered": gate_triggered,
                "avg_iters": avg_iters, "avg_latency": avg_latency,
                "avg_grounding": avg_grounding, "ft": ft,
            }

        if not arm_data:
            continue

        def fmt3(vals):
            return f"{vals[0]:<18} {vals[1]:<18} {vals[2]:<18}"

        a = [arm_data.get(i, {}) for i in range(3)]

        def _ratio(x, key):
            return f"{x.get(key, 0)}/{x.get('n', 0)}"

        def _float(x, key, fmt=".3f"):
            return format(x.get(key, 0), fmt)

        print(f"{'CORRECT+FAITHFUL (decider)':<35} {fmt3([_ratio(x, 'correct_faithful') for x in a])}")
        print(f"{'Faithfulness mean':<35} {fmt3([_float(x, 'avg_faith') for x in a])}")
        print(f"{'CORRECT+UNFAITHFUL (drift trap)':<35} {fmt3([_ratio(x, 'correct_unfaithful') for x in a])}")
        print(f"{'Correct (any faith)':<35} {fmt3([_ratio(x, 'correct') for x in a])}")
        print(f"{'Grounding score mean':<35} {fmt3([_float(x, 'avg_grounding') for x in a])}")
        print(f"{'Gate triggered':<35} {fmt3([_ratio(x, 'gate_triggered') for x in a])}")
        print(f"{'Iterations avg':<35} {fmt3([_float(x, 'avg_iters', '.1f') for x in a])}")
        print(f"{'Latency avg (ms)':<35} {fmt3([_float(x, 'avg_latency', '.0f') for x in a])}")

        # Regression check (Correction #3): CORRECT+FAITHFUL regressions
        print(f"\n  REGRESSIONS (Arm 0 CORRECT -> Arm N not-CORRECT):")
        for arm in [1, 2]:
            regressions = []
            arm0_qids = [r["query_id"] for r in corpus_recs if r["arm"] == 0]
            for qid in arm0_qids:
                v0 = answer_verdicts.get((0, qid), {}).get("verdict")
                vn = answer_verdicts.get((arm, qid), {}).get("verdict")
                if v0 == "CORRECT" and vn != "CORRECT":
                    regressions.append(qid)
            label = "DETERM" if arm == 1 else "CRITIC"
            print(f"    Arm {arm} ({label}): {len(regressions)} regressions {regressions[:5]}")

        # Gate-split analysis (interpretation frame)
        print(f"\n  GATE-SPLIT ANALYSIS (not-improved queries):")
        for arm in [1, 2]:
            label = "DETERM" if arm == 1 else "CRITIC"
            arm0_qids = [r["query_id"] for r in corpus_recs if r["arm"] == 0]
            gate_fired_no_improve = 0
            gate_never_fired = 0
            gate_fired_improved = 0
            for qid in arm0_qids:
                v0 = answer_verdicts.get((0, qid), {}).get("verdict")
                vn = answer_verdicts.get((arm, qid), {}).get("verdict")
                arm_n_rec = rec_lookup.get((arm, qid), {})
                triggered = arm_n_rec.get("gate_triggered", False)
                improved = (v0 != "CORRECT" and vn == "CORRECT")
                if improved:
                    gate_fired_improved += 1
                elif triggered:
                    gate_fired_no_improve += 1
                elif v0 != "CORRECT":
                    gate_never_fired += 1
            print(f"    Arm {arm} ({label}):")
            print(f"      Gate fired, loop improved:      {gate_fired_improved}")
            print(f"      Gate fired, loop didn't help:   {gate_fired_no_improve} (real mechanism failure)")
            print(f"      Gate never fired (out of reach): {gate_never_fired} (NOT a loop failure)")

    # Access-miss diagnostic (MAUD only)
    maud_access = ["maud-0684", "maud-1114", "maud-1452"]
    maud_recs = [r for r in records if r["query_id"] in maud_access]
    if maud_recs:
        print(f"\n{'-' * 80}")
        print("ACCESS-MISS RECOVERY (MAUD 0684/1114/1452)")
        print(f"{'-' * 80}")
        for qid in maud_access:
            for arm in [0, 1, 2]:
                r = rec_lookup.get((arm, qid))
                if not r:
                    continue
                v = answer_verdicts.get((arm, qid), {})
                fs = faith_scores.get((arm, qid), {})
                label = {0: "SINGLE", 1: "DETERM", 2: "CRITIC"}[arm]
                print(f"  {qid} Arm {arm} ({label}): "
                      f"gate={'FIRED' if r['gate_triggered'] else 'PASS'} "
                      f"verdict={v.get('verdict', '?')} "
                      f"faith={fs.get('faithfulness_score', 0):.2f} "
                      f"ft={r['failure_type']} "
                      f"iters={r['iterations_used']} "
                      f"chunks={len(r['context_chunks'])}")

    # Scope statement
    print(f"\n{'-' * 80}")
    print("SCOPE LIMITATION")
    print(f"{'-' * 80}")
    print("This experiment tests the redesigned loop (Finding 35) on the")
    print("LOW-GROUNDING / ACCESS-MISS population (gate: score < 0.75).")
    print("The grounded-but-wrong comprehension cluster (INCORRECT+FAITHFUL,")
    print("high grounding score) is STRUCTURALLY EXCLUDED by the grounding gate.")
    print("'Loop didn't fix comprehension' is NOT a loop failure — the gate")
    print("never sent those queries to the loop.")
    print("")
    print("Implication: if the comprehension cluster is the target, the gate")
    print("must change — fire on grounded-but-WRONG (needs a correctness/")
    print("reasoning gate, not a grounding score gate).")


def main():
    p = argparse.ArgumentParser(description="Analyze Mini E2E results")
    p.add_argument("--input-dir", type=Path, default=Path("data/mini_e2e"))
    p.add_argument("--skip-judges", action="store_true",
                   help="Skip running judges (use cached results)")
    args = p.parse_args()

    records = _load_records(args.input_dir)
    if not records:
        print(f"No records found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records from {args.input_dir}")
    print(f"  Arms: {Counter(r['arm'] for r in records)}")
    print(f"  Corpora: {Counter(r['query_id'].rsplit('-', 1)[0] for r in records)}")

    judge_dir = args.input_dir / "judge_results"
    judge_dir.mkdir(parents=True, exist_ok=True)

    # Split by (corpus, arm) for judging — judges require --corpus flag
    ca_files = _split_by_corpus_and_arm(records, judge_dir)

    answer_verdicts: dict[tuple[int, str], dict] = {}
    faith_scores: dict[tuple[int, str], dict] = {}

    if not args.skip_judges:
        for (corpus, arm), ca_path in sorted(ca_files.items()):
            label = {0: "SINGLE", 1: "DETERM", 2: "CRITIC"}[arm]
            print(f"\nJudging {corpus} Arm {arm} ({label})...")

            # Answer judge
            print(f"  Running judge_answers_v2.py...")
            ans_results = _run_judge_answers(corpus, ca_path, judge_dir)
            for r in ans_results:
                answer_verdicts[(arm, r["query_id"])] = r
            print(f"  -> {len(ans_results)} verdicts")

            # Faithfulness judge (holistic)
            print(f"  Running judge_faithfulness.py (holistic)...")
            faith_results = _run_judge_faith(corpus, ca_path, judge_dir)
            for r in faith_results:
                faith_scores[(arm, r["query_id"])] = r
            print(f"  -> {len(faith_results)} scores")
    else:
        # Load cached results from judge output files
        for path in judge_dir.glob("judge_v2_*.jsonl"):
            with path.open() as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        # Extract arm from filename: judge_v2_{corpus}_{corpus}_arm{N}.jsonl
                        stem = path.stem
                        for a in [0, 1, 2]:
                            if f"arm{a}" in stem:
                                answer_verdicts[(a, r["query_id"])] = r
                                break
        for path in judge_dir.glob("faith_*.jsonl"):
            with path.open() as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        stem = path.stem
                        for a in [0, 1, 2]:
                            if f"arm{a}" in stem:
                                faith_scores[(a, r["query_id"])] = r
                                break

    analyze(records, answer_verdicts, faith_scores)


if __name__ == "__main__":
    main()
