"""Produce the four-axis headline report from judged results."""
import json
import sys
from pathlib import Path

jd = "data/headline/judge_results"

def load(path):
    r = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                r[d["query_id"]] = d
    return r

PUBLISHED = {
    # arXiv 2408.10343 Table 5: RCTS 500-char, text-embedding-3-large, no reranker
    # Calibrated: MAUD, CUAD, PrivacyQA (ruler confirmed within ~2-3pp drift, Finding 44)
    # Caveated: ContractNLI (benchmark-file provenance differs)
    "contractnli": {"source": "RCTS (caveat)", "p_at_1": 0.066, "r_at_8": 0.250},
    "maud": {"source": "RCTS", "p_at_1": 0.027, "r_at_8": 0.062},
    "cuad": {"source": "RCTS", "p_at_1": 0.020, "r_at_8": 0.317},
    "privacyqa": {"source": "RCTS", "p_at_1": 0.144, "r_at_8": 0.424},
}

print("=" * 95)
print("HEADLINE MEASUREMENT REPORT")
print("Config-stack delta: CC + routing + selector over RRF baseline")
print("SAC chunking HELD CONSTANT -- NOT the full-system v1->v2 delta")
print("Embedding: voyage-4. Judge: span-informed (answer) + holistic (faithfulness)")
print("=" * 95)

all_data = {}
for corpus in ["contractnli", "privacyqa", "cuad", "maud"]:
    all_data[corpus] = {}
    hf = f"data/headline/headline_{corpus}.jsonl"
    recs = {}
    with open(hf) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs.setdefault(r["arm"], []).append(r)

    for arm in [0, 1, 2]:
        ar = recs.get(arm, [])
        if not ar:
            continue
        n = len(ar)
        p1 = sum(r["p_at_1"] for r in ar) / n
        r8 = sum(r["r_at_8"] for r in ar) / n
        lat = sum(r["latency_ms"] for r in ar) / n
        tok = sum(r["tokens_in"] + r["tokens_out"] for r in ar) / n
        prom = sum(1 for r in ar if r["n_promoted"] > 0)

        judges = load(f"{jd}/judge_v2_{corpus}_{corpus}_arm{arm}.jsonl")
        faiths = load(f"{jd}/faith_{corpus}_{corpus}_arm{arm}.jsonl")

        correct = sum(1 for v in judges.values() if v["verdict"] == "CORRECT")
        partial = sum(1 for v in judges.values() if v["verdict"] == "PARTIAL")
        incorrect = sum(1 for v in judges.values() if v["verdict"] == "INCORRECT")
        faith_mean = sum(v.get("faithfulness_score", 0) for v in faiths.values()) / max(len(faiths), 1)

        gc = 0
        for qid in judges:
            if judges[qid]["verdict"] == "CORRECT" and faiths.get(qid, {}).get("faithfulness_score", 0) >= 0.8:
                gc += 1

        all_data[corpus][arm] = {
            "n": n, "p1": p1, "r8": r8, "lat": lat, "tok": tok, "prom": prom,
            "correct": correct, "partial": partial, "incorrect": incorrect,
            "faith": faith_mean, "gc": gc, "correct_pct": correct / n * 100,
        }

# Per-corpus tables
for corpus in ["contractnli", "privacyqa", "cuad", "maud"]:
    d = all_data[corpus]
    pub = PUBLISHED[corpus]
    a0, a1, a2 = d.get(0, {}), d.get(1, {}), d.get(2, {})

    print(f"\n{'-'*95}")
    n = a0.get("n", 194)
    print(f"  {corpus.upper()} ({n} queries)")
    print(f"{'-'*95}")

    def row(label, k, fmt=".3f", typ=""):
        v0 = format(a0.get(k, 0), fmt) if a0 else "-"
        v1 = format(a1.get(k, 0), fmt) if a1 else "-"
        v2 = format(a2.get(k, 0), fmt) if a2 else "-"
        print(f"  {label:<30} {v0:>12}  {v1:>12}  {v2:>12}  {typ}")

    def row_ratio(label, k1, k2):
        def f(a):
            return f"{a.get(k1, 0)}/{a.get(k2, 0)}" if a else "-"
        print(f"  {label:<30} {f(a0):>12}  {f(a1):>12}  {f(a2):>12}")

    print(f"  {'Metric':<30} {'Arm 0 (RRF)':>12}  {'Arm 1 (flash)':>12}  {'Arm 2 (Pro)':>12}  Type")
    print(f"  {'-'*30} {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
    row("Correct %", "correct_pct", ".1f", "variance")
    row_ratio("Grounded-correct", "gc", "n")
    row("Faithfulness", "faith", ".3f", "variance")
    row("P@1", "p1", ".3f", "DETERM")
    row("R@8", "r8", ".3f", "DETERM")
    row("Latency (ms)", "lat", ".0f", "variance")
    row("Tokens/query", "tok", ".0f", "~determ")
    row_ratio("Selector promotions", "prom", "n")

    if a0 and a1:
        dp1 = a1["p1"] - a0["p1"]
        dr8 = a1["r8"] - a0["r8"]
        dc = a1["correct_pct"] - a0["correct_pct"]
        print(f"\n  CONFIG-STACK DELTA (Arm 1 - Arm 0):")
        print(f"    Correct:  {a0['correct_pct']:.1f}% -> {a1['correct_pct']:.1f}% ({dc:+.1f}pp)")
        print(f"    P@1:      {a0['p1']:.3f} -> {a1['p1']:.3f} ({dp1:+.3f})")
        print(f"    R@8:      {a0['r8']:.3f} -> {a1['r8']:.3f} ({dr8:+.3f})")

    if pub["source"]:
        print(f"\n  EXTERNAL (vs {pub['source']} published baseline):")
        print(f"    P@1: {pub['p_at_1']:.3f} (published) vs {a1['p1']:.3f} (Arm 1) = {a1['p1']/pub['p_at_1']:.1f}x")
        print(f"    R@8: {pub['r_at_8']:.3f} (published) vs {a1['r8']:.3f} (Arm 1) = {a1['r8']/pub['r_at_8']:.1f}x")
    else:
        print(f"\n  EXTERNAL: no published baseline for {corpus}")

    if a1 and a2:
        dc2 = a2["correct_pct"] - a1["correct_pct"]
        df2 = a2["faith"] - a1["faith"]
        dl2 = a2["lat"] - a1["lat"]
        print(f"\n  PRO vs FLASH (cost-quality tradeoff):")
        print(f"    Correct:  {a1['correct_pct']:.1f}% -> {a2['correct_pct']:.1f}% ({dc2:+.1f}pp)")
        print(f"    Faith:    {a1['faith']:.3f} -> {a2['faith']:.3f} ({df2:+.3f})")
        print(f"    Latency:  {a1['lat']:.0f}ms -> {a2['lat']:.0f}ms ({dl2:+.0f}ms)")

# Cross-corpus
print(f"\n{'='*95}")
print("CROSS-CORPUS AVERAGE (4 corpora, 776 queries)")
print(f"{'='*95}")
for arm, label in [(0, "Arm 0 (RRF baseline)"), (1, "Arm 1 (best flash)"), (2, "Arm 2 (best Pro)")]:
    vals = [all_data[c][arm] for c in all_data if arm in all_data[c]]
    if not vals:
        continue
    tn = sum(v["n"] for v in vals)
    ap1 = sum(v["p1"] * v["n"] for v in vals) / tn
    ar8 = sum(v["r8"] * v["n"] for v in vals) / tn
    ac = sum(v["correct"] for v in vals) / tn * 100
    agc = sum(v["gc"] for v in vals)
    af = sum(v["faith"] * v["n"] for v in vals) / tn
    al = sum(v["lat"] * v["n"] for v in vals) / tn
    at = sum(v["tok"] * v["n"] for v in vals) / tn
    tp = sum(v["prom"] for v in vals)
    print(f"  {label}:")
    print(f"    Correct: {ac:.1f}%  Grounded-correct: {agc}/{tn}  Faith: {af:.3f}")
    print(f"    P@1: {ap1:.3f}  R@8: {ar8:.3f}  Latency: {al:.0f}ms  Tok/q: {at:.0f}  Promotions: {tp}/{tn}")

print(f"\nREPORTING DISCIPLINE:")
print(f"  Config-stack delta (CC+routing+selector over RRF), SAC held constant.")
print(f"  NOT the full-system v1->v2 delta. Do NOT merge with 25.8%->75.3%.")
print(f"  Faithfulness via holistic judge (Phase-9 grounding confounded on MAUD negation).")
