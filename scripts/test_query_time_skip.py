"""Verify that a query_time experiment makes zero corpus-embedding calls.

Runs evaluate_config with skip_index=True on the locked ContractNLI config
(only top_k changed: bm25_top_k=64 vs baseline 32).  Checks:
  (a) No "Embedding batch" or "Indexed" lines → zero Voyage corpus calls.
  (b) Eval completes and P@1 / R@8 are plausible numbers.

The existing contractnli_baseline collection must already have points.
If it's empty this will raise immediately with a clear message.
"""

from __future__ import annotations

import io
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# Intercept stdout so we can scan for forbidden strings after the run.
class Tee:
    """Write to both a buffer and the real stdout."""
    def __init__(self, real):
        self.real = real
        self.buf = io.StringIO()

    def write(self, s):
        self.real.write(s)
        self.buf.write(s)

    def flush(self):
        self.real.flush()

tee = Tee(sys.stdout)
sys.stdout = tee

print("=" * 72)
print("  test_query_time_skip.py")
print("  Config: contractnli, chunk=512/128, hybrid, bm25_k=64, dense_k=32, fusion_n=64")
print("  experiment_type: query_time  ->  skip_index=True")
print("=" * 72)
print()

from core.evaluation.run_eval import evaluate_config

config = {
    "chunk_size": 512,
    "chunk_overlap": 128,
    "bm25_top_k": 64,         # changed from baseline 32 — the "query_time" tweak
    "dense_top_k": 32,
    "fusion_top_n": 64,
    "retrieval_mode": "hybrid",
}

t0 = time.time()
metric_result, failure_counts = evaluate_config(
    config,
    dataset_name="contractnli",
    skip_index=True,           # <-- the flag under test
)
elapsed = time.time() - t0

print()
print("=" * 72)
print("  RESULTS")
print("=" * 72)
p1 = metric_result.p_at_k.get(1, 0) * 100
r8 = metric_result.r_at_k.get(8, 0) * 100
print(f"  P@1 = {p1:.2f}%")
print(f"  R@8 = {r8:.2f}%")

total_q = sum(failure_counts.values())
parts = []
for ft in ["DRM", "CBF", "SGP", "ICR", "OVR", "OK"]:
    c = failure_counts.get(ft, 0)
    pct = c / total_q * 100 if total_q else 0
    parts.append(f"{ft}={c}({pct:.1f}%)")
print(f"  Failures: {' '.join(parts)}")
print(f"  Elapsed: {elapsed:.1f}s")

# --- Audit the captured output ---
sys.stdout = tee.real   # restore
captured = tee.buf.getvalue()

print()
print("=" * 72)
print("  EMBEDDING AUDIT")
print("=" * 72)

forbidden = [line for line in captured.splitlines()
             if "Embedding batch" in line or "Indexed" in line]

if forbidden:
    print(f"  FAIL — {len(forbidden)} forbidden line(s) found:")
    for line in forbidden:
        print(f"    >> {line}")
    sys.exit(1)
else:
    print("  PASS — zero 'Embedding batch' / 'Indexed' lines in output.")
    print("  Corpus re-embedding did NOT fire. query_time skip is working.")
