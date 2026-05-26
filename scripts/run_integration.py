"""Phase 3 integration gate: run the full Supervisor loop for N cycles.

Local only — no VPS, no systemd, no deploy.  Watch it walk.

Usage:
    python -m scripts.run_integration [--cycles N]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from core.supervisor.graph import build_graph
from core.supervisor.proposer import load_ledger
from core.supervisor.schemas import VARIANCE_FLOOR_PP

from langgraph.checkpoint.sqlite import SqliteSaver


CHECKPOINT_DB = Path("data/integration_checkpoints.sqlite")


def run_cycle(cycle: int, total: int) -> dict:
    """Run one graph invocation (one experiment cycle)."""
    print(f"\n{'='*72}")
    print(f"  CYCLE {cycle}/{total}")
    print(f"{'='*72}\n")

    # Fresh graph per cycle — state flows via ledger on disk
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    compiled = build_graph().compile(checkpointer=saver)

    thread_id = f"integration-cycle-{cycle}"
    config = {"configurable": {"thread_id": thread_id}}

    t0 = time.time()
    try:
        result = compiled.invoke({}, config=config)
    except Exception as exc:
        print(f"\n  CYCLE {cycle} CRASHED: {exc}")
        conn.close()
        raise
    finally:
        elapsed = time.time() - t0

    conn.close()

    # --- Report ---
    print(f"\n--- Cycle {cycle} completed in {elapsed:.1f}s ---")
    _report_cycle(cycle, result)

    return result


def _report_cycle(cycle: int, state: dict) -> None:
    """Print per-cycle diagnostics."""
    status = state.get("status", "unknown")
    print(f"  Status: {status}")

    # Was it short-circuited?
    if state.get("duplicate"):
        print(f"  ** DEDUP FIRED: config hash {state.get('config_hash', '?')[:16]}... already in ledger")
        return
    if not state.get("spend_ok", True):
        print(f"  ** SPEND CAP HIT: experiment would exceed budget")
        return
    sanity = state.get("sanity", {})
    if sanity.get("quarantined"):
        print(f"  ** QUARANTINED: {sanity.get('violations', [])}")
        return

    # Config proposed
    cfg = state.get("config", {})
    print(f"\n  Proposer proposed:")
    print(f"    Dataset:    {cfg.get('target_dataset')}")
    print(f"    Chunk:      {cfg.get('chunk_size')}/{cfg.get('chunk_overlap')}")
    print(f"    Mode:       {cfg.get('retrieval_mode')}")
    print(f"    BM25 k:     {cfg.get('bm25_top_k')}")
    print(f"    Dense k:    {cfg.get('dense_top_k')}")
    print(f"    Fusion n:   {cfg.get('fusion_top_n')}")
    print(f"    Exp type:   {state.get('experiment_type')}")
    print(f"    Est chunks: {state.get('estimated_embedding_chunks', 0):,}")

    print(f"\n  Hypothesis: {state.get('hypothesis', '(none)')[:200]}")

    pd = state.get("predicted_delta", {})
    print(f"\n  Predicted delta: {pd.get('metric')} {pd.get('delta_pp', 0):+.2f}pp "
          f"vs {pd.get('baseline_ref')} "
          f"(above_floor={pd.get('above_variance_floor')})")

    # Eval results
    mr = state.get("metric_result")
    if mr is not None:
        p1 = mr.p_at_k.get(1, 0) * 100
        r8 = mr.r_at_k.get(8, 0) * 100
        print(f"\n  Eval returned:")
        print(f"    P@1  = {p1:.2f}%")
        print(f"    R@8  = {r8:.2f}%")

    fc = state.get("failure_counts", {})
    if fc:
        total_q = sum(fc.values())
        parts = []
        for ft in ["DRM", "CBF", "SGP", "ICR", "OVR", "OK"]:
            c = fc.get(ft, 0)
            pct = c / total_q * 100 if total_q else 0
            parts.append(f"{ft}={c}({pct:.1f}%)")
        print(f"    Failures: {' '.join(parts)}")

    # Calibration
    adp = state.get("actual_delta_pp")
    pep = state.get("prediction_error_pp")
    if adp is not None:
        print(f"\n  Calibration:")
        print(f"    Actual delta:     {adp:+.2f}pp")
        print(f"    Prediction error: {pep:+.2f}pp")

    # Significance check: did the metric move past the floor?
    if mr is not None and pd:
        metric_name = pd.get("metric", "")
        floor = VARIANCE_FLOOR_PP.get(metric_name, 0.6)
        if adp is not None:
            if abs(adp) <= floor:
                print(f"    ** Delta |{adp:+.2f}| <= floor {floor}pp → NOISE (Scribe should call it noise)")
            else:
                print(f"    ** Delta |{adp:+.2f}| > floor {floor}pp → SIGNIFICANT")

    # Sanity
    sanity = state.get("sanity", {})
    if sanity.get("passed"):
        print(f"\n  Sanity: PASSED")
    else:
        print(f"\n  Sanity: FAILED — {sanity.get('violations', [])}")

    # Decision entry snippet
    de = state.get("decision_entry", "")
    if de:
        lines = de.strip().split("\n")
        print(f"\n  Decision entry ({len(lines)} lines):")
        for line in lines[:5]:
            print(f"    {line}")
        if len(lines) > 5:
            print(f"    ... ({len(lines) - 5} more lines)")

    print(f"\n  Trace ID: {state.get('trace_id', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 integration gate")
    parser.add_argument("--cycles", type=int, default=3, help="Number of cycles to run")
    args = parser.parse_args()

    n = args.cycles
    print(f"Phase 3 integration gate: {n} cycles on ContractNLI")
    print(f"Spend cap: {50_000:,} embedding chunks")
    print(f"Checkpoint DB: {CHECKPOINT_DB}")

    # Ensure data dir exists
    Path("data").mkdir(exist_ok=True)

    results = []
    for i in range(1, n + 1):
        try:
            result = run_cycle(i, n)
            results.append(result)
        except Exception as exc:
            print(f"\nFATAL: Cycle {i} failed with {type(exc).__name__}: {exc}")
            break

        # Check if spend cap was hit (stop early)
        if not result.get("spend_ok", True):
            print(f"\nSpend cap hit at cycle {i} — stopping early.")
            break

    # --- Summary ---
    print(f"\n{'='*72}")
    print(f"  INTEGRATION SUMMARY: {len(results)} cycles completed")
    print(f"{'='*72}")

    ledger = load_ledger()
    print(f"  Ledger entries: {len(ledger)}")

    for e in ledger:
        fv = e.failure_vector
        p1 = e.p_at_k.get(1, 0) * 100
        r8 = e.r_at_k.get(8, 0) * 100
        print(
            f"  Run {e.run_number}: {e.config.target_dataset} "
            f"chunk={e.config.chunk_size}/{e.config.chunk_overlap} "
            f"{e.config.retrieval_mode} "
            f"P@1={p1:.2f}% R@8={r8:.2f}% "
            f"DRM={fv.drm} CBF={fv.cbf} SGP={fv.sgp} ICR={fv.icr} OVR={fv.ovr} OK={fv.ok} "
            f"delta={e.actual_delta_pp}pp err={e.prediction_error_pp}pp"
        )

    for i, res in enumerate(results, 1):
        if res.get("duplicate"):
            print(f"  Cycle {i}: DEDUP (config already tried)")
        elif not res.get("spend_ok", True):
            print(f"  Cycle {i}: SPEND CAP")
        elif res.get("sanity", {}).get("quarantined"):
            print(f"  Cycle {i}: QUARANTINED")


if __name__ == "__main__":
    main()
