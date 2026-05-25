"""Stub nodes for the Supervisor graph.

Each function takes ExperimentState, returns a partial dict that
LangGraph merges back into state.  No real logic yet — stubs set
placeholder values so the graph is smoke-testable end to end.
"""

from __future__ import annotations

from core.supervisor.state import ExperimentState


def read_ledger(state: ExperimentState) -> dict:
    """Read the experiment ledger and assign the next run_number."""
    return {"run_number": (state.get("run_number") or 0) + 1, "status": "running"}


def propose_config(state: ExperimentState) -> dict:
    """Proposer agent: read ledger, emit config + hypothesis."""
    return {
        "config": {
            "chunk_size": 512,
            "chunk_overlap": 128,
            "bm25_top_k": 32,
            "dense_top_k": 32,
            "fusion_top_n": 64,
        },
        "hypothesis": "stub hypothesis",
        "predicted_delta": 0.0,
    }


def check_hash(state: ExperimentState) -> dict:
    """Hash the proposed config and check for duplicates."""
    return {
        "config_hash": "stub-hash",
        "duplicate": False,
    }


def check_spend(state: ExperimentState) -> dict:
    """Verify budget allows this experiment run."""
    return {"spend_ok": True}


def run_eval(state: ExperimentState) -> dict:
    """Build index, run retrieval + measurement over all queries."""
    from core.evaluation.run_eval import evaluate_config

    config = dict(state["config"])
    metric_result, failure_counts = evaluate_config(config)

    return {
        "metric_result": metric_result,
        "failure_counts": failure_counts,
        "trace_id": "TODO-langfuse-trace-id",
    }


def sanity_check(state: ExperimentState) -> dict:
    """Post-eval sanity invariants (R@k monotonic, delta cap, etc.)."""
    return {
        "sanity": {
            "passed": True,
            "violations": [],
            "quarantined": False,
        },
    }


def log_results(state: ExperimentState) -> dict:
    """Append metrics + failure distribution to the ledger."""
    return {}


def write_entry(state: ExperimentState) -> dict:
    """Scribe: write a decision_log.md entry."""
    return {"decision_entry": "stub entry"}


def notify(state: ExperimentState) -> dict:
    """Send Apprise notification (email + Discord/Slack)."""
    return {"notified": True, "status": "completed"}
