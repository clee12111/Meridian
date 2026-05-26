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
    """Proposer agent: call DeepSeek, emit config + hypothesis + predicted delta.

    Reads the experiment ledger from disk, constructs a prompt with the
    full history and cost model, and returns a validated ExperimentProposal.
    """
    from core.supervisor.proposer import propose

    proposal = propose()

    return {
        "config": proposal.config.model_dump(),
        "hypothesis": proposal.hypothesis,
        "predicted_delta": proposal.predicted_delta.model_dump(),
        "experiment_type": proposal.experiment_type,
        "estimated_embedding_chunks": proposal.estimated_embedding_chunks,
        "cost_reasoning": proposal.cost_reasoning,
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
    """Scribe: call DeepSeek to write a decision_log.md entry.

    Reads the completed experiment from state, resolves the baseline
    reference, computes significance verdicts deterministically, and
    calls the Scribe LLM to narrate the entry.
    """
    from core.supervisor.proposer import load_ledger
    from core.supervisor.schemas import (
        FailureVector, LedgerEntry, PredictedDelta, RagConfig,
    )
    from core.supervisor.scribe import write_decision_entry

    # Build a LedgerEntry from current state
    entry = LedgerEntry(
        run_number=state["run_number"],
        experiment_id=state.get("config_hash", ""),
        config=RagConfig(**state["config"]),
        hypothesis=state["hypothesis"],
        predicted_delta=PredictedDelta(**state["predicted_delta"]),
        experiment_type=state.get("experiment_type", "ingestion_time"),
        estimated_embedding_chunks=state.get("estimated_embedding_chunks", 0),
        p_at_k=state["metric_result"].p_at_k,
        r_at_k=state["metric_result"].r_at_k,
        failure_vector=FailureVector.from_counts(state["failure_counts"]),
        n_queries=sum(state["failure_counts"].values()),
        actual_embedding_chunks=state.get("estimated_embedding_chunks", 0),
        actual_delta_pp=None,   # filled by Supervisor after comparison
        prediction_error_pp=None,
        trace_id=state.get("trace_id", ""),
    )

    ledger = load_ledger()
    decision_text = write_decision_entry(entry, ledger)

    return {"decision_entry": decision_text}


def notify(state: ExperimentState) -> dict:
    """Send Apprise notification (email + Discord/Slack)."""
    return {"notified": True, "status": "completed"}
