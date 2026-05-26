"""Supervisor graph nodes — real implementations for Phase 3 integration.

Each function takes ExperimentState, returns a partial dict that
LangGraph merges back into state.
"""

from __future__ import annotations

import hashlib
import uuid

from core.supervisor.state import ExperimentState


# ── Configurable caps ──────────────────────────────────────────────────

SPEND_CAP_CHUNKS = 50_000  # total embedding chunks across all runs
SANITY_IMPROVEMENT_THRESHOLD = 0.15  # 15pp — any single k improving more is suspicious


def read_ledger(state: ExperimentState) -> dict:
    """Read the experiment ledger and assign the next run_number."""
    from core.supervisor.proposer import load_ledger

    ledger = load_ledger()
    next_run = max((e.run_number for e in ledger), default=0) + 1
    return {"run_number": next_run, "status": "running"}


def propose_config(state: ExperimentState) -> dict:
    """Proposer agent: call DeepSeek, emit config + hypothesis + predicted delta."""
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
    """Hash the proposed config and check for duplicates in the ledger."""
    from core.supervisor.proposer import load_ledger
    from core.supervisor.schemas import RagConfig

    cfg = RagConfig(**state["config"])
    config_hash = hashlib.sha256(cfg.canonical_json().encode()).hexdigest()

    ledger = load_ledger()
    duplicate = any(e.experiment_id == config_hash for e in ledger)

    return {
        "experiment_id": config_hash,
        "config_hash": config_hash,
        "duplicate": duplicate,
    }


def check_spend(state: ExperimentState) -> dict:
    """Verify cumulative embedding budget allows this experiment run."""
    from core.supervisor.proposer import load_ledger

    ledger = load_ledger()
    spent = sum(e.actual_embedding_chunks for e in ledger)
    estimated = state.get("estimated_embedding_chunks", 0)

    spend_ok = (spent + estimated) <= SPEND_CAP_CHUNKS
    return {"spend_ok": spend_ok}


def run_eval(state: ExperimentState) -> dict:
    """Build index, run retrieval + measurement over all queries.

    query_time experiments (top_k / fusion / retrieval_mode changes only)
    skip re-embedding — the Qdrant collection is reused as-is.
    ingestion_time experiments always re-embed (chunk_size / overlap / embedder
    change means the index is stale).
    """
    from core.evaluation.run_eval import evaluate_config

    config = dict(state["config"])
    dataset_name = config.pop("target_dataset", None)
    experiment_type = state.get("experiment_type", "ingestion_time")
    skip_index = experiment_type == "query_time"

    metric_result, failure_counts = evaluate_config(
        config, dataset_name=dataset_name, skip_index=skip_index
    )

    trace_id = f"integration-{uuid.uuid4().hex[:12]}"
    return {
        "metric_result": metric_result,
        "failure_counts": failure_counts,
        "trace_id": trace_id,
    }


def sanity_check(state: ExperimentState) -> dict:
    """Post-eval sanity invariants (R@k monotonic, delta cap, etc.)."""
    from core.measurement.sanity import run_sanity_checks

    metric_result = state["metric_result"]
    sanity_result = run_sanity_checks(
        metric_result,
        baseline=None,  # no per-MetricResult baseline stored yet
        improvement_threshold=SANITY_IMPROVEMENT_THRESHOLD,
    )

    return {
        "sanity": {
            "passed": sanity_result.passed,
            "violations": sanity_result.violations,
            "quarantined": not sanity_result.passed,
        },
    }


def log_results(state: ExperimentState) -> dict:
    """Compute calibration, build LedgerEntry, append to ledger on disk."""
    from core.supervisor.proposer import LOCKED_BASELINES, append_ledger, load_ledger
    from core.supervisor.schemas import (
        FailureVector, LedgerEntry, PredictedDelta, RagConfig,
    )

    cfg = RagConfig(**state["config"])
    predicted = PredictedDelta(**state["predicted_delta"])
    metric_result = state["metric_result"]

    # --- Resolve baseline for calibration ---
    ledger = load_ledger()
    baseline_ref = predicted.baseline_ref

    actual_delta_pp = None
    prediction_error_pp = None

    if baseline_ref == "locked":
        locked = LOCKED_BASELINES.get(cfg.target_dataset)
        if locked is not None:
            baseline_val = locked.get(predicted.metric)
            if baseline_val is not None:
                current_val = _extract_metric(predicted.metric, metric_result, state["failure_counts"])
                actual_delta_pp = round(current_val - baseline_val, 2)
                prediction_error_pp = round(predicted.delta_pp - actual_delta_pp, 2)
    else:
        try:
            ref_num = int(baseline_ref)
            ref_entry = next((e for e in ledger if e.run_number == ref_num), None)
            if ref_entry is not None:
                baseline_val = _extract_metric_from_entry(predicted.metric, ref_entry)
                current_val = _extract_metric(predicted.metric, metric_result, state["failure_counts"])
                actual_delta_pp = round(current_val - baseline_val, 2)
                prediction_error_pp = round(predicted.delta_pp - actual_delta_pp, 2)
        except ValueError:
            pass

    # --- Build and append ---
    entry = LedgerEntry(
        run_number=state["run_number"],
        experiment_id=state.get("config_hash", state.get("experiment_id", "")),
        config=cfg,
        hypothesis=state["hypothesis"],
        predicted_delta=predicted,
        experiment_type=state.get("experiment_type", "ingestion_time"),
        estimated_embedding_chunks=state.get("estimated_embedding_chunks", 0),
        p_at_k=metric_result.p_at_k,
        r_at_k=metric_result.r_at_k,
        failure_vector=FailureVector.from_counts(state["failure_counts"]),
        n_queries=sum(state["failure_counts"].values()),
        actual_embedding_chunks=(
            state.get("estimated_embedding_chunks", 0)
            if state.get("experiment_type") == "ingestion_time" else 0
        ),
        actual_delta_pp=actual_delta_pp,
        prediction_error_pp=prediction_error_pp,
        trace_id=state.get("trace_id", ""),
    )

    append_ledger(entry)

    return {
        "actual_delta_pp": actual_delta_pp,
        "prediction_error_pp": prediction_error_pp,
    }


def write_entry(state: ExperimentState) -> dict:
    """Scribe: call DeepSeek to write a decision_log.md entry."""
    from core.supervisor.proposer import load_ledger
    from core.supervisor.schemas import (
        FailureVector, LedgerEntry, PredictedDelta, RagConfig,
    )
    from core.supervisor.scribe import write_decision_entry

    # Build a LedgerEntry from current state (including calibration from log_results)
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
        actual_embedding_chunks=(
            state.get("estimated_embedding_chunks", 0)
            if state.get("experiment_type") == "ingestion_time" else 0
        ),
        actual_delta_pp=state.get("actual_delta_pp"),
        prediction_error_pp=state.get("prediction_error_pp"),
        trace_id=state.get("trace_id", ""),
    )

    ledger = load_ledger()
    decision_text = write_decision_entry(entry, ledger)

    return {"decision_entry": decision_text}


def notify(state: ExperimentState) -> dict:
    """Log completion (no Apprise in integration mode)."""
    return {"notified": True, "status": "completed"}


# ── Helpers ────────────────────────────────────────────────────────────

def _extract_metric(metric: str, metric_result, failure_counts: dict) -> float:
    """Extract a named metric value from MetricResult + failure_counts."""
    if metric == "p_at_1":
        return metric_result.p_at_k.get(1, 0) * 100
    elif metric == "r_at_8":
        return metric_result.r_at_k.get(8, 0) * 100
    elif metric.endswith("_pct"):
        field = metric.removesuffix("_pct").upper()
        total = sum(failure_counts.values())
        if total == 0:
            return 0.0
        return failure_counts.get(field, 0) / total * 100
    return 0.0


def _extract_metric_from_entry(metric: str, entry) -> float:
    """Extract a named metric value from a LedgerEntry."""
    if metric == "p_at_1":
        return entry.p_at_k.get(1, 0) * 100
    elif metric == "r_at_8":
        return entry.r_at_k.get(8, 0) * 100
    elif metric.endswith("_pct"):
        field = metric.removesuffix("_pct")
        return entry.failure_vector.pct(field)
    return 0.0
