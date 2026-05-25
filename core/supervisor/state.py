"""Typed state for the Supervisor experiment loop.

Every field is Optional so the graph can start from an empty state
and each node populates only its own slice.
"""

from __future__ import annotations

from typing import TypedDict

from core.measurement.metrics import MetricResult
from core.measurement.taxonomy import FailureType


class RagConfig(TypedDict, total=False):
    """Retrieval configuration proposed for one experiment run."""

    chunk_size: int
    chunk_overlap: int
    bm25_top_k: int
    dense_top_k: int
    fusion_top_n: int


class SanityVerdict(TypedDict, total=False):
    """Outcome of post-eval sanity checks."""

    passed: bool
    violations: list[str]
    quarantined: bool


class ExperimentState(TypedDict, total=False):
    """Full state for a single Supervisor loop iteration.

    Every node reads what it needs, writes what it produces.
    LangGraph merges each node's return dict into this state.
    """

    # --- identity ---
    experiment_id: str          # unique hash of the RagConfig
    run_number: int             # monotonic, set by read_ledger

    # --- proposer outputs ---
    config: RagConfig
    hypothesis: str             # verbatim from proposer
    predicted_delta: float      # predicted metric improvement

    # --- guard rails ---
    config_hash: str            # SHA-256 of canonical config JSON
    duplicate: bool             # True if hash already in ledger
    spend_ok: bool              # True if budget allows this run

    # --- eval outputs ---
    metric_result: MetricResult
    failure_counts: dict[str, int]   # FailureType.name -> count
    trace_id: str               # Langfuse trace ID for this experiment

    # --- sanity ---
    sanity: SanityVerdict

    # --- scribe ---
    decision_entry: str         # markdown for decision_log.md
    notified: bool              # True after Apprise fires

    # --- control flow ---
    status: str                 # "running" | "completed" | "quarantined" | "aborted"
    error: str                  # non-empty on abort
