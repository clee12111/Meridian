"""Typed state for the v2 per-query pipeline.

Every field is Optional (total=False) so the graph can start from an
empty state and each phase node populates only its own slice.
LangGraph merges each node's return dict into this state.
"""

from __future__ import annotations

from typing import Any, TypedDict

from core.measurement.metrics import MetricResult


class RetrievalBundle(TypedDict, total=False):
    """Carries dense + sparse results before fusion."""

    dense: dict    # RetrievalResult serialized: contents, ids, scores, spans
    sparse: dict   # RetrievalResult serialized: contents, ids, scores, spans


class ExperimentState(TypedDict, total=False):
    """Per-query pipeline state, phases 3–10.

    Each phase reads upstream fields and writes its own output slice.
    RetrievalResult is serialized as a plain dict (contents, ids, scores,
    spans) because TypedDict fields must be JSON-serializable for
    SqliteSaver checkpointing.
    """

    # --- identity ---
    experiment_id: str       # UUID for this run
    run_number: int          # ledger sequence number
    trace_id: str            # Langfuse trace id; "locked-baseline" for seed

    # --- query (Phase 3 input / output) ---
    raw_query: str           # original query, never mutated
    rewritten_query: str     # Phase 3 output; falls back to raw_query if Phase 3 is stub

    # --- retrieval (Phase 4 output) ---
    retrieval_bundle: RetrievalBundle

    # --- fusion (Phase 5 output) ---
    fused_result: dict       # RetrievalResult serialized

    # --- reranking (Phase 6 output) ---
    reranked_result: dict    # RetrievalResult serialized

    # --- context construction (Phase 7 output) ---
    context_chunks: list[str]   # ordered list of chunk texts for LLM
    context_ids: list[str]      # parallel chunk ids

    # --- synthesis (Phase 8 output) ---
    answer: str              # generated answer text
    claims: list[dict]       # structured: [{claim, cited_chunk_id, text}, ...]

    # --- verification (Phase 9 output) ---
    verification_result: dict   # {grounded: int, ungrounded: int, score: float}

    # --- agentic loop (Phase 10) ---
    iteration: int           # current loop count, starts at 1
    max_iterations: int      # stopping ceiling, default 3
    loop_complete: bool      # True when Phase 10 decides to stop

    # --- measurement (post-pipeline) ---
    metric_result: MetricResult
    failure_counts: dict[str, int]   # {drm, cbf, sgp, icr, ovr, ok}

    # --- control flow ---
    status: str              # "running" | "complete" | "error"
    error: str               # error message if status == "error"
