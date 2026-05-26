"""Pydantic v2 schemas for the Proposer, experiment config, and ledger.

Load-bearing types for Phase 3c.  The Proposer LLM emits
ExperimentProposal as structured output; the Supervisor stores a
LedgerEntry after each experiment completes.

Design drivers (hard-won from manual experiments):
  1. Cost is corpus-dependent — MAUD is 192k chunks vs ContractNLI 3.8k.
     The Proposer must see estimated cost BEFORE proposing.
  2. Variance floor exists — same config re-run drifts measurably.
     Deltas below the floor are noise, not signal.
  3. The failure vector (DRM/CBF/ICR/OVR/OK) is the primary signal,
     not headline P@1/R@8.  Every ledger entry carries the full vector.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ── Constants ────────────────────────────────────────────────────────────

# MEASURED — 7 runs of the locked 512/128 hybrid config on ContractNLI
# (194 queries, same Qdrant Cloud index).
#
# Root cause of R@8 variance: Voyage API query-time embedding
# nondeterminism.  Same text occasionally returns a slightly different
# vector (max_diff ~6.5e-3, cosine ~0.9988), reordering borderline
# chunks at RRF positions 4-8.  Affects 1-2 of 194 queries per run.
#
# P@1 is immune: rank-1 chunk is never displaced.
# Failure counts are immune: same chunks retrieved, only order at
# the R@8 boundary shifts.
#
# Measured (7 runs):
#   P@1: 8.84% every run (zero variance)
#   R@8: mean 50.29%, stdev 0.16pp, range [49.95, 50.41]
#   Failure counts: identical every run (DRM 76, CBF 35, SGP 21,
#                   ICR 6, OVR 21, OK 35)
#
# Floor set at 3-sigma (99.7% of run-to-run noise absorbed).
VARIANCE_FLOOR_PP: dict[str, float] = {
    "p_at_1": 0.00,   # deterministic (zero observed variance)
    "r_at_8": 0.50,   # 3-sigma = 0.48pp, rounded up to 0.50
    # Failure-vector percentages: zero observed variance across
    # 7 runs (all counts identical).  Floor set to the minimum
    # detectable change: 1 query on 194 = 0.52pp.
    "drm_pct": 0.52,
    "cbf_pct": 0.52,
    "sgp_pct": 0.52,
    "icr_pct": 0.52,
    "ovr_pct": 0.52,
    "ok_pct":  0.52,
}

# Exact chunk counts at 512/128 per dataset (from Qdrant point counts).
# The Proposer uses these + estimate_chunks() to reason about cost.
CORPUS_STATS: dict[str, dict[str, int]] = {
    "contractnli": {"n_docs": 95,  "total_chars": 1_460_000,  "chunks_at_512": 3_797},
    "cuad":        {"n_docs": 510, "total_chars": 37_000_000, "chunks_at_512": 96_256},
    "maud":        {"n_docs": 150, "total_chars": 52_721_337, "chunks_at_512": 192_650},
    "privacy_qa":  {"n_docs": 35,  "total_chars": 340_000,    "chunks_at_512": 620},
}

# Cost ranking (cheapest first) — the Proposer should test on the
# cheapest dataset that can falsify the hypothesis.
COST_ORDER = ["privacy_qa", "contractnli", "cuad", "maud"]

# Conservative multiplier for estimate_chunks().  The naive
# total_chars / stride underestimates by up to 40% on corpora with
# many sentence-boundary splits (observed: MAUD estimate was 29% low).
# We multiply by 1.45 and round up so the Proposer never under-budgets.
_ESTIMATE_SAFETY_FACTOR = 1.45


def estimate_chunks(dataset: str, chunk_size: int, chunk_overlap: int) -> int:
    """Conservative estimate of chunk count for a (dataset, chunk_size, overlap) triple.

    Uses total_chars / stride * safety factor, rounded UP.
    Biased high intentionally — an underestimate on the expensive
    dataset (MAUD) is the dangerous direction.  The Proposer must
    never be surprised by a bill larger than it budgeted.
    """
    stats = CORPUS_STATS.get(dataset)
    if stats is None:
        return 0
    stride = chunk_size - chunk_overlap
    if stride <= 0:
        raise ValueError(f"stride must be positive: {chunk_size=}, {chunk_overlap=}")
    naive = stats["total_chars"] / stride
    return math.ceil(naive * _ESTIMATE_SAFETY_FACTOR)


# ── Proposer output ─────────────────────────────────────────────────────

DatasetName = Literal["contractnli", "cuad", "maud", "privacy_qa"]

MetricName = Literal[
    "p_at_1", "r_at_8",
    "drm_pct", "cbf_pct", "sgp_pct", "icr_pct", "ovr_pct", "ok_pct",
]

ExperimentType = Literal["query_time", "ingestion_time"]


class RagConfig(BaseModel):
    """Retrieval parameters for one experiment.

    target_dataset is part of the config because the same parameters
    produce different results on different corpora — (config, dataset)
    is the unit of deduplication.

    chunk_size is a bounded int [128, 2048].  512 is the established
    legal-RAG baseline (RCTS); above 2048 is known-inefficient and
    OVR-spiking.  128 is the falsification floor (expected high
    CBF/ICR), not a candidate optimum.  The interesting search
    region is ~384-1024.  Specific values are suggested in the
    Proposer prompt, not enforced in the type.
    """

    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=128, ge=0)
    bm25_top_k: int = Field(default=32, ge=1, le=128)
    dense_top_k: int = Field(default=32, ge=1, le=128)
    fusion_top_n: int = Field(default=64, ge=1, le=128)
    retrieval_mode: Literal["bm25", "hybrid"] = "hybrid"
    target_dataset: DatasetName

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> RagConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    def canonical_json(self) -> str:
        """Deterministic JSON for hashing (sorted keys, no whitespace)."""
        return self.model_dump_json(exclude_none=True)


class PredictedDelta(BaseModel):
    """Quantitative prediction: which metric moves, by how much,
    measured against a NAMED baseline (not the adjacent ledger row).

    The Proposer must name which prior run (or the locked baseline)
    it predicts improvement against.  The Supervisor computes
    actual_delta against that specific reference so calibration
    measures prediction accuracy, not path-dependent drift.
    """

    metric: MetricName
    delta_pp: float = Field(
        description="Signed delta in percentage points.  Negative = drop."
    )
    baseline_ref: str = Field(
        description=(
            "Which baseline this delta is measured against.  Either "
            "'locked' (the permanent Phase 2 gate values) or a prior "
            "run_number as a string (e.g. '3').  The Supervisor looks "
            "up the named reference to compute actual_delta_pp."
        )
    )
    above_variance_floor: bool = Field(
        description=(
            "True iff abs(delta_pp) > VARIANCE_FLOOR_PP[metric].  "
            "Must be True or the experiment cannot produce a signal.  "
            "Note: VARIANCE_FLOOR_PP values are PROVISIONAL."
        )
    )


class ExperimentProposal(BaseModel):
    """Full structured output of the Proposer LLM.

    The Supervisor validates experiment_type and cost against
    the actual index state before running.
    """

    config: RagConfig

    hypothesis: str = Field(
        description=(
            "One paragraph: which failure mode this targets, why this "
            "config should move it, what tradeoff is expected."
        )
    )

    predicted_delta: PredictedDelta

    experiment_type: ExperimentType = Field(
        description=(
            "query_time: only retrieval params changed (top_k, fusion, "
            "mode) — reuses existing index, zero embedding cost.  "
            "ingestion_time: chunk_size or chunk_overlap changed — "
            "requires re-chunking and re-embedding."
        )
    )
    estimated_embedding_chunks: int = Field(
        ge=0,
        description="Chunks to embed.  0 for query_time.  "
                    "For ingestion_time, use estimate_chunks().",
    )

    cost_reasoning: str = Field(
        description=(
            "One sentence: why this is the cheapest experiment that "
            "tests the hypothesis.  Forces the Proposer to articulate "
            "cost awareness."
        )
    )


# ── Failure vector ───────────────────────────────────────────────────────

class FailureVector(BaseModel):
    """Per-query failure type counts for one (config, dataset) eval.

    Counts, not percentages — percentages are derived via .pct().
    """

    drm: int = Field(ge=0, description="Document Retrieval Miss")
    cbf: int = Field(ge=0, description="Correct But Failed (right doc, zero overlap)")
    sgp: int = Field(default=0, ge=0, description="Span Gap (some spans covered, others entirely missed)")
    icr: int = Field(ge=0, description="Incomplete Retrieval (overlap < 50%)")
    ovr: int = Field(ge=0, description="Over-Retrieval (retrieved >= 3x gt)")
    ok:  int = Field(ge=0, description="No failure")

    @property
    def total(self) -> int:
        return self.drm + self.cbf + self.sgp + self.icr + self.ovr + self.ok

    def pct(self, field: str) -> float:
        """Percentage for one failure type.  Returns 0.0 if total is 0."""
        t = self.total
        return getattr(self, field) / t * 100.0 if t else 0.0

    @classmethod
    def from_counts(cls, counts: dict[str, int]) -> FailureVector:
        """Build from the {FailureType.name: count} dict that
        evaluate_config returns."""
        return cls(
            drm=counts.get("DRM", 0),
            cbf=counts.get("CBF", 0),
            sgp=counts.get("SGP", 0),
            icr=counts.get("ICR", 0),
            ovr=counts.get("OVR", 0),
            ok=counts.get("OK", 0),
        )


# ── Ledger entry ─────────────────────────────────────────────────────────

class LedgerEntry(BaseModel):
    """One row in the experiment ledger.  Immutable after write.

    Each entry is one (config, dataset) evaluation: what was proposed,
    what was measured, and what it cost.
    """

    # ── identity ──
    run_number: int = Field(ge=1)
    experiment_id: str = Field(
        description="SHA-256 of config.canonical_json()"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── what was proposed ──
    config: RagConfig
    hypothesis: str
    predicted_delta: PredictedDelta
    experiment_type: ExperimentType
    estimated_embedding_chunks: int = Field(ge=0)

    # ── what was measured ──
    p_at_k: dict[int, float] = Field(
        description="Mean P@k: {1: 0.0924, 2: ..., 4: ..., 8: ..., ...}"
    )
    r_at_k: dict[int, float] = Field(
        description="Mean R@k: {1: ..., 2: ..., 4: ..., 8: 0.5041, ...}"
    )
    failure_vector: FailureVector
    n_queries: int = Field(ge=1)

    # ── cost accounting ──
    actual_embedding_chunks: int = Field(
        ge=0,
        description="Actual chunks embedded (0 if index was reused)",
    )

    # ── calibration ──
    # Computed by the Supervisor after the run, by comparing the
    # measured metric to the value in the run named by
    # predicted_delta.baseline_ref (a run_number or 'locked').
    actual_delta_pp: float | None = Field(
        default=None,
        description=(
            "Observed delta on predicted_delta.metric vs the run "
            "named in predicted_delta.baseline_ref.  None only if "
            "the baseline reference cannot be resolved."
        ),
    )
    prediction_error_pp: float | None = Field(
        default=None,
        description=(
            "predicted_delta.delta_pp - actual_delta_pp.  "
            "Positive = Proposer was too optimistic.  "
            "None if actual_delta_pp is None."
        ),
    )

    # ── tracing ──
    trace_id: str = Field(description="Langfuse trace ID")
