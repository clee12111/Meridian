"""Seed the ledger with the locked Phase 2 baseline as run 0.

The Phase 2 baseline was run outside the ledger system.  This script
records it so the Proposer and validation logic know an index exists.
"""

from dotenv import load_dotenv
load_dotenv()

from core.supervisor.proposer import LOCKED_BASELINES, append_ledger
from core.supervisor.schemas import (
    FailureVector, LedgerEntry, PredictedDelta, RagConfig,
)


def main():
    locked = LOCKED_BASELINES["contractnli"]

    cfg = RagConfig(
        chunk_size=locked["chunk_size"],
        chunk_overlap=locked["chunk_overlap"],
        retrieval_mode=locked["retrieval_mode"],
        target_dataset="contractnli",
        bm25_top_k=32,
        dense_top_k=32,
        fusion_top_n=64,
    )

    entry = LedgerEntry(
        run_number=1,
        experiment_id="locked-baseline-contractnli",
        config=cfg,
        hypothesis="Phase 2 locked baseline — recorded for ledger continuity.",
        predicted_delta=PredictedDelta(
            metric="r_at_8",
            delta_pp=0.0,
            baseline_ref="locked",
            above_variance_floor=False,
        ),
        experiment_type="ingestion_time",
        estimated_embedding_chunks=3797,
        p_at_k={
            1: locked["p_at_1"] / 100,
            2: 0.0, 4: 0.0,
            8: locked["r_at_8"] / 100,  # placeholder for non-headline k
            16: 0.0, 32: 0.0, 64: 0.0,
        },
        r_at_k={
            1: 0.0, 2: 0.0, 4: 0.0,
            8: locked["r_at_8"] / 100,
            16: 0.0, 32: 0.0, 64: 0.0,
        },
        failure_vector=FailureVector(
            drm=76, cbf=35, sgp=21, icr=6, ovr=21, ok=35,
        ),
        n_queries=locked["n_queries"],
        actual_embedding_chunks=3797,
        actual_delta_pp=0.0,
        prediction_error_pp=0.0,
        trace_id="locked-baseline",
    )

    append_ledger(entry)
    print(f"Seeded ledger with locked baseline (run 0)")
    print(f"  Config: {cfg.target_dataset} {cfg.chunk_size}/{cfg.chunk_overlap} {cfg.retrieval_mode}")
    print(f"  P@1={locked['p_at_1']}%  R@8={locked['r_at_8']}%")
    print(f"  Failures: DRM=76 CBF=35 SGP=21 ICR=6 OVR=21 OK=35")


if __name__ == "__main__":
    main()
