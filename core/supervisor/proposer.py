"""Proposer agent: reads the experiment ledger, calls DeepSeek,
emits an ExperimentProposal (next config + hypothesis + predicted delta).

Phase 3c.  The Proposer is the intelligence of the autonomous loop.
It does NOT run experiments -- it proposes what to run next and why.

Model: DeepSeek (OpenAI-compatible API).
"""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

from core.supervisor.schemas import (
    CORPUS_STATS,
    COST_ORDER,
    VARIANCE_FLOOR_PP,
    ExperimentProposal,
    LedgerEntry,
    estimate_chunks,
)

DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

LEDGER_PATH = Path("data/ledger.jsonl")

# Locked baselines — the Proposer references these as baseline_ref="locked".
LOCKED_BASELINES: dict[str, dict] = {
    "contractnli": {
        "chunk_size": 512, "chunk_overlap": 128,
        "retrieval_mode": "hybrid",
        "p_at_1": 9.24, "r_at_8": 50.41,
        "drm_pct": 39.7, "cbf_pct": 18.0, "icr_pct": 5.7,
        "ovr_pct": 12.4, "ok_pct": 24.2,
        "n_queries": 194,
    },
}

# ── Ledger I/O ──────────────────────────────────────────────────────────


def load_ledger(path: Path = LEDGER_PATH) -> list[LedgerEntry]:
    """Load ledger entries from a JSONL file (one JSON object per line)."""
    if not path.exists():
        return []
    entries: list[LedgerEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(LedgerEntry.model_validate_json(line))
    return entries


def append_ledger(entry: LedgerEntry, path: Path = LEDGER_PATH) -> None:
    """Append one entry to the ledger JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


# ── Prompt construction ─────────────────────────────────────────────────


def _format_corpus_stats() -> str:
    lines = ["Dataset         | Docs | Chunks@512 | Cost rank"]
    lines.append("-" * 52)
    for i, ds in enumerate(COST_ORDER):
        s = CORPUS_STATS[ds]
        lines.append(
            f"{ds:<15} | {s['n_docs']:>4} | {s['chunks_at_512']:>10,} | "
            f"{i + 1} ({'cheapest' if i == 0 else 'most expensive' if i == len(COST_ORDER) - 1 else ''})"
        )
    return "\n".join(lines)


def _format_variance_floor() -> str:
    lines = []
    for metric, floor in VARIANCE_FLOOR_PP.items():
        lines.append(f"  {metric}: {floor} pp")
    return "\n".join(lines)


def _format_locked_baselines() -> str:
    lines = []
    for ds, vals in LOCKED_BASELINES.items():
        lines.append(
            f"{ds} ({vals['chunk_size']}/{vals['chunk_overlap']} {vals['retrieval_mode']}):\n"
            f"  P@1: {vals['p_at_1']}%  R@8: {vals['r_at_8']}%\n"
            f"  DRM: {vals['drm_pct']}%  CBF: {vals['cbf_pct']}%  "
            f"ICR: {vals['icr_pct']}%  OVR: {vals['ovr_pct']}%  OK: {vals['ok_pct']}%\n"
            f"  n_queries: {vals['n_queries']}"
        )
    return "\n".join(lines)


def _format_ledger_entry(e: LedgerEntry) -> str:
    fv = e.failure_vector
    return (
        f"Run {e.run_number} [{e.timestamp:%Y-%m-%d %H:%M}] "
        f"({e.experiment_type}, {e.actual_embedding_chunks:,} chunks embedded)\n"
        f"  Config: {e.config.target_dataset} | "
        f"chunk={e.config.chunk_size}/{e.config.chunk_overlap} | "
        f"{e.config.retrieval_mode} | "
        f"bm25_k={e.config.bm25_top_k} dense_k={e.config.dense_top_k} "
        f"fusion_n={e.config.fusion_top_n}\n"
        f"  Hypothesis: {e.hypothesis}\n"
        f"  Predicted: {e.predicted_delta.metric} {e.predicted_delta.delta_pp:+.1f}pp "
        f"vs {e.predicted_delta.baseline_ref}\n"
        f"  Measured: P@1={e.p_at_k.get(1, 0) * 100:.2f}%  "
        f"R@8={e.r_at_k.get(8, 0) * 100:.2f}%  (n={e.n_queries})\n"
        f"  Failures: DRM={fv.pct('drm'):.1f}% CBF={fv.pct('cbf'):.1f}% "
        f"ICR={fv.pct('icr'):.1f}% OVR={fv.pct('ovr'):.1f}% OK={fv.pct('ok'):.1f}%\n"
        f"  Calibration: actual_delta={e.actual_delta_pp}pp  "
        f"error={e.prediction_error_pp}pp"
    )


def _format_ledger(entries: list[LedgerEntry]) -> str:
    if not entries:
        return "(empty -- this is the first experiment)"
    return "\n\n".join(_format_ledger_entry(e) for e in entries)


def _estimate_table() -> str:
    """Show chunk estimates at several sizes for cost reasoning."""
    sizes = [256, 512, 768, 1024, 2048]
    header = f"{'Dataset':<15}" + "".join(f" | {s:>6}" for s in sizes)
    lines = [header, "-" * len(header)]
    for ds in COST_ORDER:
        row = f"{ds:<15}"
        for cs in sizes:
            overlap = cs // 4
            est = estimate_chunks(ds, cs, overlap)
            row += f" | {est:>6,}"
        lines.append(row)
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are the Proposer agent in an autonomous retrieval experiment loop.
Your job: read the experiment ledger and propose the NEXT experiment
to run.  You do NOT run experiments -- you decide WHAT to run and WHY.

## Output format

Respond with a single JSON object matching this schema EXACTLY:
{{
  "config": {{
    "chunk_size": <int 128-2048>,
    "chunk_overlap": <int, must be < chunk_size>,
    "bm25_top_k": <int 1-128>,
    "dense_top_k": <int 1-128>,
    "fusion_top_n": <int 1-128>,
    "retrieval_mode": "bm25" or "hybrid",
    "target_dataset": "contractnli" | "cuad" | "maud" | "privacy_qa"
  }},
  "hypothesis": "<one paragraph>",
  "predicted_delta": {{
    "metric": "<one of: p_at_1, r_at_8, drm_pct, cbf_pct, icr_pct, ovr_pct, ok_pct>",
    "delta_pp": <float, signed percentage points>,
    "baseline_ref": "<'locked' or a prior run_number as string, e.g. '3'>",
    "above_variance_floor": <bool>
  }},
  "experiment_type": "query_time" or "ingestion_time",
  "estimated_embedding_chunks": <int, 0 for query_time>,
  "cost_reasoning": "<one sentence>"
}}

## Decision policy

1. TARGET THE DOMINANT FAILURE on the dataset where it dominates.
   Read the failure vector (DRM/CBF/ICR/OVR/OK).  The largest
   percentage is the dominant failure.

2. DRM (Document Retrieval Miss) is a routing/query-time failure.
   Fix with: top_k changes, hybrid weighting, retrieval_mode switch.
   These are query_time experiments -- ZERO embedding cost.

3. CBF/ICR/OVR are chunking/ingestion failures.
   Fix with: chunk_size, chunk_overlap changes.
   These are ingestion_time experiments -- cost is proportional to
   chunk count (real Voyage API tokens).
   chunk_size guidance:
     - 512 is the established legal-RAG baseline (RCTS paper).
     - The interesting search region is ~384-1024, centered on 512.
       This is where conditional-clause survival trades off against
       over-retrieval.
     - 128 is the falsification floor: use it to confirm that
       small chunks produce high CBF/ICR, not as a candidate optimum.
     - Above 1024 expect diminishing returns and OVR spikes,
       especially on MAUD's huge docs.  2048 is the hard ceiling.

4. OPTIMIZE ROUTING (DRM) FIRST.  Chunking failures can't be
   addressed until the retriever is finding the right documents.
   Routing fixes are cheap.  Do them first.

5. PREFER CHEAP EXPERIMENTS.
   - query_time = free (reuses existing index).
   - ingestion_time = costs Voyage tokens (chunk count below).
   Always ask: "Can I falsify this cheaply on ContractNLI (~3,800
   chunks) or PrivacyQA (~620 chunks) before paying to confirm on
   MAUD (~192,000 chunks)?"

6. RESPECT THE VARIANCE FLOOR.
   These values are PROVISIONAL (not yet measured at statistical rigor):
{variance_floor}
   If abs(delta_pp) <= floor for the predicted metric, the experiment
   CANNOT distinguish signal from noise.  Propose a larger intervention
   or target a different metric.  Set above_variance_floor=false if
   the predicted delta is below the floor -- the Supervisor will reject.

7. MOVE TO THE NEXT DATASET only when the dominant failure on the
   current dataset has been addressed (cleared a gate or shifted to
   a different dominant type).

8. NAME YOUR BASELINE.  predicted_delta.baseline_ref must be either
   "locked" (the permanent Phase 2 baselines below) or a specific
   prior run_number (e.g. "3").  The Supervisor computes actual
   improvement against THAT reference, not the adjacent row.

## Cost model

{corpus_stats}

Estimated chunks at various sizes (25% overlap, conservative estimate):
{estimate_table}

Cost order (cheapest first): {cost_order}

## Locked baselines (baseline_ref="locked")

{locked_baselines}

## Experiment ledger (prior runs)

{ledger}
"""


# ── LLM call ────────────────────────────────────────────────────────────


def _build_prompt(ledger: list[LedgerEntry]) -> str:
    return SYSTEM_PROMPT.format(
        variance_floor=_format_variance_floor(),
        corpus_stats=_format_corpus_stats(),
        estimate_table=_estimate_table(),
        cost_order=" -> ".join(COST_ORDER),
        locked_baselines=_format_locked_baselines(),
        ledger=_format_ledger(ledger),
    )


def propose(ledger_path: Path = LEDGER_PATH) -> ExperimentProposal:
    """Load the ledger, call DeepSeek, return a validated proposal.

    Raises
    ------
    EnvironmentError
        If DEEPSEEK_API_KEY is not set.
    ValueError
        If the LLM response fails Pydantic validation.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY not set.  Required for the Proposer."
        )

    ledger = load_ledger(ledger_path)
    system_prompt = _build_prompt(ledger)

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Propose the next experiment.  Respond with a single "
                    "JSON object, no markdown fences, no commentary."
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()

    # Parse and validate against Pydantic schema
    proposal = ExperimentProposal.model_validate_json(raw)

    # Cross-check: experiment_type vs config
    _validate_experiment_type(proposal, ledger)

    return proposal


def _validate_experiment_type(
    proposal: ExperimentProposal, ledger: list[LedgerEntry]
) -> None:
    """Verify the Proposer's experiment_type claim is consistent.

    If chunk_size or chunk_overlap differ from the most recent run
    on the same dataset, it MUST be ingestion_time.  If the Proposer
    claims query_time but changed chunking params, override.
    """
    cfg = proposal.config

    # Find most recent run on same dataset
    prior = None
    for entry in reversed(ledger):
        if entry.config.target_dataset == cfg.target_dataset:
            prior = entry
            break

    if prior is None:
        # No prior run on this dataset -- must be ingestion_time
        # (need to build the index from scratch)
        if proposal.experiment_type == "query_time":
            raise ValueError(
                f"No prior run on {cfg.target_dataset} -- cannot be query_time "
                f"(no index exists to reuse)."
            )
        return

    chunking_changed = (
        cfg.chunk_size != prior.config.chunk_size
        or cfg.chunk_overlap != prior.config.chunk_overlap
    )

    if chunking_changed and proposal.experiment_type == "query_time":
        raise ValueError(
            f"Proposer claimed query_time but chunk_size/overlap changed "
            f"({prior.config.chunk_size}/{prior.config.chunk_overlap} -> "
            f"{cfg.chunk_size}/{cfg.chunk_overlap}).  "
            f"This requires re-embedding (ingestion_time)."
        )
