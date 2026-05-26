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
        "p_at_1": 8.84, "r_at_8": 50.41,
        "drm_pct": 39.2, "cbf_pct": 18.0, "sgp_pct": 10.8,
        "icr_pct": 3.1, "ovr_pct": 10.8, "ok_pct": 18.0,
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


# ── Calibration summary helpers ──────────────────────────────────────────


def _lever_for_entry(entry: LedgerEntry, prior: LedgerEntry | None) -> str:
    """Derive the primary lever changed from prior to this entry."""
    if prior is None:
        return "initial"
    cfg, pcfg = entry.config, prior.config
    if cfg.target_dataset != pcfg.target_dataset:
        return "dataset_switch"
    if cfg.chunk_size != pcfg.chunk_size or cfg.chunk_overlap != pcfg.chunk_overlap:
        return "chunking"
    if cfg.retrieval_mode != pcfg.retrieval_mode:
        return "retrieval_mode"
    if (
        cfg.bm25_top_k != pcfg.bm25_top_k
        or cfg.dense_top_k != pcfg.dense_top_k
        or cfg.fusion_top_n != pcfg.fusion_top_n
    ):
        return "top_k"
    return "same_config"


def _floor_verdict(entry: LedgerEntry) -> str:
    """SIGNAL / NOISE / WRONG_DIR / no_ref based on actual_delta_pp."""
    if entry.actual_delta_pp is None:
        return "no_ref"
    metric = entry.predicted_delta.metric
    floor = VARIANCE_FLOOR_PP.get(metric, 0.52)
    adp = entry.actual_delta_pp
    if abs(adp) <= floor:
        return "NOISE"
    if (entry.predicted_delta.delta_pp > 0 and adp < 0) or (
        entry.predicted_delta.delta_pp < 0 and adp > 0
    ):
        return "WRONG_DIR"
    return "SIGNAL"


def _detect_saturated_levers(
    ledger: list[LedgerEntry], consecutive_threshold: int = 2
) -> list[str]:
    """Return lever names tried >= threshold consecutive times without SIGNAL.

    Only looks at the trailing window — saturation must be CURRENT, not
    something that happened mid-ledger and was already resolved.
    """
    if len(ledger) < consecutive_threshold:
        return []

    pairs: list[tuple[str, str]] = []
    for i, entry in enumerate(ledger):
        prior = ledger[i - 1] if i > 0 else None
        pairs.append((_lever_for_entry(entry, prior), _floor_verdict(entry)))

    tail = pairs[-consecutive_threshold:]
    levers = [p[0] for p in tail]
    verdicts = [p[1] for p in tail]

    if len(set(levers)) == 1 and all(v != "SIGNAL" for v in verdicts):
        return [levers[0]]
    return []


def _build_calibration_summary(ledger: list[LedgerEntry], n: int = 5) -> str:
    """Compact calibration table for the last N runs — injected into user message."""
    if not ledger:
        return "(no prior runs -- this is the first experiment)"

    recent = ledger[-n:]

    header = (
        f"{'run':>3} | {'dataset':<11} | {'lever':<14} | "
        f"{'predicted':>11} | {'actual':>8} | {'error':>8} | verdict"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for entry in recent:
        idx = next(
            (i for i, e in enumerate(ledger) if e.run_number == entry.run_number), -1
        )
        prior = ledger[idx - 1] if idx > 0 else None
        lever = _lever_for_entry(entry, prior)
        verdict = _floor_verdict(entry)
        metric = entry.predicted_delta.metric if entry.predicted_delta else "?"
        pred = (
            f"{entry.predicted_delta.delta_pp:+.1f}pp({metric})"
            if entry.predicted_delta
            else "?"
        )
        actual = (
            f"{entry.actual_delta_pp:+.1f}pp"
            if entry.actual_delta_pp is not None
            else "no_ref"
        )
        error = (
            f"{entry.prediction_error_pp:+.1f}pp"
            if entry.prediction_error_pp is not None
            else "no_ref"
        )
        lines.append(
            f"{entry.run_number:>3} | {entry.config.target_dataset:<11} | "
            f"{lever:<14} | {pred:>11} | {actual:>8} | {error:>8} | {verdict}"
        )

    saturated = _detect_saturated_levers(ledger)
    if saturated:
        lines.append("")
        lines.append(
            f"SATURATED (>= 2 consecutive non-SIGNAL): {', '.join(saturated)}"
        )
        lines.append(
            "You MUST NOT propose these levers again. Pivot to a different"
            " intervention class."
        )

    return "\n".join(lines)


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
    "metric": "<one of: p_at_1, r_at_8, drm_pct, cbf_pct, sgp_pct, icr_pct, ovr_pct, ok_pct>",
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
   Read the failure vector (DRM/CBF/SGP/ICR/OVR/OK).  The largest
   percentage is the dominant failure.

2. DRM (Document Retrieval Miss) is a routing/query-time failure.
   Fix with: retrieval_mode switch (bm25-only vs hybrid), hybrid weight
   skew (bm25_top_k >> dense_top_k), lower fusion_top_n.
   BLOCKED: top_k increases for DRM on ContractNLI -- see Finding 2 below.
   These are query_time experiments -- ZERO embedding cost.

3. SGP (Span Gap) is a coverage/diversity failure -- retriever
   found some required spans but missed others entirely (same doc).
   Fix with: diversity-aware retrieval, MMR reranking, higher top_k
   to surface more distinct chunks.  Can be query_time (top_k) or
   ingestion_time (chunk size/overlap to create more distinct chunks).

5. CBF/ICR/OVR are chunking/ingestion failures.
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

6. OPTIMIZE ROUTING (DRM) FIRST.  Chunking failures can't be
   addressed until the retriever is finding the right documents.
   Routing fixes are cheap.  Do them first.

7. PREFER CHEAP EXPERIMENTS.
   - query_time = free (reuses existing index).
   - ingestion_time = costs Voyage tokens (chunk count below).
   Always ask: "Can I falsify this cheaply on ContractNLI (~3,800
   chunks) or PrivacyQA (~620 chunks) before paying to confirm on
   MAUD (~192,000 chunks)?"

8. RESPECT THE VARIANCE FLOOR.
   These values are PROVISIONAL (not yet measured at statistical rigor):
{variance_floor}
   If abs(delta_pp) <= floor for the predicted metric, the experiment
   CANNOT distinguish signal from noise.  Propose a larger intervention
   or target a different metric.  Set above_variance_floor=false if
   the predicted delta is below the floor -- the Supervisor will reject.

9. MOVE TO THE NEXT DATASET only when the dominant failure on the
   current dataset has been addressed (cleared a gate or shifted to
   a different dominant type).

10. NAME YOUR BASELINE.  predicted_delta.baseline_ref must be either
   "locked" (the permanent Phase 2 baselines below) or a specific
   prior run_number (e.g. "3").  The Supervisor computes actual
   improvement against THAT reference, not the adjacent row.

## Anti-repetition rules (enforced, not advisory)

11. DEPRIORITIZE AFTER FAILURE.  If the last run targeted a lever and
    actual_delta was NOISE or WRONG_DIR, that lever is deprioritized.
    Propose a different lever or failure type next.

12. SATURATED LEVER = BLOCKED.  If the calibration history (provided
    in your user message each call) shows a lever tried >= 2 consecutive
    times with no SIGNAL, it is SATURATED.  The summary will say
    "SATURATED: <lever>".  You MUST NOT propose that lever again.

13. PIVOT AFTER SATURATION.  Priority order:
    - DRM dominant + top_k saturated:
        first try retrieval_mode="bm25" (isolate dense noise);
        then try bm25_top_k >> dense_top_k (e.g., 64/16) to skew hybrid;
        then try lower fusion_top_n (tighter discrimination window).
    - CBF/ICR dominant: pivot to chunk_size / chunk_overlap changes.
    - OVR dominant: reduce fusion_top_n or chunk_size.
    - SGP dominant: increase fusion_top_n or try MMR-style diverse retrieval.

## Finding 2 — DRM on ContractNLI is discrimination-bound (HARD RULE)

Evidence: top_k experiments ran at 64 / 96 / 128.  DRM count was
72 / 72 / 73 across all three -- completely flat.  OVR rose from 21 to 26.
P@1 gained one step (8.84 -> 11.44) at the 32->64 transition, then held.

Conclusion: the correct document IS in the candidate set at top_k=64.
The retriever fails to RANK it first, not to include it.  Adding more
candidates does not fix a ranking failure.

RULE: top_k increases for DRM reduction on ContractNLI are BLOCKED.
Next DRM experiments in priority order:
  1. retrieval_mode="bm25" -- isolate whether Voyage dense embeddings
     are hurting discrimination on legal near-duplicate documents.
  2. bm25_top_k >> dense_top_k (e.g., 64/16, fusion_top_n=32) -- reduce
     the fraction of dense noise entering RRF.
  3. Lower fusion_top_n (e.g., 32) at current bm25_k/dense_k -- tighter
     fusion window may improve rank-1 precision.

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
    calibration_block = _build_calibration_summary(ledger)

    user_message = (
        "## Recent calibration history (last 5 runs)\n\n"
        f"{calibration_block}\n\n"
        "Propose the next experiment. Respond with a single JSON object, "
        "no markdown fences, no commentary."
    )

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
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
