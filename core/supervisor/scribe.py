"""Scribe / Forensic Writer: narrates completed experiments into
decision_log.md entries and nightly Apprise email summaries.

Phase 3d.  The Scribe does NOT judge significance -- it is TOLD,
per metric, whether a delta exceeds VARIANCE_FLOOR_PP.  Sub-floor
deltas MUST be written as "within noise / no significant change."
The Scribe is forbidden from words like "improved," "gained," or
"better" for sub-floor deltas.

Model: DeepSeek (cheaper than Proposer -- narration, not strategy).
"""

from __future__ import annotations

import os
from typing import Literal

from openai import OpenAI

from core.supervisor.proposer import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, LOCKED_BASELINES
from core.supervisor.schemas import (
    VARIANCE_FLOOR_PP,
    FailureVector,
    LedgerEntry,
)


# ── Significance gating (deterministic, upstream of the LLM) ────────────

SignificanceVerdict = Literal["significant", "within_noise"]

FAILURE_FIELDS = ["drm", "cbf", "icr", "ovr", "ok"]
METRIC_KEYS = ["p_at_1", "r_at_8", "drm_pct", "cbf_pct", "icr_pct", "ovr_pct", "ok_pct"]


def _resolve_baseline_metric(
    metric: str,
    baseline_entry: LedgerEntry | None,
    locked_vals: dict | None,
) -> float | None:
    """Extract a metric value from a baseline entry or locked dict."""
    if baseline_entry is not None:
        if metric == "p_at_1":
            return baseline_entry.p_at_k.get(1, 0) * 100
        elif metric == "r_at_8":
            return baseline_entry.r_at_k.get(8, 0) * 100
        elif metric.endswith("_pct"):
            field = metric.removesuffix("_pct")
            return baseline_entry.failure_vector.pct(field)
    elif locked_vals is not None:
        return locked_vals.get(metric)
    return None


def _resolve_current_metric(metric: str, entry: LedgerEntry) -> float:
    """Extract a metric value from the current entry."""
    if metric == "p_at_1":
        return entry.p_at_k.get(1, 0) * 100
    elif metric == "r_at_8":
        return entry.r_at_k.get(8, 0) * 100
    elif metric.endswith("_pct"):
        field = metric.removesuffix("_pct")
        return entry.failure_vector.pct(field)
    return 0.0


def compute_significance_verdicts(
    entry: LedgerEntry,
    baseline_entry: LedgerEntry | None,
    locked_vals: dict | None,
) -> dict[str, dict]:
    """Compute per-metric significance verdicts DETERMINISTICALLY.

    Returns a dict keyed by metric name, each containing:
      - baseline_val: float
      - current_val: float
      - delta_pp: float
      - floor: float
      - verdict: "significant" | "within_noise"

    The Scribe receives these verdicts -- it does not compute them.
    """
    verdicts: dict[str, dict] = {}

    for metric in METRIC_KEYS:
        floor = VARIANCE_FLOOR_PP.get(metric, 0.6)
        current = _resolve_current_metric(metric, entry)
        baseline = _resolve_baseline_metric(metric, baseline_entry, locked_vals)

        if baseline is None:
            verdicts[metric] = {
                "baseline_val": None,
                "current_val": current,
                "delta_pp": None,
                "floor": floor,
                "verdict": "within_noise",
            }
            continue

        delta = current - baseline
        verdict: SignificanceVerdict = (
            "significant" if abs(delta) > floor else "within_noise"
        )
        verdicts[metric] = {
            "baseline_val": round(baseline, 2),
            "current_val": round(current, 2),
            "delta_pp": round(delta, 2),
            "floor": floor,
            "verdict": verdict,
        }

    return verdicts


def _format_config_diff(entry: LedgerEntry, baseline_entry: LedgerEntry | None, locked_vals: dict | None) -> str:
    """Show config params that differ from baseline."""
    cfg = entry.config
    diffs: list[str] = []

    if baseline_entry is not None:
        bcfg = baseline_entry.config
        if cfg.chunk_size != bcfg.chunk_size:
            diffs.append(f"chunk_size: {bcfg.chunk_size} -> {cfg.chunk_size}")
        if cfg.chunk_overlap != bcfg.chunk_overlap:
            diffs.append(f"chunk_overlap: {bcfg.chunk_overlap} -> {cfg.chunk_overlap}")
        if cfg.bm25_top_k != bcfg.bm25_top_k:
            diffs.append(f"bm25_top_k: {bcfg.bm25_top_k} -> {cfg.bm25_top_k}")
        if cfg.dense_top_k != bcfg.dense_top_k:
            diffs.append(f"dense_top_k: {bcfg.dense_top_k} -> {cfg.dense_top_k}")
        if cfg.fusion_top_n != bcfg.fusion_top_n:
            diffs.append(f"fusion_top_n: {bcfg.fusion_top_n} -> {cfg.fusion_top_n}")
        if cfg.retrieval_mode != bcfg.retrieval_mode:
            diffs.append(f"retrieval_mode: {bcfg.retrieval_mode} -> {cfg.retrieval_mode}")
    elif locked_vals is not None:
        if cfg.chunk_size != locked_vals.get("chunk_size", 512):
            diffs.append(f"chunk_size: {locked_vals.get('chunk_size', 512)} -> {cfg.chunk_size}")
        if cfg.chunk_overlap != locked_vals.get("chunk_overlap", 128):
            diffs.append(f"chunk_overlap: {locked_vals.get('chunk_overlap', 128)} -> {cfg.chunk_overlap}")
        # top_k defaults not in locked_vals, only show if non-default
        for param, default in [("bm25_top_k", 32), ("dense_top_k", 32), ("fusion_top_n", 64)]:
            val = getattr(cfg, param)
            if val != default:
                diffs.append(f"{param}: {default} -> {val}")

    if not diffs:
        return "(no config changes vs baseline)"
    return "\n".join(f"  {d}" for d in diffs)


def _format_verdicts_for_prompt(verdicts: dict[str, dict]) -> str:
    """Format significance verdicts as a table for the Scribe prompt."""
    lines = [
        "Metric       | Baseline | Current | Delta   | Floor | Verdict",
        "-------------|----------|---------|---------|-------|--------",
    ]
    for metric, v in verdicts.items():
        bl = f"{v['baseline_val']:.2f}%" if v["baseline_val"] is not None else "N/A"
        cur = f"{v['current_val']:.2f}%"
        delta = f"{v['delta_pp']:+.2f}pp" if v["delta_pp"] is not None else "N/A"
        lines.append(
            f"{metric:<12} | {bl:>8} | {cur:>7} | {delta:>7} | {v['floor']:.1f}pp | {v['verdict']}"
        )
    return "\n".join(lines)


def _format_failure_vector_comparison(
    entry: LedgerEntry,
    baseline_entry: LedgerEntry | None,
    locked_vals: dict | None,
) -> str:
    """Side-by-side failure vector: baseline vs current."""
    fv = entry.failure_vector
    lines = ["Type | Baseline | Current | Delta"]
    lines.append("-----|----------|---------|------")
    for field in FAILURE_FIELDS:
        cur_pct = fv.pct(field)
        if baseline_entry is not None:
            bl_pct = baseline_entry.failure_vector.pct(field)
        elif locked_vals is not None:
            bl_pct = locked_vals.get(f"{field}_pct", 0)
        else:
            bl_pct = 0
        delta = cur_pct - bl_pct
        lines.append(
            f"{field.upper():>4} | {bl_pct:>7.1f}% | {cur_pct:>6.1f}% | {delta:+.1f}pp"
        )
    return "\n".join(lines)


# ── Calibration summary ─────────────────────────────────────────────────


def _format_calibration(entry: LedgerEntry) -> str:
    """Format the predicted vs actual calibration block."""
    pd = entry.predicted_delta
    lines = [
        f"Predicted metric: {pd.metric}",
        f"Predicted delta:  {pd.delta_pp:+.2f}pp vs baseline_ref={pd.baseline_ref!r}",
    ]

    if entry.actual_delta_pp is not None:
        lines.append(f"Actual delta:     {entry.actual_delta_pp:+.2f}pp")
    else:
        lines.append("Actual delta:     N/A (baseline not resolved)")

    if entry.prediction_error_pp is not None:
        err = entry.prediction_error_pp
        lines.append(f"Prediction error: {err:+.2f}pp")
        if abs(err) <= VARIANCE_FLOOR_PP.get(pd.metric, 0.6):
            lines.append("Calibration:      ACCURATE (error within noise floor)")
        elif abs(pd.delta_pp) > abs(entry.actual_delta_pp or 0):
            # Predicted a bigger effect than observed
            lines.append("Calibration:      OVER-PREDICTED (Proposer too optimistic)")
        else:
            # Predicted a smaller effect than observed
            lines.append("Calibration:      UNDER-PREDICTED (Proposer too conservative)")
    else:
        lines.append("Prediction error: N/A")

    return "\n".join(lines)


# ── Scribe prompt ────────────────────────────────────────────────────────

SCRIBE_SYSTEM_PROMPT = """\
You are the Scribe in an autonomous retrieval experiment loop.
Your job: write a decision_log.md entry for one completed experiment.

## ABSOLUTE RULES

1. SIGNIFICANCE IS PRE-COMPUTED.  You are given a verdict table below.
   Metrics marked "within_noise" MUST be described as
   "within noise" or "no significant change."
   You are FORBIDDEN from using "improved," "gained," "better,"
   "increased," "decreased," or any directional language for
   within_noise metrics.  The number moved; the signal did not.

2. NARRATE FROM DATA ONLY.  Write only what is grounded in the
   config, metrics, failure vector, and calibration data provided.
   Do NOT speculate about WHY a metric moved unless the mechanism
   is directly implied by the config change (e.g. "increased top_k
   retrieves more candidates").  Do NOT invent specifics about
   query types, document content, or failure causes not shown here.

3. CALIBRATION IS THE POINT.  The entry must state plainly whether
   the Proposer's prediction was right, wrong, or within noise.
   This is what makes the log a research instrument.

## Output format

Write ONLY the decision_log.md entry in markdown.  No preamble,
no commentary outside the entry.  Use this structure:

```
## Run {run_number}: {one-line summary}

**Config:** {dataset} | chunk={size}/{overlap} | {mode} | bm25_k={} dense_k={} fusion_n={}
**Config diff vs {baseline_ref}:**
{diff}

**Hypothesis:** {verbatim from Proposer}

**Predicted delta:** {metric} {delta}pp vs {baseline_ref}

**Results:**
| Metric | Value |
|--------|-------|
| P@1    | ...%  |
| R@8    | ...%  |

**Failure vector:**
{table with baseline comparison}

**Significance verdicts:**
{which metrics moved significantly, which are noise}

**Calibration:**
{predicted vs actual vs error, was prediction accurate}

**Outcome:** {one line: CONFIRMED / PARTIALLY CONFIRMED / FALSIFIED / WITHIN NOISE}
```
"""


def _build_scribe_user_prompt(
    entry: LedgerEntry,
    baseline_entry: LedgerEntry | None,
    locked_vals: dict | None,
    verdicts: dict[str, dict],
) -> str:
    """Build the user prompt with all data the Scribe needs."""
    cfg = entry.config
    baseline_ref = entry.predicted_delta.baseline_ref

    config_diff = _format_config_diff(entry, baseline_entry, locked_vals)
    verdict_table = _format_verdicts_for_prompt(verdicts)
    fv_comparison = _format_failure_vector_comparison(entry, baseline_entry, locked_vals)
    calibration = _format_calibration(entry)

    return f"""\
Write the decision_log.md entry for this completed experiment.

## Experiment data

Run number: {entry.run_number}
Timestamp: {entry.timestamp:%Y-%m-%d %H:%M UTC}
Experiment type: {entry.experiment_type}
Chunks embedded: {entry.actual_embedding_chunks:,}

Config: {cfg.target_dataset} | chunk={cfg.chunk_size}/{cfg.chunk_overlap} | \
{cfg.retrieval_mode} | bm25_k={cfg.bm25_top_k} dense_k={cfg.dense_top_k} \
fusion_n={cfg.fusion_top_n}

Config diff vs {baseline_ref}:
{config_diff}

Hypothesis (verbatim from Proposer):
{entry.hypothesis}

Predicted delta: {entry.predicted_delta.metric} \
{entry.predicted_delta.delta_pp:+.2f}pp vs {baseline_ref}

## Results

P@1: {entry.p_at_k.get(1, 0) * 100:.2f}%
R@8: {entry.r_at_k.get(8, 0) * 100:.2f}%
n_queries: {entry.n_queries}

## Failure vector (vs {baseline_ref})

{fv_comparison}

## Pre-computed significance verdicts (DO NOT OVERRIDE)

{verdict_table}

## Calibration

{calibration}

Trace ID: {entry.trace_id}
"""


# ── LLM call ────────────────────────────────────────────────────────────


def write_decision_entry(
    entry: LedgerEntry,
    ledger: list[LedgerEntry],
) -> str:
    """Generate a decision_log.md entry for a completed experiment.

    Resolves baseline_ref, computes significance verdicts deterministically,
    then calls DeepSeek to narrate.

    Returns the markdown entry text.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY not set.  Required for the Scribe.")

    # Resolve baseline
    baseline_ref = entry.predicted_delta.baseline_ref
    baseline_entry: LedgerEntry | None = None
    locked_vals: dict | None = None

    if baseline_ref == "locked":
        ds = entry.config.target_dataset
        locked_vals = LOCKED_BASELINES.get(ds)
    else:
        # Find the named run_number in the ledger
        try:
            ref_num = int(baseline_ref)
            for e in ledger:
                if e.run_number == ref_num:
                    baseline_entry = e
                    break
        except ValueError:
            pass

    # Compute significance verdicts DETERMINISTICALLY
    verdicts = compute_significance_verdicts(entry, baseline_entry, locked_vals)

    # Build prompts
    user_prompt = _build_scribe_user_prompt(entry, baseline_entry, locked_vals, verdicts)

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SCRIBE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    return response.choices[0].message.content.strip()


def format_email_summary(
    entry: LedgerEntry,
    decision_entry: str,
) -> str:
    """Format a concise nightly email summary from the decision_log entry.

    No LLM call -- this is a deterministic template.
    """
    cfg = entry.config
    pd = entry.predicted_delta
    fv = entry.failure_vector

    # Headline verdict (compare magnitude of predicted vs actual effect)
    if entry.prediction_error_pp is not None:
        err = abs(entry.prediction_error_pp)
        floor = VARIANCE_FLOOR_PP.get(pd.metric, 0.6)
        if err <= floor:
            cal = "ACCURATE"
        elif abs(pd.delta_pp) > abs(entry.actual_delta_pp or 0):
            cal = "OVER-PREDICTED"
        else:
            cal = "UNDER-PREDICTED"
    else:
        cal = "N/A"

    return f"""\
Meridian Experiment Run {entry.run_number}
{'=' * 40}

Dataset:    {cfg.target_dataset}
Config:     chunk={cfg.chunk_size}/{cfg.chunk_overlap} {cfg.retrieval_mode} \
bm25_k={cfg.bm25_top_k} dense_k={cfg.dense_top_k}
Type:       {entry.experiment_type} ({entry.actual_embedding_chunks:,} chunks)

P@1: {entry.p_at_k.get(1, 0) * 100:.2f}%  R@8: {entry.r_at_k.get(8, 0) * 100:.2f}%

Failures: DRM={fv.pct('drm'):.1f}% CBF={fv.pct('cbf'):.1f}% \
ICR={fv.pct('icr'):.1f}% OVR={fv.pct('ovr'):.1f}% OK={fv.pct('ok'):.1f}%

Prediction: {pd.metric} {pd.delta_pp:+.1f}pp vs {pd.baseline_ref}
Actual:     {entry.actual_delta_pp:+.1f}pp
Calibration: {cal}

Trace: {entry.trace_id}
"""
