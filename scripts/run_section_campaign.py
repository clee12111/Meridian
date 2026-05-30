#!/usr/bin/env python
"""Sequential section-chunking A/B campaign: MAUD -> ContractNLI -> PrivacyQA.

Gated at every step. Fails loudly and stops if any gate check fails.
Logs timestamped progress to data/section_campaign_log.md.

Usage:
    python scripts/run_section_campaign.py           # Full run
    python scripts/run_section_campaign.py --dry-run  # Preflight checks only
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "data" / "section_campaign_log.md"

# ── Per-corpus configs ────────────────────────────────────────────────────

CORPORA = [
    {
        "name": "maud",
        "dataset_name": "maud",
        "corpus_dir": "maud",
        "cc_alpha": 0.2,
        "routing_alpha": 0.7,
        "routing_index": "data/routing_index_maud_v4.npz",
        "summaries": "data/sac_summaries_maud.json",
        "benchmark": "data/benchmarks/maud.json",
        "baseline_collection": "maud_sac_v4",
        "section_collection": "maud_section_sac_v4",
        "baseline_parquet": "data/corpus_maud.parquet",
        "section_parquet": "data/corpus_maud_section.parquet",
        "section_sac_parquet": "data/corpus_maud_section_sac.parquet",
        "doc_type": "merger agreement",
        "note": "Real generalization test: CBF 15.5% + 20.6% PARTIAL gap",
    },
    {
        "name": "contractnli",
        "dataset_name": "contractnli",
        "corpus_dir": "contractnli",
        "cc_alpha": 0.2,
        "routing_alpha": 0.3,
        "routing_index": "data/routing_index_v4.npz",
        "summaries": "data/sac_summaries.json",
        "benchmark": "data/benchmarks/contractnli.json",
        "baseline_collection": "contractnli_sac_v4",
        "section_collection": "contractnli_section_sac_v4",
        "baseline_parquet": "data/corpus_contractnli.parquet",
        "section_parquet": "data/corpus_contractnli_section.parquet",
        "section_sac_parquet": "data/corpus_contractnli_section_sac.parquet",
        "doc_type": "NDA",
        "note": "Negative control: DRM-bound, expect little CBF benefit",
    },
    {
        "name": "privacyqa",
        "dataset_name": "privacy_qa",
        "corpus_dir": "privacy_qa",
        "cc_alpha": 0.1,
        "routing_alpha": 0.5,
        "routing_index": "data/routing_index_privacyqa_v4.npz",
        "summaries": "data/sac_summaries_privacyqa.json",
        "benchmark": "data/benchmarks/privacy_qa.json",
        "baseline_collection": "privacyqa_sac_v4",
        "section_collection": "privacyqa_section_sac_v4",
        "baseline_parquet": "data/corpus_privacy_qa.parquet",
        "section_parquet": "data/corpus_privacy_qa_section.parquet",
        "section_sac_parquet": "data/corpus_privacy_qa_section_sac.parquet",
        "doc_type": "privacy policy",
        "note": "Graceful-degradation check: prose, should ~= baseline, must not hurt",
    },
]

EXPECTED_QUERIES = 194


# ── Logging ───────────────────────────────────────────────────────────────


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, *, level: str = "INFO") -> None:
    """Print and append to log file."""
    line = f"[{_ts()}] [{level}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_md(block: str) -> None:
    """Append raw markdown block to log (no timestamp prefix)."""
    print(block, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(block + "\n")


def gate(condition: bool, msg: str) -> None:
    """Hard gate: if condition is False, log FATAL and exit."""
    if not condition:
        log(f"GATE FAILED: {msg}", level="FATAL")
        log("Campaign stopped. See log above for details.", level="FATAL")
        sys.exit(1)
    log(f"GATE PASSED: {msg}")


# ── Step 1: Build section parquet ─────────────────────────────────────────


def step_1_section_parquet(cfg: dict) -> int:
    """Build section-aware corpus parquet. Returns chunk count."""
    name = cfg["name"]
    output = ROOT / cfg["section_parquet"]

    if output.exists():
        import pandas as pd
        df = pd.read_parquet(output)
        log(f"[{name}] Section parquet already exists: {len(df)} chunks at {output}")
        # Re-validate offset integrity
        _validate_offsets(cfg, df)
        return len(df)

    log(f"[{name}] Step 1: Building section parquet...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_section_chunks.py"), name],
        capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8",
    )

    # Log the output
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log(f"  [chunker] {line}")

    gate(result.returncode == 0, f"{name} section chunker exited cleanly")
    gate(output.exists(), f"{name} section parquet written at {output}")

    import pandas as pd
    df = pd.read_parquet(output)
    count = len(df)
    log(f"[{name}] Section parquet: {count} chunks")
    return count


def _validate_offsets(cfg: dict, df) -> None:
    """Re-run offset round-trip check on existing parquet."""
    name = cfg["name"]
    corpus_dir = ROOT / "data" / "corpus" / cfg["corpus_dir"]
    failures = 0
    checked = 0
    for doc_id in df["doc_id"].unique():
        fname = doc_id.split("/", 1)[1] if "/" in doc_id else doc_id
        txt_path = corpus_dir / fname
        if not txt_path.exists():
            continue
        source = txt_path.read_text(encoding="utf-8", errors="replace")
        doc_df = df[df["doc_id"] == doc_id]
        for _, row in doc_df.iterrows():
            se = row["start_end_idx"]
            start, end = int(se[0]), int(se[1])
            expected = source[start:end]
            if expected != row["content"]:
                failures += 1
            checked += 1
    gate(failures == 0, f"{name} offset round-trip: {checked} checked, {failures} failures")


# ── Step 2: Build SAC parquet ─────────────────────────────────────────────


def step_2_sac_parquet(cfg: dict) -> int:
    """Add cached SAC summaries to section parquet. Returns chunk count."""
    name = cfg["name"]
    section_path = ROOT / cfg["section_parquet"]
    sac_path = ROOT / cfg["section_sac_parquet"]
    summaries_path = ROOT / cfg["summaries"]

    if sac_path.exists():
        import pandas as pd
        df = pd.read_parquet(sac_path)
        if "sac_content" in df.columns:
            log(f"[{name}] SAC parquet already exists: {len(df)} chunks")
            return len(df)

    log(f"[{name}] Step 2: Building SAC parquet from cached summaries...")

    import pandas as pd
    section_df = pd.read_parquet(section_path)
    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))

    # Check doc_id coverage
    doc_ids = set(section_df["doc_id"].unique())
    summary_ids = set(summaries.keys())
    missing = doc_ids - summary_ids
    if missing:
        log(f"[{name}] WARNING: {len(missing)} doc_ids without summaries "
            f"(first 3: {sorted(missing)[:3]}). Using raw content for these.")

    section_df = section_df.copy()
    section_df["sac_content"] = section_df.apply(
        lambda row: (
            f"[Document: {summaries[row['doc_id']]}]\n\n{row['content']}"
            if row["doc_id"] in summaries and summaries[row["doc_id"]]
            else row["content"]
        ),
        axis=1,
    )
    section_df.to_parquet(sac_path, index=False)
    count = len(section_df)
    log(f"[{name}] SAC parquet: {count} chunks -> {sac_path}")

    gate("sac_content" in section_df.columns, f"{name} SAC parquet has sac_content column")
    gate(count > 0, f"{name} SAC parquet is non-empty")
    return count


# ── Step 3: Index to Qdrant ───────────────────────────────────────────────


def step_3_index(cfg: dict, expected_count: int) -> None:
    """Embed and index section SAC parquet into a new Qdrant collection."""
    name = cfg["name"]
    collection = cfg["section_collection"]
    sac_path = ROOT / cfg["section_sac_parquet"]
    baseline_collection = cfg["baseline_collection"]

    from dotenv import load_dotenv
    load_dotenv()

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Check baseline untouched
    baseline_info = client.get_collection(baseline_collection)
    baseline_count = baseline_info.points_count
    log(f"[{name}] Baseline {baseline_collection}: {baseline_count} points (untouched)")

    # Check if section collection already complete
    try:
        info = client.get_collection(collection)
        existing = info.points_count or 0
        if existing == expected_count:
            log(f"[{name}] Section collection already complete: {existing} points, skipping")
            return
        log(f"[{name}] Section collection exists with {existing}/{expected_count} points — "
            f"resuming")
    except Exception:
        existing = 0
        log(f"[{name}] Creating section collection {collection}...")
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=collection,
            field_name="dataset_name",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection,
            field_name="chunk_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    log(f"[{name}] Step 3: Indexing {expected_count} chunks with voyage-4...")

    import pandas as pd
    corpus_df = pd.read_parquet(sac_path)

    # Find pending chunks for resume
    if existing > 0:
        log(f"[{name}] Scanning existing chunk_ids for resume...")
        existing_ids: set[str] = set()
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=collection, limit=1000,
                offset=offset, with_payload=["chunk_id"], with_vectors=False,
            )
            for pt in results:
                cid = pt.payload.get("chunk_id")
                if cid:
                    existing_ids.add(cid)
            if offset is None:
                break
        pending_df = corpus_df[~corpus_df["chunk_id"].isin(existing_ids)]
        log(f"[{name}] Found {len(existing_ids)} indexed, {len(pending_df)} pending")
    else:
        pending_df = corpus_df

    if len(pending_df) == 0:
        log(f"[{name}] All chunks already indexed")
    else:
        # Set embed workers/sleep env
        os.environ["MERIDIAN_EMBED_WORKERS"] = "6"
        os.environ["MERIDIAN_EMBED_SLEEP"] = "0"

        from core.ingestion.embed_index import embed_and_upsert

        chunks = pending_df.to_dict("records")
        try:
            embed_and_upsert(
                chunks=chunks,
                client=client,
                collection=collection,
                model="voyage-4",
                dataset_name=cfg["dataset_name"],
            )
        except RuntimeError as e:
            if "COMPLETENESS" in str(e):
                # Expected on resume: assertion compares pending vs total
                pass
            else:
                raise

    # Final completeness gate
    final_info = client.get_collection(collection)
    actual = final_info.points_count or 0
    gate(actual == expected_count,
         f"{name} collection completeness: {actual}/{expected_count} points")

    # Confirm baseline still untouched
    baseline_after = client.get_collection(baseline_collection)
    gate(baseline_after.points_count == baseline_count,
         f"{name} baseline {baseline_collection} untouched: "
         f"{baseline_after.points_count} == {baseline_count}")

    log(f"[{name}] Indexing complete: {actual} points in {collection}")


# ── Step 4: Run eval (one arm) ────────────────────────────────────────────


def _run_eval(cfg: dict, arm: str, collection: str, parquet: str,
              output: str) -> None:
    """Run run_corpus_eval.py for one arm. Gates on output file."""
    name = cfg["name"]
    output_path = ROOT / output

    if output_path.exists():
        count = sum(1 for _ in open(output_path, encoding="utf-8"))
        if count == EXPECTED_QUERIES:
            log(f"[{name}] {arm} eval already exists: {count} records at {output}")
            return
        log(f"[{name}] {arm} eval incomplete ({count}/{EXPECTED_QUERIES}), re-running")

    log(f"[{name}] Step 4: Running {arm} eval ({collection})...")

    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_corpus_eval.py"),
        "--corpus", cfg["name"],
        "--collection", collection,
        "--parquet", str(ROOT / parquet),
        "--chunk-alpha", str(cfg["cc_alpha"]),
        "--routing-topk", "3",
        "--routing-alpha", str(cfg["routing_alpha"]),
        "--workers", "14",
        "--output", str(output_path),
    ]

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8",
        timeout=3600,  # 1 hour max
    )
    elapsed = time.time() - t0

    # Log summary lines (skip httpx noise)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            if "results ->" in line or "P@1=" in line or "Failure:" in line:
                log(f"  [eval] {line}")

    if result.stderr:
        err_lines = [l for l in result.stderr.strip().split("\n")
                     if "ERROR" in l or "FAILED" in l]
        for line in err_lines[:5]:
            log(f"  [eval:err] {line}")

    gate(result.returncode == 0,
         f"{name} {arm} eval exited cleanly (took {elapsed:.0f}s)")
    gate(output_path.exists(), f"{name} {arm} eval output exists")

    count = sum(1 for _ in open(output_path, encoding="utf-8"))
    gate(count == EXPECTED_QUERIES,
         f"{name} {arm} eval record count: {count}/{EXPECTED_QUERIES}")

    # Verify context_chunks present
    first = json.loads(open(output_path, encoding="utf-8").readline())
    gate("context_chunks" in first,
         f"{name} {arm} eval has context_chunks field")


def step_4_eval(cfg: dict) -> None:
    """Run both A/B arms sequentially."""
    name = cfg["name"]
    baseline_parquet = cfg["baseline_parquet"]
    section_sac_parquet = cfg["section_sac_parquet"]
    baseline_collection = cfg["baseline_collection"]
    section_collection = cfg["section_collection"]

    baseline_output = f"data/eval_{name}_baseline_ctx.jsonl"
    section_output = f"data/eval_{name}_section_ctx.jsonl"

    _run_eval(cfg, "baseline", baseline_collection,
              baseline_parquet, baseline_output)
    _run_eval(cfg, "section", section_collection,
              section_sac_parquet, section_output)


# ── Step 5: Judge both arms ──────────────────────────────────────────────


def _run_judge(script: str, corpus: str, input_path: str,
               expected_output: str, workers: int, name: str,
               label: str) -> None:
    """Run a judge script on one file. Gates on output."""
    output_path = ROOT / expected_output

    if output_path.exists():
        count = sum(1 for _ in open(output_path, encoding="utf-8"))
        if count == EXPECTED_QUERIES:
            log(f"[{name}] {label} already exists: {count} records")
            return

    log(f"[{name}] Step 5: Running {label}...")

    cmd = [
        sys.executable, str(ROOT / "scripts" / script),
        "--corpus", corpus,
        "--input", str(ROOT / input_path),
        "--workers", str(workers),
    ]

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8",
        timeout=3600,
    )
    elapsed = time.time() - t0

    # Log summary
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            if ("CORRECT" in line or "INCORRECT" in line or "PARTIAL" in line
                    or "faithfulness" in line or "Perfect" in line
                    or "Claims:" in line or "Saved:" in line):
                log(f"  [judge] {line.strip()}")

    gate(result.returncode == 0,
         f"{name} {label} exited cleanly (took {elapsed:.0f}s)")
    gate(output_path.exists(), f"{name} {label} output exists")

    count = sum(1 for _ in open(output_path, encoding="utf-8"))
    gate(count == EXPECTED_QUERIES,
         f"{name} {label} record count: {count}/{EXPECTED_QUERIES}")


def step_5_judge(cfg: dict) -> None:
    """Judge correctness + faithfulness on both arms."""
    name = cfg["name"]
    corpus = cfg["name"]

    baseline_eval = f"data/eval_{name}_baseline_ctx.jsonl"
    section_eval = f"data/eval_{name}_section_ctx.jsonl"

    baseline_stem = f"eval_{name}_baseline_ctx"
    section_stem = f"eval_{name}_section_ctx"

    # Correctness (both arms)
    _run_judge("judge_answers_v2.py", corpus, baseline_eval,
               f"data/judge_v2_{corpus}_{baseline_stem}.jsonl", 8,
               name, "correctness (baseline)")
    _run_judge("judge_answers_v2.py", corpus, section_eval,
               f"data/judge_v2_{corpus}_{section_stem}.jsonl", 8,
               name, "correctness (section)")

    # Faithfulness (both arms)
    _run_judge("judge_faithfulness.py", corpus, baseline_eval,
               f"data/faith_{corpus}_{baseline_stem}.jsonl", 14,
               name, "faithfulness (baseline)")
    _run_judge("judge_faithfulness.py", corpus, section_eval,
               f"data/faith_{corpus}_{section_stem}.jsonl", 14,
               name, "faithfulness (section)")


# ── Step 6: Cross-tab and report ─────────────────────────────────────────


def _load_jsonl(path: str) -> dict:
    fp = ROOT / path
    with open(fp, encoding="utf-8") as f:
        return {json.loads(l)["query_id"]: json.loads(l) for l in f}


def step_6_crosstab(cfg: dict) -> None:
    """Build cross-tabs and write results to log."""
    name = cfg["name"]
    corpus = cfg["name"]

    baseline_stem = f"eval_{name}_baseline_ctx"
    section_stem = f"eval_{name}_section_ctx"

    # Load all data
    baseline_eval = _load_jsonl(f"data/eval_{name}_baseline_ctx.jsonl")
    section_eval = _load_jsonl(f"data/eval_{name}_section_ctx.jsonl")
    baseline_corr = _load_jsonl(f"data/judge_v2_{corpus}_{baseline_stem}.jsonl")
    section_corr = _load_jsonl(f"data/judge_v2_{corpus}_{section_stem}.jsonl")
    baseline_faith = _load_jsonl(f"data/faith_{corpus}_{baseline_stem}.jsonl")
    section_faith = _load_jsonl(f"data/faith_{corpus}_{section_stem}.jsonl")

    log_md(f"\n---\n")
    log_md(f"## {name.upper()} — Section A/B Results")
    log_md(f"*{cfg['note']}*\n")

    for arm_label, eval_data, corr_data, faith_data in [
        ("Baseline", baseline_eval, baseline_corr, baseline_faith),
        ("Section", section_eval, section_corr, section_faith),
    ]:
        # Layer 1: Taxonomy
        ft = Counter(r["failure_type"] for r in eval_data.values())
        p1 = sum(r["p_at_1"] for r in eval_data.values()) / len(eval_data)
        r8 = sum(r["r_at_8"] for r in eval_data.values()) / len(eval_data)

        log_md(f"\n### {arm_label} arm\n")
        log_md(f"**Layer 1 — Retrieval taxonomy:**")
        log_md(f"P@1={p1:.4f}  R@8={r8:.4f}")
        log_md(f"| Type | Count |")
        log_md(f"|------|-------|")
        for t in ["OK", "OVR", "CBF", "ICR", "SGP", "DRM"]:
            if ft.get(t, 0) > 0:
                log_md(f"| {t} | {ft[t]} |")

        # Layer 2: Correctness
        vc = Counter(r["verdict"] for r in corr_data.values())
        log_md(f"\n**Layer 2 — Correctness (span-informed judge):**")
        for v in ["CORRECT", "PARTIAL", "INCORRECT"]:
            pct = vc.get(v, 0) / len(corr_data) * 100
            log_md(f"  {v}: {vc.get(v, 0)} ({pct:.1f}%)")

        # Layer 2: Faithfulness
        scores = [r["faithfulness_score"] for r in faith_data.values()]
        mean_f = sum(scores) / len(scores)
        perfect = sum(1 for s in scores if s == 1.0)
        total_claims = sum(r["n_claims"] for r in faith_data.values())
        total_entailed = sum(r["n_entailed"] for r in faith_data.values())
        log_md(f"\n**Layer 2 — Faithfulness (pinned judge):**")
        log_md(f"  Mean: {mean_f:.3f}")
        log_md(f"  Perfect (==1.0): {perfect}/{len(scores)} ({perfect/len(scores)*100:.1f}%)")
        log_md(f"  Claims entailed: {total_entailed}/{total_claims} "
               f"({total_entailed/total_claims*100:.1f}%)")

        # Cross-tab
        cells = defaultdict(list)
        for qid in sorted(corr_data.keys()):
            verdict = corr_data[qid]["verdict"].lower()
            faith_score = faith_data[qid]["faithfulness_score"]
            col = "faithful" if faith_score == 1.0 else "unfaithful"
            cells[(verdict, col)].append(qid)

        log_md(f"\n**Correctness x Faithfulness cross-tab** (faithful = score == 1.0):")
        log_md(f"| | faithful (==1.0) | unfaithful (<1.0) | total |")
        log_md(f"|---|---|---|---|")
        for row in ["correct", "partial", "incorrect"]:
            fc = len(cells[(row, "faithful")])
            uc = len(cells[(row, "unfaithful")])
            log_md(f"| {row} | {fc} | {uc} | {fc + uc} |")
        f_tot = sum(len(cells[(r, "faithful")]) for r in ["correct", "partial", "incorrect"])
        u_tot = sum(len(cells[(r, "unfaithful")]) for r in ["correct", "partial", "incorrect"])
        log_md(f"| **total** | {f_tot} | {u_tot} | {f_tot + u_tot} |")

        # Diagnostic cells
        cu = cells[("correct", "unfaithful")]
        iu = cells[("incorrect", "unfaithful")]
        log_md(f"\nCORRECT+UNFAITHFUL (parametric leak): **{len(cu)}**")
        if cu:
            log_md(f"  Examples: {cu[:3]}")
            for qid in cu[:3]:
                fs = faith_data[qid]["faithfulness_score"]
                nc = faith_data[qid]["n_claims"]
                ne = faith_data[qid]["n_entailed"]
                log_md(f"  - {qid}: faith={fs:.3f} ({ne}/{nc})")

        log_md(f"\nINCORRECT+UNFAITHFUL (synthesis failure): **{len(iu)}**")
        if iu:
            log_md(f"  Examples: {iu[:3]}")
            for qid in iu[:3]:
                fs = faith_data[qid]["faithfulness_score"]
                nc = faith_data[qid]["n_claims"]
                ne = faith_data[qid]["n_entailed"]
                log_md(f"  - {qid}: faith={fs:.3f} ({ne}/{nc})")

        # Mechanism check (section arm only): CORRECT+UNFAITHFUL x failure_type
        if arm_label == "Section" and cu:
            ft_cu = Counter(eval_data[qid]["failure_type"] for qid in cu)
            sgp_count = ft_cu.get("SGP", 0)
            log_md(f"\n**Mechanism check — CORRECT+UNFAITHFUL failure types:**")
            for ftype, fcount in ft_cu.most_common():
                log_md(f"  {ftype}: {fcount}")
            log_md(f"  SGP/multi-span fraction: {sgp_count}/{len(cu)} "
                   f"({sgp_count/len(cu)*100:.0f}%)")

    log_md("")


# ── Dry run ───────────────────────────────────────────────────────────────


def dry_run() -> bool:
    """Preflight checks. Returns True if all pass."""
    log("=" * 60)
    log("DRY RUN — Preflight checks")
    log("=" * 60)

    ok = True

    # Check raw corpora
    for cfg in CORPORA:
        corpus_dir = ROOT / "data" / "corpus" / cfg["corpus_dir"]
        txt_files = list(corpus_dir.glob("*.txt"))
        status = f"{len(txt_files)} files" if txt_files else "MISSING"
        log(f"  Raw corpus {cfg['name']}: {corpus_dir} -> {status}")
        if not txt_files:
            ok = False

    # Check baseline collections
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from qdrant_client import QdrantClient

        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        for cfg in CORPORA:
            try:
                info = client.get_collection(cfg["baseline_collection"])
                log(f"  Baseline collection {cfg['baseline_collection']}: "
                    f"{info.points_count} points")
            except Exception:
                log(f"  Baseline collection {cfg['baseline_collection']}: MISSING",
                    level="ERROR")
                ok = False
    except Exception as e:
        log(f"  Qdrant connection failed: {e}", level="ERROR")
        ok = False

    # Check SAC summaries
    for cfg in CORPORA:
        sp = ROOT / cfg["summaries"]
        if sp.exists():
            summaries = json.loads(sp.read_text(encoding="utf-8"))
            log(f"  SAC summaries {cfg['name']}: {len(summaries)} docs at {sp}")
        else:
            log(f"  SAC summaries {cfg['name']}: MISSING at {sp}", level="ERROR")
            ok = False

    # Check routing indexes
    for cfg in CORPORA:
        ri = ROOT / cfg["routing_index"]
        log(f"  Routing index {cfg['name']}: {'OK' if ri.exists() else 'MISSING'} at {ri}")
        if not ri.exists():
            ok = False

    # Check benchmarks
    for cfg in CORPORA:
        bp = ROOT / cfg["benchmark"]
        if bp.exists():
            data = json.loads(bp.read_text(encoding="utf-8"))
            log(f"  Benchmark {cfg['name']}: {len(data['tests'])} queries")
        else:
            log(f"  Benchmark {cfg['name']}: MISSING at {bp}", level="ERROR")
            ok = False

    # Check importability
    imports_ok = True
    for mod_name in [
        "core.ingestion.section_chunker",
        "core.ingestion.embed_index",
        "core.measurement.taxonomy",
        "core.measurement.metrics",
    ]:
        try:
            __import__(mod_name)
            log(f"  Import {mod_name}: OK")
        except Exception as e:
            log(f"  Import {mod_name}: FAILED ({e})", level="ERROR")
            imports_ok = False
            ok = False

    # Check judge scripts exist and are importable (as scripts)
    for script in ["judge_answers_v2.py", "judge_faithfulness.py"]:
        sp = ROOT / "scripts" / script
        if sp.exists():
            log(f"  Script {script}: OK")
        else:
            log(f"  Script {script}: MISSING", level="ERROR")
            ok = False

    # Check API keys
    from dotenv import load_dotenv
    load_dotenv()
    for key in ["VOYAGE_API_KEY", "DEEPSEEK_API_KEY"]:
        val = os.environ.get(key)
        log(f"  {key}: {'SET' if val else 'MISSING'}")
        if not val:
            ok = False

    log("=" * 60)
    log(f"Dry run result: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    log("=" * 60)
    return ok


# ── Run one corpus ────────────────────────────────────────────────────────


def run_corpus(cfg: dict) -> None:
    """Run all steps for one corpus."""
    name = cfg["name"]
    log_md(f"\n{'=' * 60}")
    log(f"Starting corpus: {name.upper()}")
    log(f"  Note: {cfg['note']}")
    log(f"  CC alpha: {cfg['cc_alpha']}, routing alpha: {cfg['routing_alpha']}")
    log(f"  Baseline: {cfg['baseline_collection']}")
    log(f"  Section: {cfg['section_collection']}")

    # Step 1: section parquet
    chunk_count = step_1_section_parquet(cfg)

    # Step 2: SAC parquet
    sac_count = step_2_sac_parquet(cfg)
    gate(sac_count == chunk_count,
         f"{name} SAC count ({sac_count}) == section count ({chunk_count})")

    # Step 3: index
    step_3_index(cfg, sac_count)

    # Step 4: eval both arms
    step_4_eval(cfg)

    # Step 5: judge both arms
    step_5_judge(cfg)

    # Step 6: cross-tab
    step_6_crosstab(cfg)

    log(f"{name.upper()} COMPLETE")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sequential section-chunking A/B campaign")
    p.add_argument("--dry-run", action="store_true",
                   help="Run preflight checks only, don't execute")
    args = p.parse_args()

    # Initialize log (truncate on full run, append on dry-run)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("")  # Truncate
    log_md(f"# Section-Chunking A/B Campaign Log")
    log_md(f"Started: {_ts()}\n")
    print(f"\nLog: {LOG_PATH}\n", flush=True)

    # Dry run (always, even if not --dry-run)
    checks_ok = dry_run()
    if args.dry_run:
        sys.exit(0 if checks_ok else 1)
    gate(checks_ok, "All preflight checks passed")

    # Run campaign
    log_md(f"\n# Campaign Execution\n")
    t0 = time.time()

    for cfg in CORPORA:
        run_corpus(cfg)

    elapsed = time.time() - t0
    log_md(f"\n{'=' * 60}")
    log(f"CAMPAIGN COMPLETE — all 3 corpora done in {elapsed:.0f}s "
        f"({elapsed/60:.1f} min)")
    log_md(f"\n**DONE** {_ts()}")


if __name__ == "__main__":
    main()
