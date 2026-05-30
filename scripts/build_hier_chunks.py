"""Build hierarchical corpus parquet: parent sections + child sub-splits.

Children are the retrieval units (indexed to Qdrant). Parents are the
context units (fed to the model via Phase 7 parent swap). Each child
carries a parent_id pointer; leaves (no split needed) have parent_id=None.

Usage:
    python scripts/build_hier_chunks.py cuad
    python scripts/build_hier_chunks.py maud
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core.ingestion.section_chunker import (
    CHUNK_CAP,
    CHUNK_FLOOR,
    chunk_document_hierarchical,
    compute_checksum,
)

CORPUS_DIR = Path("data/corpus")
OUTPUT_DIR = Path("data")

CORPUS_MAP = {
    "cuad": {"dir": "cuad", "dataset_name": "cuad"},
    "maud": {"dir": "maud", "dataset_name": "maud"},
    "contractnli": {"dir": "contractnli", "dataset_name": "contractnli"},
    "privacyqa": {"dir": "privacy_qa", "dataset_name": "privacy_qa"},
}


def build_corpus(corpus_name: str) -> None:
    config = CORPUS_MAP[corpus_name]
    corpus_dir = CORPUS_DIR / config["dir"]
    dataset_name = config["dataset_name"]
    output_path = OUTPUT_DIR / f"corpus_{dataset_name}_hier.parquet"

    txt_files = sorted(corpus_dir.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found in {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Hierarchical chunking {len(txt_files)} documents from {corpus_dir}")
    print(f"  Cap: {CHUNK_CAP}, Floor: {CHUNK_FLOOR}")

    rows: list[dict] = []
    source_texts: dict[str, str] = {}

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        doc_id = f"{config['dir']}/{txt_file.name}"
        source_texts[doc_id] = text

        result = chunk_document_hierarchical(text, doc_id)

        # Emit parent rows (is_parent=True, not indexed)
        parent_chunk_ids: list[str] = []
        for pi, (ps, pe) in enumerate(result.parents):
            parent_cid = f"{doc_id}#parent-{pi:05d}"
            parent_chunk_ids.append(parent_cid)
            rows.append({
                "doc_id": doc_id,
                "content": text[ps:pe],
                "start_end_idx": np.array([ps, pe], dtype=np.int64),
                "chunk_id": parent_cid,
                "checksum": compute_checksum(text[ps:pe]),
                "dataset_name": dataset_name,
                "parent_id": None,
                "is_parent": True,
            })

        # Emit child rows (is_parent=False, indexed)
        for ci, (cs, ce, pi) in enumerate(result.children):
            child_cid = f"{doc_id}#chunk-{ci:05d}"
            parent_cid = parent_chunk_ids[pi]
            ps, pe = result.parents[pi]
            # If leaf (child == parent span), parent_id = None
            is_leaf = (cs == ps and ce == pe)
            rows.append({
                "doc_id": doc_id,
                "content": text[cs:ce],
                "start_end_idx": np.array([cs, ce], dtype=np.int64),
                "chunk_id": child_cid,
                "checksum": compute_checksum(text[cs:ce]),
                "dataset_name": dataset_name,
                "parent_id": None if is_leaf else parent_cid,
                "is_parent": False,
            })

    df = pd.DataFrame(rows)
    n_parents = df["is_parent"].sum()
    n_children = (~df["is_parent"]).sum()
    n_leaves = df[(~df["is_parent"]) & (df["parent_id"].isna())].shape[0]
    n_with_parent = df[(~df["is_parent"]) & (df["parent_id"].notna())].shape[0]

    print(f"\n  Total rows: {len(df)}")
    print(f"  Parents (not indexed): {n_parents}")
    print(f"  Children (indexed): {n_children}")
    print(f"    Leaves (parent==child): {n_leaves}")
    print(f"    With parent (sub-splits): {n_with_parent}")

    # ── Offset gate: BOTH levels ──────────────────────────────────────
    print("\n=== OFFSET GATE: Parents ===")
    parent_fails = 0
    for _, row in df[df["is_parent"]].iterrows():
        se = row["start_end_idx"]
        source = source_texts[row["doc_id"]]
        if source[se[0]:se[1]] != row["content"]:
            parent_fails += 1
    print(f"  Passed: {n_parents - parent_fails}/{n_parents}")
    if parent_fails:
        print(f"  GATE FAILED: {parent_fails} parent offset mismatches")
        sys.exit(1)
    print("  GATE PASSED")

    print("\n=== OFFSET GATE: Children ===")
    child_fails = 0
    for _, row in df[~df["is_parent"]].iterrows():
        se = row["start_end_idx"]
        source = source_texts[row["doc_id"]]
        if source[se[0]:se[1]] != row["content"]:
            child_fails += 1
    print(f"  Passed: {n_children - child_fails}/{n_children}")
    if child_fails:
        print(f"  GATE FAILED: {child_fails} child offset mismatches")
        sys.exit(1)
    print("  GATE PASSED")

    # ── Parent coverage gate ──────────────────────────────────────────
    print("\n=== PARENT COVERAGE GATE ===")
    coverage_fails = 0
    parent_lookup = {row["chunk_id"]: row for _, row in df[df["is_parent"]].iterrows()}
    for _, row in df[(~df["is_parent"]) & (df["parent_id"].notna())].iterrows():
        p_row = parent_lookup.get(row["parent_id"])
        if p_row is None:
            coverage_fails += 1
            continue
        cs, ce = row["start_end_idx"]
        ps, pe = p_row["start_end_idx"]
        if cs < ps or ce > pe:
            coverage_fails += 1
    print(f"  Children within parent bounds: {n_with_parent - coverage_fails}/{n_with_parent}")
    if coverage_fails:
        print(f"  GATE FAILED: {coverage_fails} children outside parent bounds")
        sys.exit(1)
    print("  GATE PASSED")

    # ── Child size distribution ───────────────────────────────────────
    print("\n=== CHILD SIZE DISTRIBUTION ===")
    child_lens = df[~df["is_parent"]]["start_end_idx"].apply(lambda x: x[1] - x[0])
    print(f"  Count: {len(child_lens):,}")
    print(f"  Min: {child_lens.min()}")
    print(f"  Median: {child_lens.median():.0f}")
    print(f"  P95: {child_lens.quantile(0.95):.0f}")
    print(f"  Max: {child_lens.max()}")
    print(f"  Under-floor (<{CHUNK_FLOOR}): {(child_lens < CHUNK_FLOOR).sum()} "
          f"({(child_lens < CHUNK_FLOOR).mean()*100:.1f}%)")

    print("\n=== PARENT SIZE DISTRIBUTION ===")
    parent_lens = df[df["is_parent"]]["start_end_idx"].apply(lambda x: x[1] - x[0])
    print(f"  Count: {len(parent_lens):,}")
    print(f"  Min: {parent_lens.min()}")
    print(f"  Median: {parent_lens.median():.0f}")
    print(f"  P95: {parent_lens.quantile(0.95):.0f}")
    print(f"  Max: {parent_lens.max()}")

    # ── Write parquet ─────────────────────────────────────────────────
    df.to_parquet(output_path, index=False)
    print(f"\nWrote {len(df)} rows to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build hierarchical corpus parquet")
    p.add_argument("corpus", choices=list(CORPUS_MAP.keys()))
    args = p.parse_args()
    build_corpus(args.corpus)


if __name__ == "__main__":
    main()
