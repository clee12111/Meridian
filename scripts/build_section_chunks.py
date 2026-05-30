"""Build section-aware corpus parquet for A/B testing against SAC baseline.

Reads raw .txt documents, chunks with section-aware boundaries, emits a
parquet with the same schema as the existing corpus parquets. Does NOT
embed, index, or touch Qdrant.

Usage:
    python scripts/build_section_chunks.py cuad
    python scripts/build_section_chunks.py contractnli
    python scripts/build_section_chunks.py maud
    python scripts/build_section_chunks.py privacyqa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core.ingestion.section_chunker import (
    CHUNK_CAP,
    CHUNK_FLOOR,
    SectionChunk,
    chunk_document,
    compute_checksum,
)

CORPUS_DIR = Path("data/corpus")
OUTPUT_DIR = Path("data")

CORPUS_MAP = {
    "contractnli": {"dir": "contractnli", "dataset_name": "contractnli"},
    "cuad":        {"dir": "cuad",        "dataset_name": "cuad"},
    "maud":        {"dir": "maud",        "dataset_name": "maud"},
    "privacyqa":   {"dir": "privacy_qa",  "dataset_name": "privacy_qa"},
}


def build_corpus(corpus_name: str) -> None:
    config = CORPUS_MAP[corpus_name]
    corpus_dir = CORPUS_DIR / config["dir"]
    dataset_name = config["dataset_name"]
    output_path = OUTPUT_DIR / f"corpus_{dataset_name}_section.parquet"

    txt_files = sorted(corpus_dir.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found in {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Chunking {len(txt_files)} documents from {corpus_dir}")
    print(f"  Cap: {CHUNK_CAP}, Floor: {CHUNK_FLOOR}")

    all_chunks: list[SectionChunk] = []
    source_texts: dict[str, str] = {}
    fallback_count = 0

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        doc_id = f"{config['dir']}/{txt_file.name}"
        source_texts[doc_id] = text

        chunks = chunk_document(text, doc_id)
        if chunks and chunks[0].chunk_index == 0 and len(chunks) > 0:
            # Check if this was a fallback (no section boundaries found)
            from core.ingestion.section_chunker import _find_boundaries
            if len(_find_boundaries(text)) < 2:
                fallback_count += 1
        all_chunks.extend(chunks)

    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Docs with section boundaries: {len(txt_files) - fallback_count}")
    print(f"  Docs using fixed-stride fallback: {fallback_count}")

    # ── Offset integrity gate ───────────────────────────────────────────
    print("\n=== ACCEPTANCE CHECK 1: Offset round-trip ===")
    failures = 0
    for chunk in all_chunks:
        source = source_texts[chunk.doc_id]
        expected = source[chunk.start:chunk.end]
        if expected != chunk.content:
            failures += 1
            if failures <= 5:
                print(f"  FAIL: {chunk.doc_id}#chunk-{chunk.chunk_index:05d} "
                      f"offset [{chunk.start},{chunk.end}) content mismatch")
    print(f"  Passed: {len(all_chunks) - failures}/{len(all_chunks)}")
    if failures:
        print(f"  GATE FAILED: {failures} offset mismatches — aborting")
        sys.exit(1)
    print("  GATE PASSED")

    # ── Coverage check ──────────────────────────────────────────────────
    print("\n=== ACCEPTANCE CHECK 2: Coverage ===")
    total_gaps = 0
    docs_with_gaps = 0
    for doc_id, source in source_texts.items():
        doc_chunks = [c for c in all_chunks if c.doc_id == doc_id]
        covered = set()
        for c in doc_chunks:
            covered.update(range(c.start, c.end))
        # Only check non-whitespace characters
        missing = []
        for i in range(len(source)):
            if i not in covered and not source[i].isspace():
                missing.append(i)
        if missing:
            total_gaps += len(missing)
            docs_with_gaps += 1
            if docs_with_gaps <= 3:
                print(f"  Gap in {doc_id}: {len(missing)} uncovered non-ws chars "
                      f"(first at offset {missing[0]})")
    if total_gaps == 0:
        print(f"  PASS: All non-whitespace characters covered in all {len(source_texts)} docs")
    else:
        print(f"  WARN: {total_gaps} uncovered non-ws chars across {docs_with_gaps} docs")

    # ── Overlap behavior ────────────────────────────────────────────────
    print("\n=== ACCEPTANCE CHECK 3: Overlap behavior ===")
    abut_count = 0
    overlap_count = 0
    gap_count = 0
    overlap_sizes: list[int] = []
    gap_sizes: list[int] = []
    for doc_id in source_texts:
        doc_chunks = sorted(
            [c for c in all_chunks if c.doc_id == doc_id],
            key=lambda c: c.start,
        )
        for i in range(len(doc_chunks) - 1):
            curr_end = doc_chunks[i].end
            next_start = doc_chunks[i + 1].start
            if curr_end == next_start:
                abut_count += 1
            elif curr_end > next_start:
                overlap_count += 1
                overlap_sizes.append(curr_end - next_start)
            else:
                gap_count += 1
                gap_sizes.append(next_start - curr_end)
    total_boundaries = abut_count + overlap_count + gap_count
    print(f"  Total inter-chunk boundaries: {total_boundaries}")
    print(f"  Abutting (end == next start):  {abut_count}")
    print(f"  Overlapping (end > next start): {overlap_count}", end="")
    if overlap_sizes:
        print(f"  (sizes: min={min(overlap_sizes)}, max={max(overlap_sizes)}, "
              f"median={sorted(overlap_sizes)[len(overlap_sizes)//2]})")
    else:
        print()
    print(f"  Gaps (end < next start):       {gap_count}", end="")
    if gap_sizes:
        print(f"  (sizes: min={min(gap_sizes)}, max={max(gap_sizes)})")
    else:
        print()

    # ── Build DataFrame ─────────────────────────────────────────────────
    rows = []
    for chunk in all_chunks:
        rows.append({
            "doc_id": chunk.doc_id,
            "content": chunk.content,
            "start_end_idx": np.array([chunk.start, chunk.end], dtype=np.int64),
            "chunk_id": f"{chunk.doc_id}#chunk-{chunk.chunk_index:05d}",
            "checksum": compute_checksum(chunk.content),
            "dataset_name": dataset_name,
        })
    df = pd.DataFrame(rows)

    # ── Schema/dtype match ──────────────────────────────────────────────
    print("\n=== ACCEPTANCE CHECK 4: Schema/dtype match ===")
    existing_path = OUTPUT_DIR / f"corpus_{dataset_name}.parquet"
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        for col in existing.columns:
            if col not in df.columns:
                print(f"  MISSING column: {col}")
            else:
                e_dtype = str(existing[col].dtype)
                n_dtype = str(df[col].dtype)
                match = "OK" if e_dtype == n_dtype else f"MISMATCH ({e_dtype} vs {n_dtype})"
                print(f"  {col}: {match}")
                # Check start_end_idx element dtype
                if col == "start_end_idx":
                    e_inner = existing[col].iloc[0].dtype
                    n_inner = df[col].iloc[0].dtype
                    inner_match = "OK" if e_inner == n_inner else f"MISMATCH ({e_inner} vs {n_inner})"
                    print(f"    inner array dtype: {inner_match}")
        for col in df.columns:
            if col not in existing.columns:
                print(f"  EXTRA column: {col}")
    else:
        print(f"  No existing parquet at {existing_path} to compare")

    # ── Distribution ────────────────────────────────────────────────────
    print("\n=== ACCEPTANCE CHECK 5: Distribution ===")
    new_lens = df["start_end_idx"].apply(lambda x: x[1] - x[0])
    print(f"  {'Metric':<12} {'New (section)':>15} ", end="")
    if existing_path.exists():
        old_lens = existing["start_end_idx"].apply(lambda x: x[1] - x[0])
        print(f"{'Existing (fixed)':>15}")
        print(f"  {'count':<12} {len(df):>15,} {len(existing):>15,}")
        print(f"  {'min':<12} {new_lens.min():>15} {old_lens.min():>15}")
        print(f"  {'median':<12} {new_lens.median():>15.0f} {old_lens.median():>15.0f}")
        print(f"  {'p95':<12} {new_lens.quantile(0.95):>15.0f} {old_lens.quantile(0.95):>15.0f}")
        print(f"  {'max':<12} {new_lens.max():>15} {old_lens.max():>15}")
        print(f"  {'mean':<12} {new_lens.mean():>15.1f} {old_lens.mean():>15.1f}")
    else:
        print()
        print(f"  {'count':<12} {len(df):>15,}")
        print(f"  {'min':<12} {new_lens.min():>15}")
        print(f"  {'median':<12} {new_lens.median():>15.0f}")
        print(f"  {'p95':<12} {new_lens.quantile(0.95):>15.0f}")
        print(f"  {'max':<12} {new_lens.max():>15}")

    # ── Sample chunks ───────────────────────────────────────────────────
    print("\n=== SAMPLE: 5 consecutive chunks from first doc ===")
    first_doc = df["doc_id"].iloc[0]
    doc_df = df[df["doc_id"] == first_doc].head(5)
    for _, row in doc_df.iterrows():
        se = row["start_end_idx"]
        content_preview = row["content"][:120].replace("\n", "\\n")
        content_preview = content_preview.encode("ascii", errors="replace").decode()
        print(f"  {row['chunk_id']}")
        print(f"    [{se[0]:>6}, {se[1]:>6})  len={se[1]-se[0]:>4}  "
              f"\"{content_preview}...\"")

    # ── Write parquet ───────────────────────────────────────────────────
    df.to_parquet(output_path, index=False)
    print(f"\nWrote {len(df)} chunks to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build section-aware corpus parquet")
    p.add_argument("corpus", choices=list(CORPUS_MAP.keys()),
                   help="Which corpus to chunk")
    args = p.parse_args()
    build_corpus(args.corpus)


if __name__ == "__main__":
    main()
