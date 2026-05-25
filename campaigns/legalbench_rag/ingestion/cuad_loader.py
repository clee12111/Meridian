"""CUAD ingestion from LegalBench-RAG pre-generated data.

Loads title-prefixed queries and corpus text from two HF datasets:
  - orgrctera/legalbenchrag_cuad (4,042 queries, parquet)
  - thethomasmore/legalbench-rag-bilingual_v1_public (462 corpus .txt files)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaigns.legalbench_rag.ingestion.common import (
    build_corpus_df,
    download_hf_corpus,
    download_hf_queries,
    sample_mini,
    verify_span_integrity,
    write_benchmark_json,
)

HF_QUERIES = "orgrctera/legalbenchrag_cuad"
HF_CORPUS = "thethomasmore/legalbench-rag-bilingual_v1_public"
HF_CORPUS_PREFIX = "corpus/en/cuad"
DATASET_NAME = "cuad"


def ingest_cuad(
    data_dir: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    max_queries: int | None = 194,
) -> tuple[pd.DataFrame, list[dict]]:
    """Full ingestion pipeline for CUAD."""
    data_dir = Path(data_dir)
    corpus_dir = data_dir / "corpus" / DATASET_NAME

    print("  Downloading from HuggingFace...")
    queries = download_hf_queries(HF_QUERIES, DATASET_NAME)

    query_file_paths = {
        s["file_path"] for q in queries for s in q["snippets"]
    }
    raw_docs, lookup = download_hf_corpus(
        HF_CORPUS, HF_CORPUS_PREFIX, corpus_dir, DATASET_NAME, query_file_paths
    )

    verify_span_integrity(queries, lookup)
    print(f"  Span integrity verified for all {len(queries)} queries")

    if max_queries is not None:
        queries = sample_mini(queries, max_queries)

    write_benchmark_json(queries, data_dir / "benchmarks", DATASET_NAME)

    corpus_df = build_corpus_df(raw_docs, chunk_size, chunk_overlap, DATASET_NAME)
    corpus_df.to_parquet(data_dir / "corpus_cuad.parquet", index=False)
    print(
        f"Ingested {len(raw_docs)} documents, "
        f"{len(corpus_df)} chunks, {len(queries)} queries"
    )

    return corpus_df, queries
