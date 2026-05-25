"""Unified loader for all four LegalBench-RAG datasets.

Calls individual loaders and merges into a single corpus DataFrame
with a ``dataset_name`` column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaigns.legalbench_rag.ingestion.contractnli_loader import ingest_contractnli
from campaigns.legalbench_rag.ingestion.cuad_loader import ingest_cuad
from campaigns.legalbench_rag.ingestion.maud_loader import ingest_maud
from campaigns.legalbench_rag.ingestion.privacyqa_loader import ingest_privacyqa


def ingest_all(
    data_dir: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    max_queries_per_dataset: int | None = 194,
) -> tuple[pd.DataFrame, list[dict]]:
    """Ingest all four LegalBench-RAG datasets.

    Returns a single merged corpus DataFrame (with ``dataset_name`` column)
    and a combined query list across all datasets.
    """
    data_dir = Path(data_dir)
    all_dfs: list[pd.DataFrame] = []
    all_queries: list[dict] = []

    print("\n--- ContractNLI ---")
    df, queries = ingest_contractnli(
        data_dir, chunk_size, chunk_overlap, max_queries_per_dataset
    )
    all_dfs.append(df)
    all_queries.extend(queries)

    print("\n--- CUAD ---")
    df, queries = ingest_cuad(
        data_dir, chunk_size, chunk_overlap, max_queries_per_dataset
    )
    all_dfs.append(df)
    all_queries.extend(queries)

    print("\n--- PrivacyQA ---")
    df, queries = ingest_privacyqa(
        data_dir, chunk_size, chunk_overlap, max_queries_per_dataset
    )
    all_dfs.append(df)
    all_queries.extend(queries)

    print("\n--- MAUD ---")
    df, queries = ingest_maud(
        data_dir, chunk_size, chunk_overlap, max_queries_per_dataset
    )
    all_dfs.append(df)
    all_queries.extend(queries)

    # Merge into unified corpus
    corpus_df = pd.concat(all_dfs, ignore_index=True)
    corpus_df.to_parquet(data_dir / "corpus.parquet", index=False)

    n_docs = corpus_df["doc_id"].nunique()
    n_datasets = corpus_df["dataset_name"].nunique()
    print(f"\n{'=' * 50}")
    print(
        f"TOTAL: {n_datasets} datasets, {n_docs} documents, "
        f"{len(corpus_df)} chunks, {len(all_queries)} queries"
    )
    print("=" * 50)

    return corpus_df, all_queries
