"""PipelineContext — runtime dependencies built once at startup.

State carries what happened. Context carries what is needed to make things
happen. Nothing in context is serializable or checkpointed — it lives
in memory for the duration of a pipeline run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.qdrant_retriever import QdrantRetriever

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Runtime dependencies injected into every phase node."""

    qdrant_retriever: QdrantRetriever
    bm25_retriever: BM25Retriever
    corpus_df: pd.DataFrame
    llm_client: Any           # placeholder, typed later
    langfuse_client: Any      # Langfuse instance or None
    dataset_name: str         # e.g. "contractnli" — passed to retrievers

    @classmethod
    def build(
        cls,
        corpus_path: Path,
        qdrant_url: str,
        collection_name: str,
        dataset_name: str,
        top_k: int = 50,
        qdrant_api_key: str | None = None,
    ) -> "PipelineContext":
        """Assemble a PipelineContext from disk and network resources.

        Parameters
        ----------
        corpus_path : Path
            Path to a parquet file with columns: chunk_id, content,
            start_end_idx, doc_id (and optionally dataset_name).
        qdrant_url : str
            Qdrant server URL (e.g. "http://localhost:6333").
        collection_name : str
            Name of the Qdrant collection to search.
        dataset_name : str
            Dataset filter passed to retrievers on every query.
        top_k : int
            Number of results per retriever channel.
        qdrant_api_key : str or None
            Qdrant Cloud API key. Leave None for local Docker.
        """
        from qdrant_client import QdrantClient

        corpus_df = pd.read_parquet(corpus_path)

        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        qdrant_retriever = QdrantRetriever(
            client=qdrant_client,
            collection_name=collection_name,
            corpus_df=corpus_df,
            top_k=top_k,
        )

        bm25_retriever = BM25Retriever(corpus_df, top_k=top_k)

        # Langfuse tracing — optional, fail-silent
        import os

        langfuse_client = None
        if os.environ.get("LANGFUSE_PUBLIC_KEY"):
            try:
                from langfuse import Langfuse

                langfuse_client = Langfuse(
                    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                    host=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
                )
                logger.info("Langfuse tracing enabled")
            except Exception as e:
                logger.warning("Langfuse init failed: %s — tracing disabled", e)

        return cls(
            qdrant_retriever=qdrant_retriever,
            bm25_retriever=bm25_retriever,
            corpus_df=corpus_df,
            llm_client=None,
            langfuse_client=langfuse_client,
            dataset_name=dataset_name,
        )
