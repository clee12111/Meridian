"""BM25 sparse retriever wrapping rank_bm25.BM25Okapi.

AutoRAG's BM25Retrieval is not importable on Python 3.13 (broken
llama_index dependency chain). This wraps rank_bm25 directly —
the same engine AutoRAG uses internally.

Instantiate ONCE per experiment run, not per query.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from core.retrieval.base import RetrievalResult


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer."""
    return re.split(r"\s+", text.strip().lower())


class BM25Retriever:
    """BM25 sparse retriever over a corpus DataFrame.

    Builds per-dataset BM25 indexes when ``corpus_df`` contains a
    ``dataset_name`` column, so each query can search only its own
    dataset's chunks.  Instantiate ONCE per experiment run.

    Parameters
    ----------
    corpus_df : pd.DataFrame
        Must have columns: ``chunk_id``, ``content``, ``start_end_idx``, ``doc_id``.
        If a ``dataset_name`` column is present, per-dataset indexes are built.
    top_k : int
        Number of results to return per query.
    """

    def __init__(self, corpus_df: pd.DataFrame, top_k: int = 8) -> None:
        self._corpus_df = corpus_df.reset_index(drop=True)
        self._top_k = top_k

        # Always build the full-corpus index (used when dataset_name is None)
        corpus_texts = self._corpus_df["content"].tolist()
        tokenized = [_tokenize(t) for t in corpus_texts]
        self._bm25 = BM25Okapi(tokenized)

        # Build per-dataset indexes if the column exists
        self._dataset_indexes: dict[str, tuple[BM25Okapi, pd.DataFrame]] = {}
        if "dataset_name" in self._corpus_df.columns:
            for ds_name, group_df in self._corpus_df.groupby("dataset_name"):
                ds_df = group_df.reset_index(drop=True)
                ds_tokenized = [_tokenize(t) for t in ds_df["content"].tolist()]
                self._dataset_indexes[ds_name] = (BM25Okapi(ds_tokenized), ds_df)

    def retrieve(
        self, query: str, top_k: int | None = None, dataset_name: str | None = None
    ) -> RetrievalResult:
        """Retrieve top-k chunks for *query*.

        Spans are populated from ``corpus_df.start_end_idx`` —
        never reconstructed from text.

        Parameters
        ----------
        dataset_name : str, optional
            If provided, search only that dataset's BM25 index.
        """
        k = top_k if top_k is not None else self._top_k
        tokenized_query = _tokenize(query)

        if dataset_name is not None:
            if dataset_name not in self._dataset_indexes:
                raise KeyError(
                    f"No BM25 index for dataset {dataset_name!r}. "
                    f"Available: {sorted(self._dataset_indexes)}"
                )
            bm25_index, df = self._dataset_indexes[dataset_name]
        else:
            bm25_index, df = self._bm25, self._corpus_df

        scores = bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]

        contents: list[str] = []
        ids: list[str] = []
        result_scores: list[float] = []
        spans: list[tuple[int, int]] = []

        for idx in top_indices:
            row = df.iloc[idx]
            contents.append(row["content"])
            ids.append(row["chunk_id"])
            result_scores.append(float(scores[idx]))
            spans.append(tuple(row["start_end_idx"]))

        return RetrievalResult(
            contents=contents,
            ids=ids,
            scores=result_scores,
            spans=spans,
        )
