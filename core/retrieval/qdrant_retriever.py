"""Hand-rolled Qdrant dense retriever with Voyage embeddings.

Does not use AutoRAG's VectorDBRetrieval (which wraps ChromaDB,
not Qdrant).
"""

from __future__ import annotations

import time

import pandas as pd
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    Filter, FieldCondition, MatchValue,
)

from core.retrieval.base import RetrievalResult

VOYAGE_MODEL = "voyage-4-large"
VOYAGE_DIMENSION = 1024


def _embed_with_retry(
    client: voyageai.Client,
    texts: list[str],
    model: str,
    input_type: str = "document",
    max_retries: int = 5,
) -> list[list[float]]:
    """Embed with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return client.embed(texts, model=model, input_type=input_type).embeddings
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt
                print(f"  Rate limit hit, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded on Voyage embed")


def _upsert_with_retry(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct],
    max_retries: int = 5,
) -> None:
    """Upsert with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return
        except Exception as e:
            err = str(e).lower()
            if "timeout" in err or "429" in err or "unexpected" in err:
                wait = 2 ** attempt
                print(f"  Qdrant upsert error, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded on Qdrant upsert")


class QdrantRetriever:
    """Dense retriever: Voyage embeddings + Qdrant vector search.

    Parameters
    ----------
    client : QdrantClient
        Connected Qdrant client (local or remote).
    collection_name : str
        Name of the Qdrant collection.
    corpus_df : pd.DataFrame
        Must have columns: ``chunk_id``, ``content``, ``start_end_idx``, ``doc_id``.
    top_k : int
        Number of results to return per query.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        corpus_df: pd.DataFrame,
        top_k: int = 8,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._corpus_df = corpus_df.reset_index(drop=True)
        self._top_k = top_k
        self._voyage = voyageai.Client()

        # Build lookup from chunk_id -> row index
        self._id_to_idx: dict[str, int] = {
            row["chunk_id"]: i
            for i, row in self._corpus_df.iterrows()
        }

    def collection_point_count(self) -> int:
        """Return the number of points in the collection, or 0 if it doesn't exist."""
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            return 0

    def index(self, batch_size: int = 128) -> None:
        """Create collection and upsert all corpus embeddings."""
        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=VOYAGE_DIMENSION, distance=Distance.COSINE
            ),
        )

        texts = self._corpus_df["content"].tolist()
        chunk_ids = self._corpus_df["chunk_id"].tolist()
        has_dataset = "dataset_name" in self._corpus_df.columns
        dataset_names = (
            self._corpus_df["dataset_name"].tolist() if has_dataset else [None] * len(texts)
        )

        total_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_num, i in enumerate(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            batch_ids = chunk_ids[i : i + batch_size]
            batch_ds = dataset_names[i : i + batch_size]

            embeddings = _embed_with_retry(
                self._voyage, batch_texts, VOYAGE_MODEL, input_type="document"
            )

            points = []
            for j, (emb, cid, ds) in enumerate(zip(embeddings, batch_ids, batch_ds)):
                payload = {"chunk_id": cid}
                if ds is not None:
                    payload["dataset_name"] = ds
                points.append(PointStruct(id=i + j, vector=emb, payload=payload))

            _upsert_with_retry(self._client, self._collection, points)

            if (batch_num + 1) % 10 == 0 or (batch_num + 1) == total_batches:
                print(
                    f"  Embedding batch {batch_num + 1}/{total_batches} "
                    f"({min(i + batch_size, len(texts))}/{len(texts)} chunks)",
                    flush=True,
                )

            time.sleep(0.25)

        print(f"Indexed {len(texts)} chunks into '{self._collection}'", flush=True)

    def retrieve(
        self, query: str, top_k: int | None = None, dataset_name: str | None = None
    ) -> RetrievalResult:
        """Retrieve top-k chunks for *query*.

        Embeds query with Voyage, searches Qdrant, joins spans from
        ``corpus_df`` — spans are never reconstructed from text.

        Parameters
        ----------
        dataset_name : str, optional
            If provided, restrict search to chunks with this dataset_name
            in their Qdrant payload.
        """
        k = top_k if top_k is not None else self._top_k

        query_emb = _embed_with_retry(
            self._voyage, [query], VOYAGE_MODEL, input_type="query"
        )[0]

        query_filter = None
        if dataset_name is not None:
            query_filter = Filter(
                must=[FieldCondition(key="dataset_name", match=MatchValue(value=dataset_name))]
            )

        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_emb,
            limit=k,
            query_filter=query_filter,
        ).points

        contents: list[str] = []
        ids: list[str] = []
        scores: list[float] = []
        spans: list[tuple[int, int]] = []

        for hit in hits:
            chunk_id = hit.payload["chunk_id"]
            row_idx = self._id_to_idx[chunk_id]
            row = self._corpus_df.iloc[row_idx]

            contents.append(row["content"])
            ids.append(chunk_id)
            scores.append(float(hit.score))
            spans.append(tuple(row["start_end_idx"]))

        return RetrievalResult(
            contents=contents,
            ids=ids,
            scores=scores,
            spans=spans,
        )
