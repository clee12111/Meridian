"""OpenAI text-embedding-3-large dense retriever.

For paper replication baseline only — not used in production
experiments. Production uses voyage-4-large via qdrant_retriever.py.
"""

from __future__ import annotations

import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    Filter, FieldCondition, MatchValue,
)

from core.retrieval.base import RetrievalResult

OPENAI_MODEL = "text-embedding-3-large"
OPENAI_DIMENSION = 3072


class OpenAIRetriever:
    """Dense retriever: OpenAI embeddings + Qdrant vector search."""

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
        self._openai = OpenAI()
        self._id_to_idx: dict[str, int] = {
            row["chunk_id"]: i
            for i, row in self._corpus_df.iterrows()
        }

    def index(self, batch_size: int = 128) -> None:
        """Create collection and upsert all corpus embeddings."""
        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=OPENAI_DIMENSION, distance=Distance.COSINE
            ),
        )

        texts = self._corpus_df["content"].tolist()
        chunk_ids = self._corpus_df["chunk_id"].tolist()
        has_dataset = "dataset_name" in self._corpus_df.columns
        dataset_names = (
            self._corpus_df["dataset_name"].tolist() if has_dataset else [None] * len(texts)
        )

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = chunk_ids[i : i + batch_size]
            batch_ds = dataset_names[i : i + batch_size]

            resp = self._openai.embeddings.create(
                model=OPENAI_MODEL, input=batch_texts
            )
            embeddings = [d.embedding for d in resp.data]

            points = []
            for j, (emb, cid, ds) in enumerate(zip(embeddings, batch_ids, batch_ds)):
                payload = {"chunk_id": cid}
                if ds is not None:
                    payload["dataset_name"] = ds
                points.append(PointStruct(id=i + j, vector=emb, payload=payload))
            self._client.upsert(
                collection_name=self._collection, points=points
            )

        print(f"Indexed {len(texts)} chunks into '{self._collection}'")

    def retrieve(
        self, query: str, top_k: int | None = None, dataset_name: str | None = None
    ) -> RetrievalResult:
        """Retrieve top-k chunks for *query*.

        Parameters
        ----------
        dataset_name : str, optional
            If provided, restrict search to chunks with this dataset_name
            in their Qdrant payload.
        """
        k = top_k if top_k is not None else self._top_k

        resp = self._openai.embeddings.create(
            model=OPENAI_MODEL, input=[query]
        )
        query_emb = resp.data[0].embedding

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
            contents=contents, ids=ids, scores=scores, spans=spans,
        )
