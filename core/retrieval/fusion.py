"""RRF (Reciprocal Rank Fusion) of two RetrievalResult objects.

~30 LOC. Hand-rolled because AutoRAG's HybridRetrieval assumes ChromaDB.
"""

from __future__ import annotations

from core.retrieval.base import RetrievalResult


def rrf(
    sparse: RetrievalResult,
    dense: RetrievalResult,
    k: int = 60,
    top_n: int = 8,
) -> RetrievalResult:
    """Fuse two retrieval results with Reciprocal Rank Fusion.

    ``score(d) = sum(1 / (k + rank_i(d)))`` across all input lists
    where the document appears.

    Parameters
    ----------
    sparse, dense:
        Retrieval results to fuse.
    k : int
        RRF smoothing constant (default 60).
    top_n : int
        Number of results to return after fusion.

    Returns
    -------
    RetrievalResult
        Fused result with spans from whichever source provided each chunk.
    """
    # Map chunk_id -> metadata from both sources
    chunk_meta: dict[str, dict] = {}

    for rank, (cid, content, span) in enumerate(
        zip(sparse.ids, sparse.contents, sparse.spans)
    ):
        chunk_meta[cid] = {"content": content, "span": span}

    for rank, (cid, content, span) in enumerate(
        zip(dense.ids, dense.contents, dense.spans)
    ):
        if cid not in chunk_meta:
            chunk_meta[cid] = {"content": content, "span": span}

    # Compute RRF scores
    rrf_scores: dict[str, float] = {}
    for rank, cid in enumerate(sparse.ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(dense.ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    # Sort by RRF score descending, take top_n
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_n]

    return RetrievalResult(
        contents=[chunk_meta[cid]["content"] for cid in sorted_ids],
        ids=sorted_ids,
        scores=[rrf_scores[cid] for cid in sorted_ids],
        spans=[chunk_meta[cid]["span"] for cid in sorted_ids],
    )
