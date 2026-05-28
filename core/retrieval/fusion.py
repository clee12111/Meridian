"""Fusion methods for combining sparse and dense retrieval results.

Provides RRF (Reciprocal Rank Fusion) and CC (Convex Combination).
Hand-rolled because AutoRAG's HybridRetrieval assumes ChromaDB.
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


def _minmax_normalize(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def cc_fusion(
    sparse: RetrievalResult,
    dense: RetrievalResult,
    alpha: float = 0.5,
    top_n: int = 50,
) -> RetrievalResult:
    """Convex combination fusion of BM25 (sparse) and dense scores.

    ``CC_score = alpha * norm(BM25) + (1 - alpha) * norm(dense)``

    Parameters
    ----------
    sparse, dense:
        Retrieval results to fuse.
    alpha : float
        Weight for sparse channel. 1.0 = pure BM25, 0.0 = pure dense.
    top_n : int
        Number of results to return after fusion.

    Returns
    -------
    RetrievalResult
        Fused result with CC scores.
    """
    # Normalize scores per channel
    sparse_norm = _minmax_normalize(list(sparse.scores))
    dense_norm = _minmax_normalize(list(dense.scores))

    # Build lookup: chunk_id -> metadata + normalized scores
    chunk_meta: dict[str, dict] = {}

    for cid, content, span, norm_score in zip(
        sparse.ids, sparse.contents, sparse.spans, sparse_norm
    ):
        chunk_meta[cid] = {
            "content": content,
            "span": span,
            "sparse_norm": norm_score,
            "dense_norm": 0.0,
        }

    for cid, content, span, norm_score in zip(
        dense.ids, dense.contents, dense.spans, dense_norm
    ):
        if cid in chunk_meta:
            chunk_meta[cid]["dense_norm"] = norm_score
        else:
            chunk_meta[cid] = {
                "content": content,
                "span": span,
                "sparse_norm": 0.0,
                "dense_norm": norm_score,
            }

    # Compute CC scores
    cc_scores: dict[str, float] = {}
    for cid, meta in chunk_meta.items():
        cc_scores[cid] = alpha * meta["sparse_norm"] + (1 - alpha) * meta["dense_norm"]

    # Sort descending, take top_n
    sorted_ids = sorted(cc_scores, key=cc_scores.get, reverse=True)[:top_n]

    return RetrievalResult(
        contents=[chunk_meta[cid]["content"] for cid in sorted_ids],
        ids=sorted_ids,
        scores=[cc_scores[cid] for cid in sorted_ids],
        spans=[chunk_meta[cid]["span"] for cid in sorted_ids],
    )


def weighted_rrf(
    sparse: RetrievalResult,
    dense: RetrievalResult,
    sparse_weight: float = 0.25,
    k: int = 60,
    top_n: int = 50,
) -> RetrievalResult:
    """Weighted RRF: rank-based fusion with asymmetric channel weights.

    ``score = sparse_weight / (k + sparse_rank)
            + (1 - sparse_weight) / (k + dense_rank)``

    Compare against cc_fusion() at the same nominal weights to test
    whether score preservation (CC) beats rank compression (wRRF).

    Parameters
    ----------
    sparse, dense:
        Retrieval results to fuse.
    sparse_weight : float
        Weight for sparse channel. 0.25 = 25% BM25, 75% dense.
    k : int
        RRF smoothing constant (default 60).
    top_n : int
        Number of results to return after fusion.
    """
    chunk_meta: dict[str, dict] = {}

    for cid, content, span in zip(sparse.ids, sparse.contents, sparse.spans):
        chunk_meta[cid] = {"content": content, "span": span}

    for cid, content, span in zip(dense.ids, dense.contents, dense.spans):
        if cid not in chunk_meta:
            chunk_meta[cid] = {"content": content, "span": span}

    # Compute weighted rank scores per channel
    scores: dict[str, float] = {}
    dense_weight = 1.0 - sparse_weight

    for rank, cid in enumerate(sparse.ids):
        scores[cid] = scores.get(cid, 0.0) + sparse_weight / (k + rank + 1)
    for rank, cid in enumerate(dense.ids):
        scores[cid] = scores.get(cid, 0.0) + dense_weight / (k + rank + 1)

    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]

    return RetrievalResult(
        contents=[chunk_meta[cid]["content"] for cid in sorted_ids],
        ids=sorted_ids,
        scores=[scores[cid] for cid in sorted_ids],
        spans=[chunk_meta[cid]["span"] for cid in sorted_ids],
    )
