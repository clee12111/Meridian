"""Document-level routing via hybrid dense + BM25 summary/filename matching.

Two channels per document:
  - Dense: voyage-4-large embeddings of SAC summaries (party names, subject)
  - BM25: token matching over filename + summary text (catches abbreviations
    like CEII, SE_NDCA that dense embeddings can't bridge to full party names)

Combined via CC fusion (same proven approach as chunk retrieval — score
magnitude preservation matters for rare-token abbreviation signals).

Occupies the architectural slot of the reranker (filter between retrieval
and context) but uses document identity — the signal cross-encoder
rerankers are blind to (Finding 21).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import voyageai
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

ROUTING_INDEX_PATH = Path("data/routing_index.npz")
SUMMARIES_PATH = Path("data/sac_summaries.json")
VOYAGE_MODEL = "voyage-4-large"


def _filename_tokens(doc_id: str) -> str:
    """Extract searchable tokens from a doc_id (filename).

    "contractnli/ceii-and-nda.txt"
    -> strip prefix and extension -> "ceii-and-nda"
    -> split on hyphens/underscores/dots -> "ceii and nda"
    -> also keep the unsplit form for substring matching
    """
    # Strip directory prefix and file extension
    name = doc_id.split("/")[-1]
    name = re.sub(r"\.[^.]+$", "", name)  # strip extension

    # Split on delimiters, keep originals too
    tokens = re.split(r"[-_.\s]+", name.lower())
    # Also include the full unsplit name for phrase matching
    return name.lower() + " " + " ".join(tokens)


# Domain stopwords: tokens that appear in nearly every NDA routing text
# and provide zero discrimination for document identity routing.
_ROUTING_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "with", "that", "this", "it", "be", "has", "have", "and",
    "or", "but", "not", "by", "from", "between", "consider", "does",
    "document", "may", "shall", "any", "all", "its", "their", "party",
    # NDA-specific: appear in all 95 documents, zero discrimination
    "non-disclosure", "nondisclosure", "nda", "agreement", "confidential",
    "confidentiality", "information", "mutual", "one-way", "parties",
    "undated", "type", "subject", "date", "disclosed", "disclosing",
    "receiving", "contractnli",
})


def _tokenize(text: str) -> list[str]:
    """Tokenize with domain-specific stopword removal for routing BM25."""
    # Split on whitespace, strip punctuation from each token
    raw_tokens = re.split(r"\s+", text.strip().lower())
    tokens = [re.sub(r"[^a-z0-9_\-]", "", t) for t in raw_tokens]
    return [t for t in tokens if t and t not in _ROUTING_STOPWORDS]


def build_routing_index(
    summaries_path: Path = SUMMARIES_PATH,
    output_path: Path = ROUTING_INDEX_PATH,
) -> int:
    """Embed SAC summaries and build BM25 routing texts. Persist to disk.

    Returns the number of documents indexed.
    """
    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    doc_ids = sorted(summaries.keys())
    texts = [summaries[d] for d in doc_ids]

    # Dense channel: embed summaries
    vo = voyageai.Client()
    embeddings = vo.embed(texts, model=VOYAGE_MODEL, input_type="document").embeddings
    vectors = np.array(embeddings, dtype=np.float32)

    # BM25 channel: build routing texts (filename tokens + summary)
    routing_texts = []
    for doc_id in doc_ids:
        fn_tokens = _filename_tokens(doc_id)
        summary = summaries[doc_id]
        routing_texts.append(fn_tokens + " " + summary)

    np.savez(
        output_path,
        vectors=vectors,
        doc_ids=np.array(doc_ids),
        routing_texts=np.array(routing_texts),
    )
    logger.info("Routing index: %d documents (dense + BM25) -> %s",
                len(doc_ids), output_path)
    return len(doc_ids)


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


class DocumentRouter:
    """Hybrid routing index: dense (summary) + BM25 (filename + summary)."""

    def __init__(self, index_path: Path = ROUTING_INDEX_PATH) -> None:
        data = np.load(index_path, allow_pickle=True)
        self._vectors = data["vectors"]  # (N, dim)
        self._doc_ids = list(data["doc_ids"])
        self._vo = voyageai.Client()

        # Pre-normalize dense vectors for cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._normed = self._vectors / norms

        # Build BM25 index from routing texts
        routing_texts = list(data["routing_texts"])
        tokenized = [_tokenize(t) for t in routing_texts]
        self._bm25 = BM25Okapi(tokenized)

    def route(self, query: str, top_k: int = 3,
              alpha: float | None = None) -> list[str]:
        """Return the top_k nearest document IDs using hybrid scoring.

        alpha controls BM25 weight in CC fusion:
          combined = alpha * norm(bm25) + (1-alpha) * norm(dense)
        Default alpha from MERIDIAN_ROUTING_ALPHA env var, or 0.3.
        """
        if alpha is None:
            alpha = float(os.environ.get("MERIDIAN_ROUTING_ALPHA", "0.5"))

        # Dense channel
        query_emb = self._vo.embed(
            [query], model=VOYAGE_MODEL, input_type="query"
        ).embeddings[0]
        query_vec = np.array(query_emb, dtype=np.float32)
        query_norm = query_vec / (np.linalg.norm(query_vec) or 1.0)
        dense_scores = self._normed @ query_norm

        # BM25 channel
        query_tokens = _tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)

        # CC fusion: min-max normalize each, then combine
        dense_norm = _minmax_normalize(dense_scores)
        bm25_norm = _minmax_normalize(bm25_scores)
        combined = alpha * bm25_norm + (1 - alpha) * dense_norm

        top_indices = np.argsort(combined)[::-1][:top_k]
        return [self._doc_ids[i] for i in top_indices]


# Module-level singleton (lazy-loaded, thread-safe enough for read-only)
_router: DocumentRouter | None = None


def route_query(query: str, top_k: int = 3) -> list[str]:
    """Route a query to the nearest top_k documents.

    Lazy-loads the routing index on first call. Thread-safe for
    concurrent reads (no mutation after init).
    """
    global _router
    if _router is None:
        _router = DocumentRouter()
    return _router.route(query, top_k)
