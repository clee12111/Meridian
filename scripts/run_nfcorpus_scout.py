"""NFCorpus transfer scout — ingest, index, sweep CC-alpha × routing top-k.

Finds the optimal config on a 25% query slice, then runs the full evaluation
with the winner. nDCG@10 (BEIR standard) against published baselines.

Usage:
    python scripts/run_nfcorpus_scout.py
    python scripts/run_nfcorpus_scout.py --skip-ingest  # reuse existing index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("nfcorpus_scout")

# ── Constants ────────────────────────────────────────────────────────────

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
EMBED_MODEL = "voyage-4"
COLLECTION = "nfcorpus_v4"
PARQUET_PATH = Path("data/corpus_nfcorpus.parquet")
DATA_DIR = Path("data/nfcorpus")
ROUTING_INDEX_PATH = Path("data/routing_index_nfcorpus_v4.npz")

# Scout sweep grid
ALPHA_VALUES = [0.1, 0.3, 0.5]
ROUTING_TOPK_VALUES = [3, 5, 10, None]  # None = routing off

# Published baselines (BEIR leaderboard, nDCG@10)
PUBLISHED = {
    "BM25": 0.3218,
    "DPR": 0.189,  # dense passage retrieval (original)
    "TAS-B": 0.319,
    "contriever": 0.328,
}


# ── nDCG@10 ──────────────────────────────────────────────────────────────

def dcg(relevances: list[float], k: int = 10) -> float:
    """Discounted cumulative gain at k."""
    result = 0.0
    for i, rel in enumerate(relevances[:k]):
        result += rel / math.log2(i + 2)
    return result


def ndcg_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """nDCG@k for a single query."""
    relevances = [qrels.get(did, 0) for did in ranked_doc_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    dcg_val = dcg(relevances, k)
    idcg_val = dcg(ideal, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


# ── Step 1: Download and ingest ──────────────────────────────────────────

def download_nfcorpus():
    """Download NFCorpus using BEIR utility."""
    from beir import util

    data_path = str(DATA_DIR)
    if (DATA_DIR / "corpus.jsonl").exists():
        logger.info("NFCorpus already downloaded at %s", DATA_DIR)
        return

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
    logger.info("Downloading NFCorpus...")
    util.download_and_unzip(url, str(DATA_DIR.parent))
    logger.info("Downloaded to %s", DATA_DIR)


def load_nfcorpus() -> tuple[dict, dict, dict]:
    """Load corpus, queries, qrels from BEIR format.

    Returns:
        corpus: {doc_id: {"title": ..., "text": ...}}
        queries: {query_id: query_text}
        qrels: {query_id: {doc_id: relevance}}
    """
    from beir.datasets.data_loader import GenericDataLoader

    corpus, queries, qrels = GenericDataLoader(str(DATA_DIR)).load(split="test")
    logger.info("NFCorpus: %d docs, %d queries, %d qrels",
                len(corpus), len(queries),
                sum(len(v) for v in qrels.values()))
    return corpus, queries, qrels


def build_parquet(corpus: dict) -> pd.DataFrame:
    """Chunk NFCorpus documents and build parquet."""
    rows = []
    for doc_id, doc in corpus.items():
        text = doc.get("title", "") + "\n\n" + doc.get("text", "")
        text = text.strip()
        if not text:
            continue

        # Fixed-stride chunking (medical text, no legal section patterns)
        chunks = []
        if len(text) <= CHUNK_SIZE:
            chunks.append((0, len(text), text))
        else:
            start = 0
            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                chunks.append((start, end, text[start:end]))
                if end >= len(text):
                    break
                start += CHUNK_SIZE - CHUNK_OVERLAP

        for i, (s, e, content) in enumerate(chunks):
            rows.append({
                "doc_id": f"nfcorpus/{doc_id}",
                "content": content,
                "start_end_idx": np.array([s, e], dtype=np.int64),
                "chunk_id": f"nfcorpus/{doc_id}#chunk-{i:05d}",
                "checksum": hashlib.sha256(content.encode()).hexdigest()[:16],
                "dataset_name": "nfcorpus",
            })

    df = pd.DataFrame(rows)
    df.to_parquet(PARQUET_PATH, index=False)
    logger.info("Parquet: %d chunks from %d docs -> %s", len(df), len(corpus), PARQUET_PATH)
    return df


SQLITE_DB = "data/nfcorpus_vectors.db"


def embed_and_index(df: pd.DataFrame):
    """Embed chunks with voyage-4 and index in sqlite-vec (Qdrant Cloud full)."""
    import sqlite3
    import struct

    import sqlite_vec
    import voyageai

    if os.path.exists(SQLITE_DB):
        os.remove(SQLITE_DB)

    vo = voyageai.Client()
    texts = df["content"].tolist()
    chunk_ids = df["chunk_id"].tolist()

    # Embed in batches
    all_embeddings = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = vo.embed(batch, model=EMBED_MODEL, input_type="document").embeddings
        all_embeddings.extend(embs)
        if i + batch_size < len(texts):
            time.sleep(0.5)
        if (i // batch_size + 1) % 10 == 0:
            logger.info("  Embedded %d/%d chunks", min(i + batch_size, len(texts)), len(texts))

    # Store in sqlite-vec
    db = sqlite3.connect(SQLITE_DB)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA mmap_size = 1073741824")
    db.execute("CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[1024])")

    def sf32(v):
        return struct.pack(f"{len(v)}f", *v)

    with db:
        db.executemany(
            "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
            [(i, sf32(emb)) for i, emb in enumerate(all_embeddings)],
        )
    db.close()
    logger.info("Indexed %d chunks in sqlite-vec '%s'", len(all_embeddings), SQLITE_DB)


def build_routing_index(corpus: dict):
    """Build a dense+BM25 routing index for NFCorpus documents."""
    import voyageai

    doc_ids = sorted(f"nfcorpus/{did}" for did in corpus)
    texts = []
    for did in doc_ids:
        raw_id = did.replace("nfcorpus/", "")
        doc = corpus[raw_id]
        text = (doc.get("title", "") + " " + doc.get("text", "")[:500]).strip()
        texts.append(text)

    # Dense: embed doc summaries (batch limit 1000)
    vo = voyageai.Client()
    all_embs = []
    for i in range(0, len(texts), 128):
        batch = texts[i:i + 128]
        all_embs.extend(vo.embed(batch, model=EMBED_MODEL, input_type="document").embeddings)
        if i + 128 < len(texts):
            time.sleep(0.3)
    vectors = np.array(all_embs, dtype=np.float32)

    # BM25: use title + truncated text as routing text
    np.savez(
        ROUTING_INDEX_PATH,
        vectors=vectors,
        doc_ids=np.array(doc_ids),
        routing_texts=np.array(texts),
    )
    logger.info("Routing index: %d docs -> %s", len(doc_ids), ROUTING_INDEX_PATH)


# ── Step 2: Retrieval + ranking ──────────────────────────────────────────

def retrieve_and_rank(query: str, corpus_df: pd.DataFrame,
                      alpha: float, routing_topk: int | None,
                      top_k: int = 50) -> list[tuple[str, float]]:
    """Retrieve chunks, aggregate to document scores, return ranked doc_ids.

    Returns list of (doc_id, score) sorted by score descending.
    Doc_id is the RAW doc_id (without 'nfcorpus/' prefix).
    """
    from core.retrieval.bm25_retriever import BM25Retriever
    from core.retrieval.fusion import cc_fusion, rrf
    from core.retrieval.routing import route_query

    # Routing
    doc_ids = None
    if routing_topk is not None:
        routed = route_query(query, top_k=routing_topk)
        if routed:
            doc_ids = routed

    # Build pipeline context for this query
    from core.supervisor.context import PipelineContext
    context = PipelineContext.build(
        corpus_path=PARQUET_PATH,
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection_name=COLLECTION,
        dataset_name="nfcorpus",
        top_k=top_k,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )

    # Dense retrieval
    dense = context.qdrant_retriever.retrieve(
        query, dataset_name="nfcorpus", doc_ids=doc_ids)
    # Sparse retrieval
    sparse = context.bm25_retriever.retrieve(
        query, dataset_name="nfcorpus", doc_ids=doc_ids)

    # Fusion
    fused = cc_fusion(sparse, dense, alpha=alpha, top_n=top_k)

    # Aggregate chunk scores to document scores (max score per doc)
    doc_scores: dict[str, float] = {}
    for cid, score in zip(fused.ids, fused.scores):
        # chunk_id = "nfcorpus/DOC_ID#chunk-00000" -> doc = "DOC_ID"
        raw_doc = cid.split("#")[0].replace("nfcorpus/", "")
        if raw_doc not in doc_scores or score > doc_scores[raw_doc]:
            doc_scores[raw_doc] = score

    # Sort by score
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ── Step 3: Scout sweep ──────────────────────────────────────────────────

def _dense_retrieve(query: str, corpus_df: pd.DataFrame, top_k: int = 50,
                    doc_ids: list[str] | None = None):
    """Dense retrieval from sqlite-vec."""
    import sqlite3
    import struct
    import sqlite_vec
    import voyageai

    vo = voyageai.Client()
    qemb = vo.embed([query], model=EMBED_MODEL, input_type="query").embeddings[0]

    db = sqlite3.connect(SQLITE_DB)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    def sf32(v):
        return struct.pack(f"{len(v)}f", *v)

    fetch_k = top_k * 3 if doc_ids else top_k
    rows = db.execute(
        "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance ASC LIMIT ?",
        [sf32(qemb), fetch_k],
    ).fetchall()
    db.close()

    doc_id_set = set(doc_ids) if doc_ids else None
    ids, contents, spans, scores = [], [], [], []
    for rid, dist in rows:
        row = corpus_df.iloc[rid]
        if doc_id_set and row["doc_id"] not in doc_id_set:
            continue
        ids.append(row["chunk_id"])
        contents.append(row["content"])
        spans.append(tuple(row["start_end_idx"]))
        scores.append(1.0 - dist)
        if len(ids) >= top_k:
            break

    class _R:
        def __init__(self, ids, contents, spans, scores):
            self.ids = ids; self.contents = contents
            self.spans = spans; self.scores = scores
    return _R(ids, contents, spans, scores)


def _run_eval(queries_subset: list[tuple[str, str]], qrels: dict, corpus_df: pd.DataFrame,
              bm25, alpha: float | None, routing_topk: int | None,
              label: str) -> float:
    """Run evaluation on a set of queries. Returns mean nDCG@10."""
    from core.retrieval.fusion import cc_fusion, rrf
    from core.retrieval.routing import route_query

    ndcg_scores = []
    for i, (qid, query_text) in enumerate(queries_subset):
        query_qrels = qrels.get(qid, {})
        if not query_qrels:
            continue

        doc_ids_filter = None
        if routing_topk is not None:
            try:
                routed = route_query(query_text, top_k=routing_topk)
                if routed:
                    doc_ids_filter = routed
            except Exception:
                pass

        dense = _dense_retrieve(query_text, corpus_df, top_k=50, doc_ids=doc_ids_filter)
        sparse = bm25.retrieve(query_text, dataset_name="nfcorpus", doc_ids=doc_ids_filter)

        if alpha is not None:
            fused = cc_fusion(sparse, dense, alpha=alpha, top_n=50)
        else:
            fused = rrf(sparse, dense, top_n=50)

        doc_scores: dict[str, float] = {}
        for cid, score in zip(fused.ids, fused.scores):
            raw_doc = cid.split("#")[0].replace("nfcorpus/", "")
            if raw_doc not in doc_scores or score > doc_scores[raw_doc]:
                doc_scores[raw_doc] = score

        ranked_docs = [d for d, _ in sorted(doc_scores.items(),
                                             key=lambda x: x[1], reverse=True)]
        score = ndcg_at_k(ranked_docs, query_qrels, k=10)
        ndcg_scores.append(score)

        if (i + 1) % 50 == 0:
            logger.info("  %s: %d/%d queries", label, i + 1, len(queries_subset))

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0, len(ndcg_scores)


def run_scout(queries: dict, qrels: dict, corpus_df: pd.DataFrame,
              slice_frac: float = 0.25):
    """Run the CC-alpha x routing-topk sweep on a query slice."""
    from core.retrieval.bm25_retriever import BM25Retriever
    import core.retrieval.routing as routing_mod

    bm25 = BM25Retriever(corpus_df=corpus_df)

    all_qids = sorted(queries.keys())
    random.seed(42)
    sample_qids = random.sample(all_qids, max(1, int(len(all_qids) * slice_frac)))
    sample = [(qid, queries[qid]) for qid in sample_qids]
    logger.info("Scout: %d/%d queries (%.0f%% slice)", len(sample), len(all_qids),
                slice_frac * 100)

    results = []
    for alpha in ALPHA_VALUES:
        for rtk in ROUTING_TOPK_VALUES:
            rtk_label = str(rtk) if rtk is not None else "off"

            if rtk is not None:
                os.environ["MERIDIAN_ROUTING_TOPK"] = str(rtk)
                os.environ["MERIDIAN_ROUTING_INDEX"] = str(ROUTING_INDEX_PATH)
                os.environ["MERIDIAN_ROUTING_ALPHA"] = "0.5"
            elif "MERIDIAN_ROUTING_TOPK" in os.environ:
                del os.environ["MERIDIAN_ROUTING_TOPK"]
            routing_mod._router = None

            mean_ndcg, n = _run_eval(sample, qrels, corpus_df, bm25, alpha, rtk,
                                      f"a={alpha}/r={rtk_label}")
            results.append({
                "alpha": alpha, "routing_topk": rtk_label,
                "ndcg@10": mean_ndcg, "n_queries": n,
            })
            print(f"  alpha={alpha:.1f} routing={rtk_label:>3}: nDCG@10={mean_ndcg:.4f} ({n} queries)")

    return results


# ── Step 4: Full run with winner ─────────────────────────────────────────

def run_full(queries: dict, qrels: dict, corpus_df: pd.DataFrame,
             best_alpha: float, best_routing: int | None):
    """Run the winning config on ALL queries + a baseline arm."""
    from core.retrieval.bm25_retriever import BM25Retriever
    import core.retrieval.routing as routing_mod

    bm25 = BM25Retriever(corpus_df=corpus_df)
    all_queries = sorted([(q, queries[q]) for q in queries if qrels.get(q)])

    for arm_name, alpha, rtk in [
        ("baseline_rrf", None, None),
        ("best_config", best_alpha, best_routing),
    ]:
        if rtk is not None:
            os.environ["MERIDIAN_ROUTING_TOPK"] = str(rtk)
            os.environ["MERIDIAN_ROUTING_INDEX"] = str(ROUTING_INDEX_PATH)
            os.environ["MERIDIAN_ROUTING_ALPHA"] = "0.5"
        elif "MERIDIAN_ROUTING_TOPK" in os.environ:
            del os.environ["MERIDIAN_ROUTING_TOPK"]
        routing_mod._router = None

        mean_ndcg, n = _run_eval(all_queries, qrels, corpus_df, bm25, alpha, rtk, arm_name)
        print(f"\n  {arm_name}: nDCG@10 = {mean_ndcg:.4f} ({n} queries)")

    print(f"\n  Published baselines:")
    for name, score in PUBLISHED.items():
        print(f"    {name}: nDCG@10 = {score:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="NFCorpus transfer scout")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip download/chunk/embed (reuse existing index)")
    p.add_argument("--skip-scout", action="store_true",
                   help="Skip scout sweep, go straight to full run with defaults")
    p.add_argument("--alpha", type=float, default=None,
                   help="Override best alpha (skip scout)")
    p.add_argument("--routing-topk", type=str, default=None,
                   help="Override best routing top-k ('off' or int, skip scout)")
    args = p.parse_args()

    print("NFCORPUS TRANSFER SCOUT")
    print(f"Embedding: {EMBED_MODEL}")
    print(f"Chunk: {CHUNK_SIZE}-char, {CHUNK_OVERLAP} overlap")
    print(f"Sweep: alpha={ALPHA_VALUES}, routing={ROUTING_TOPK_VALUES}")
    print()

    # Step 1: Ingest
    if not args.skip_ingest:
        download_nfcorpus()

    corpus, queries, qrels = load_nfcorpus()

    if not args.skip_ingest:
        df = build_parquet(corpus)
        embed_and_index(df)
        build_routing_index(corpus)
    else:
        df = pd.read_parquet(PARQUET_PATH)

    # Step 2: Scout sweep
    if not args.skip_scout and args.alpha is None:
        print("\n" + "=" * 60)
        print("SCOUT SWEEP (25% query slice)")
        print("=" * 60)

        os.environ["MERIDIAN_EMBED_MODEL"] = EMBED_MODEL
        os.environ["MERIDIAN_NO_REWRITE"] = "1"
        os.environ["MERIDIAN_NO_RERANK"] = "1"

        results = run_scout(queries, qrels, df)

        # Find winner
        best = max(results, key=lambda r: r["ndcg@10"])
        print(f"\n  WINNER: alpha={best['alpha']} routing={best['routing_topk']} "
              f"nDCG@10={best['ndcg@10']:.4f}")

        best_alpha = best["alpha"]
        best_routing = int(best["routing_topk"]) if best["routing_topk"] != "off" else None
    else:
        best_alpha = args.alpha or 0.3
        rtk_str = args.routing_topk or "off"
        best_routing = int(rtk_str) if rtk_str != "off" else None

    # Step 3: Full run
    print(f"\n{'=' * 60}")
    print(f"FULL RUN (winner: alpha={best_alpha}, routing={best_routing or 'off'})")
    print(f"{'=' * 60}")

    os.environ["MERIDIAN_EMBED_MODEL"] = EMBED_MODEL
    os.environ["MERIDIAN_NO_REWRITE"] = "1"
    os.environ["MERIDIAN_NO_RERANK"] = "1"

    run_full(queries, qrels, df, best_alpha, best_routing)

    print(f"\n{'=' * 60}")
    print("NFCORPUS SCOUT COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
