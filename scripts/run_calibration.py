"""Measurement-layer calibration: replicate the LegalBench-RAG paper's exact
baseline stack through OUR measurement layer and compare to published Table 5.

Paper: arXiv 2408.10343 (ZeroEntropy, LegalBench-RAG)
Stack: RCTS 500-char / text-embedding-3-large / dense cosine / no reranker
Split: 194-query mini per dataset

This is the empirical foundation — if our P@k/R@k matches the paper's within
noise, our measurement layer is calibrated to the published standard. Every
comparison built on it is valid.

Usage:
    python scripts/run_calibration.py
    python scripts/run_calibration.py --corpus contractnli --skip-index
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("calibration")

# ── Paper's exact config (from zeroentropy-ai/legalbenchrag) ─────────────

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
RCTS_SEPARATORS = ["\n\n", "\n", "!", "?", ".", ":", ";", ",", " ", ""]
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
K_VALUES = [1, 2, 4, 8, 16, 32, 64]

# Paper Table 5: RCTS, no reranker (the calibration target)
PAPER_TABLE5 = {
    "contractnli": {
        "p_at_k": {1: 6.63, 2: 5.29, 4: 3.89, 8: 2.81, 16: 1.98, 32: 1.29, 64: 0.90},
        "r_at_k": {1: 7.63, 2: 11.33, 4: 17.34, 8: 24.99, 16: 35.80, 32: 46.57, 64: 61.72},
    },
    "cuad": {
        "p_at_k": {1: 1.97, 2: 4.03, 4: 4.83, 8: 4.20, 16: 2.94, 32: 1.99, 64: 1.25},
        "r_at_k": {1: 1.62, 2: 8.11, 4: 17.72, 8: 31.68, 16: 44.38, 32: 60.04, 64: 74.70},
    },
    "maud": {
        "p_at_k": {1: 2.65, 2: 1.77, 4: 1.96, 8: 1.40, 16: 1.39, 32: 1.15, 64: 0.82},
        "r_at_k": {1: 1.65, 2: 2.09, 4: 4.59, 8: 6.18, 16: 12.93, 32: 21.04, 64: 28.28},
    },
    "privacyqa": {
        "p_at_k": {1: 14.38, 2: 13.55, 4: 12.34, 8: 9.03, 16: 6.06, 32: 4.17, 64: 2.81},
        "r_at_k": {1: 8.85, 2: 15.21, 4: 27.92, 8: 42.37, 16: 55.12, 32: 71.19, 64: 84.19},
    },
}

CORPUS_CONFIG = {
    "contractnli": {
        "benchmark": Path("data/benchmarks/contractnli.json"),
        "corpus_dir": Path("data/corpus"),
        "gt_module": "core.evaluation.ground_truth_contractnli",
        "gt_class": "ContractNLIGroundTruth",
        "dataset_name": "contractnli",
    },
    "cuad": {
        "benchmark": Path("data/benchmarks/cuad.json"),
        "corpus_dir": Path("data/corpus"),
        "gt_module": "core.evaluation.ground_truth_cuad",
        "gt_class": "CUADGroundTruth",
        "dataset_name": "cuad",
    },
    "maud": {
        "benchmark": Path("data/benchmarks/maud.json"),
        "corpus_dir": Path("data/corpus"),
        "gt_module": "core.evaluation.ground_truth_maud",
        "gt_class": "MAUDGroundTruth",
        "dataset_name": "maud",
    },
    "privacyqa": {
        "benchmark": Path("data/benchmarks/privacy_qa.json"),
        "corpus_dir": Path("data/corpus"),
        "gt_module": "core.evaluation.ground_truth_privacyqa",
        "gt_class": "PrivacyQAGroundTruth",
        "dataset_name": "privacy_qa",
    },
}

QDRANT_COLLECTION = "calibration_rcts_combined"

# ── Chunking (paper-exact RCTS) ─────────────────────────────────────────

def chunk_corpus_rcts(corpus: str) -> list[dict]:
    """Chunk corpus documents with the paper's exact RCTS config.

    Returns list of {doc_id, chunk_id, content, span: (start, end)}.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cfg = CORPUS_CONFIG[corpus]
    bench = json.loads(cfg["benchmark"].read_text(encoding="utf-8"))

    # Get mini-split document paths
    doc_paths = set()
    for t in bench["tests"]:
        for s in t["snippets"]:
            doc_paths.add(s["file_path"])

    splitter = RecursiveCharacterTextSplitter(
        separators=RCTS_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        strip_whitespace=False,
    )

    chunks = []
    for doc_path in sorted(doc_paths):
        full_path = cfg["corpus_dir"] / doc_path
        if not full_path.exists():
            full_path = cfg["corpus_dir"] / unquote(doc_path)
        if not full_path.exists():
            logger.warning("Missing document: %s", doc_path)
            continue

        text = full_path.read_text(encoding="utf-8")
        text_splits = splitter.split_text(text)

        # Verify no characters lost (paper asserts this)
        assert "".join(text_splits) == text, f"Split/join mismatch on {doc_path}"

        # Build spans
        offset = 0
        for i, split_text in enumerate(text_splits):
            span = (offset, offset + len(split_text))
            chunks.append({
                "doc_id": doc_path,
                "chunk_id": f"{doc_path}#rcts-{i:04d}",
                "content": split_text,
                "span": span,
            })
            offset += len(split_text)

    logger.info("%s: %d docs → %d chunks (RCTS %d-char)",
                corpus, len(doc_paths), len(chunks), CHUNK_SIZE)
    return chunks


# ── Embedding (text-embedding-3-large) ───────────────────────────────────

def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed texts with OpenAI text-embedding-3-large."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    all_embeddings = []
    batch_size = 2048

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_embs = [d.embedding for d in resp.data]
        all_embeddings.extend(batch_embs)
        if i + batch_size < len(texts):
            time.sleep(0.5)

    return all_embeddings


# ── Indexing (Qdrant) ────────────────────────────────────────────────────

def index_combined(all_chunks: list[dict]) -> tuple[str, list[dict]]:
    """Create ONE sqlite-vec DB with chunks from ALL corpora (paper-exact).

    Returns (db_path, chunk_metadata) for retrieval.
    """
    import sqlite3
    import struct

    import sqlite_vec

    db_path = "data/calibration_combined.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Embed all chunks
    logger.info("Embedding %d chunks with %s...", len(all_chunks), EMBED_MODEL)
    embeddings = embed_texts([c["content"] for c in all_chunks])

    # Create sqlite-vec DB (paper-exact approach)
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute(f"PRAGMA mmap_size = {3*1024*1024*1024}")
    db.execute(
        f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{EMBED_DIM}])"
    )

    def serialize_f32(vector):
        return struct.pack(f"{len(vector)}f", *vector)

    with db:
        insert_data = [
            (i, serialize_f32(emb)) for i, emb in enumerate(embeddings)
        ]
        db.executemany(
            "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
            insert_data,
        )

    db.close()
    logger.info("Indexed %d vectors in sqlite-vec DB '%s'", len(embeddings), db_path)
    return db_path, all_chunks


# ── Retrieval (sqlite-vec, paper-exact) ──────────────────────────────────

_db_conn = None
_db_path_cached = None


def retrieve_query(query: str, db_path: str, chunks_meta: list[dict],
                   top_k: int = 64) -> list[dict]:
    """Retrieve top-k chunks using sqlite-vec exact cosine search (paper-exact)."""
    import sqlite3
    import struct

    import sqlite_vec

    global _db_conn, _db_path_cached
    if _db_conn is None or _db_path_cached != db_path:
        if _db_conn is not None:
            _db_conn.close()
        _db_conn = sqlite3.connect(db_path)
        _db_conn.enable_load_extension(True)
        sqlite_vec.load(_db_conn)
        _db_conn.enable_load_extension(False)
        _db_path_cached = db_path

    def serialize_f32(vector):
        return struct.pack(f"{len(vector)}f", *vector)

    query_emb = embed_texts([query])[0]

    rows = _db_conn.execute(
        """
        SELECT rowid, distance
        FROM vec_items
        WHERE embedding MATCH ?
        ORDER BY distance ASC
        LIMIT ?
        """,
        [serialize_f32(query_emb), top_k],
    ).fetchall()

    return [
        {
            "chunk_id": chunks_meta[row_id]["chunk_id"],
            "doc_id": chunks_meta[row_id]["doc_id"],
            "span": chunks_meta[row_id]["span"],
            "score": 1.0 / (i + 1),
        }
        for i, (row_id, dist) in enumerate(rows)
    ]


# ── Measurement (through OUR layer) ─────────────────────────────────────

def _paper_precision_recall(retrieved: list[dict], gt_snippets: list[dict],
                            top_k: int) -> tuple[float, float]:
    """Compute precision and recall using the paper's EXACT formula.

    Key: doc_id must match for overlap to count, but wrong-doc chunks still
    inflate the precision denominator (they occupy top-k slots).
    """
    top_k_results = retrieved[:top_k]

    # Precision
    total_retrieved_len = 0
    relevant_retrieved_len = 0
    for r in top_k_results:
        span_len = r["span"][1] - r["span"][0]
        total_retrieved_len += span_len
        for gt in gt_snippets:
            if r["doc_id"] == gt["file_path"]:
                common_min = max(r["span"][0], gt["span"][0])
                common_max = min(r["span"][1], gt["span"][1])
                if common_max > common_min:
                    relevant_retrieved_len += common_max - common_min

    precision = relevant_retrieved_len / total_retrieved_len if total_retrieved_len > 0 else 0.0

    # Recall
    total_relevant_len = 0
    relevant_found_len = 0
    for gt in gt_snippets:
        gt_len = gt["span"][1] - gt["span"][0]
        total_relevant_len += gt_len
        for r in top_k_results:
            if r["doc_id"] == gt["file_path"]:
                common_min = max(r["span"][0], gt["span"][0])
                common_max = min(r["span"][1], gt["span"][1])
                if common_max > common_min:
                    relevant_found_len += common_max - common_min

    recall = relevant_found_len / total_relevant_len if total_relevant_len > 0 else 0.0

    return precision, recall


def measure_corpus(corpus: str, db_path: str, chunks_meta: list[dict]) -> dict:
    """Run all 194 queries, compute P@k/R@k using the paper's exact formula."""
    cfg = CORPUS_CONFIG[corpus]
    bench = json.loads(cfg["benchmark"].read_text(encoding="utf-8"))

    queries = bench["tests"]

    # Accumulate per-query metrics
    all_p = {k: [] for k in K_VALUES}
    all_r = {k: [] for k in K_VALUES}

    for i, q in enumerate(queries):
        query_text = q["query"]

        # Build ground-truth snippet list (with file_path + span)
        gt_snippets = [
            {"file_path": s["file_path"], "span": tuple(s["span"])}
            for s in q["snippets"]
        ]

        # Retrieve top-64 from combined index
        results = retrieve_query(query_text, db_path, chunks_meta, top_k=64)

        # Compute paper-exact metrics at each k
        for k in K_VALUES:
            p, r = _paper_precision_recall(results, gt_snippets, top_k=k)
            all_p[k].append(p)
            all_r[k].append(r)

        if (i + 1) % 20 == 0 or i == len(queries) - 1:
            logger.info("  %s: %d/%d queries measured", corpus, i + 1, len(queries))

    # Average across queries (paper does equal-weight per query)
    avg_p = {k: sum(v) / len(v) * 100 for k, v in all_p.items()}
    avg_r = {k: sum(v) / len(v) * 100 for k, v in all_r.items()}

    return {"p_at_k": avg_p, "r_at_k": avg_r, "n_queries": len(queries)}


# ── Comparison ───────────────────────────────────────────────────────────

def compare_to_paper(corpus: str, measured: dict) -> dict:
    """Compare measured metrics to paper Table 5."""
    paper = PAPER_TABLE5[corpus]

    print(f"\n{'=' * 80}")
    print(f"CALIBRATION: {corpus.upper()}")
    print(f"{'=' * 80}")

    print(f"\n{'k':>4}  {'Paper P@k':>10} {'Ours P@k':>10} {'Delta':>8}  "
          f"{'Paper R@k':>10} {'Ours R@k':>10} {'Delta':>8}")
    print("-" * 72)

    max_p_delta = 0
    max_r_delta = 0

    for k in K_VALUES:
        pp = paper["p_at_k"][k]
        mp = measured["p_at_k"][k]
        dp = mp - pp

        pr = paper["r_at_k"][k]
        mr = measured["r_at_k"][k]
        dr = mr - pr

        max_p_delta = max(max_p_delta, abs(dp))
        max_r_delta = max(max_r_delta, abs(dr))

        flag_p = " !" if abs(dp) > 0.5 else ""
        flag_r = " !" if abs(dr) > 2.0 else ""

        print(f"  {k:>2}  {pp:>9.2f}% {mp:>9.2f}% {dp:>+7.2f}{flag_p}  "
              f"{pr:>9.2f}% {mr:>9.2f}% {dr:>+7.2f}{flag_r}")

    # Verdict
    p1_ok = max_p_delta <= 0.5
    r8_ok = max_r_delta <= 2.0
    calibrated = p1_ok and r8_ok

    print(f"\n  Max |delta| P@k: {max_p_delta:.2f}pp  (threshold ±0.5pp)")
    print(f"  Max |delta| R@k: {max_r_delta:.2f}pp  (threshold ±2.0pp)")

    if calibrated:
        print(f"\n  VERDICT: CALIBRATED — measurement layer reproduces paper within noise")
    else:
        print(f"\n  VERDICT: DIVERGES — measurement layer does NOT match paper")
        if not p1_ok:
            print(f"    P@k exceeds ±0.5pp threshold")
        if not r8_ok:
            print(f"    R@k exceeds ±2.0pp threshold")

    return {
        "corpus": corpus,
        "calibrated": calibrated,
        "max_p_delta": max_p_delta,
        "max_r_delta": max_r_delta,
        "measured": measured,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Measurement-layer calibration")
    p.add_argument("--corpus", nargs="*", default=list(CORPUS_CONFIG.keys()),
                   choices=list(CORPUS_CONFIG.keys()))
    p.add_argument("--skip-index", action="store_true",
                   help="Skip chunking/embedding/indexing (reuse existing collections)")
    p.add_argument("--output", type=Path, default=Path("data/calibration_results.json"))
    args = p.parse_args()

    print("MEASUREMENT-LAYER CALIBRATION")
    print(f"Paper: arXiv 2408.10343, Table 5 (RCTS, no reranker)")
    print(f"Stack: RCTS {CHUNK_SIZE}-char / {EMBED_MODEL} / dense cosine / no reranker")
    print(f"Index: COMBINED (all 4 corpora in one collection, matching paper)")
    print(f"Corpora to measure: {args.corpus}")
    print(f"Skip index: {args.skip_index}")
    print()

    db_path = "data/calibration_combined.db"
    all_chunks = []

    if not args.skip_index:
        # Step 1: Chunk ALL corpora (paper indexes all 4 into one collection)
        for corpus in list(CORPUS_CONFIG.keys()):
            chunks = chunk_corpus_rcts(corpus)
            all_chunks.extend(chunks)
        logger.info("Total: %d chunks across all corpora", len(all_chunks))

        # Step 2: Index into ONE combined sqlite-vec DB
        db_path, all_chunks = index_combined(all_chunks)
    else:
        # Rebuild chunk metadata even when skipping index
        for corpus in list(CORPUS_CONFIG.keys()):
            chunks = chunk_corpus_rcts(corpus)
            all_chunks.extend(chunks)

    all_results = {}

    for corpus in args.corpus:
        # Step 3: Measure per-corpus (queries from this corpus, retrieval from combined)
        measured = measure_corpus(corpus, db_path, all_chunks)

        # Step 4: Compare
        result = compare_to_paper(corpus, measured)
        all_results[corpus] = result

    # Summary
    print(f"\n{'=' * 80}")
    print("CALIBRATION SUMMARY")
    print(f"{'=' * 80}")

    all_calibrated = True
    for corpus, result in all_results.items():
        status = "CALIBRATED" if result["calibrated"] else "DIVERGES"
        print(f"  {corpus:<15} {status}  (max |dP|={result['max_p_delta']:.2f}pp, "
              f"max |dR|={result['max_r_delta']:.2f}pp)")
        if not result["calibrated"]:
            all_calibrated = False

    if all_calibrated:
        print(f"\n  ALL CORPORA CALIBRATED — measurement layer is empirically validated")
        print(f"  against the published LegalBench-RAG standard.")
    else:
        print(f"\n  CALIBRATION FAILED — STOP and investigate divergence before any")
        print(f"  external comparison is trusted.")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {args.output}")


if __name__ == "__main__":
    main()
