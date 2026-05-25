"""Ingestion integrity tests for ContractNLI.

These tests require the data to be downloaded and ingested first.
Run ``python scripts/run_baseline.py`` to trigger ingestion,
then run these tests.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from campaigns.legalbench_rag.ground_truth_adapter import ContractNLIGroundTruth
from campaigns.legalbench_rag.ingestion.contractnli_loader import ingest_contractnli

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PARQUET = DATA_DIR / "corpus.parquet"
BENCHMARK_JSON = DATA_DIR / "benchmarks" / "contractnli.json"
CORPUS_DIR = DATA_DIR / "corpus"


@pytest.fixture(scope="module")
def ingested_data():
    """Run ingestion if corpus.parquet doesn't exist yet."""
    if not CORPUS_PARQUET.exists():
        corpus_df, queries = ingest_contractnli(DATA_DIR)
        return corpus_df, queries
    corpus_df = pd.read_parquet(CORPUS_PARQUET)
    return corpus_df, None


@pytest.fixture(scope="module")
def corpus_df(ingested_data):
    return ingested_data[0]


@pytest.fixture(scope="module")
def gt():
    """Load ground truth adapter (requires ingested data on disk)."""
    if not BENCHMARK_JSON.exists():
        pytest.skip("Benchmark JSON not found — run ingestion first")
    return ContractNLIGroundTruth(BENCHMARK_JSON, CORPUS_DIR)


# ==================================================================
# Offset integrity: doc_text[start:end] == chunk_text
# ==================================================================


def test_offset_integrity(corpus_df):
    """doc_text[start:end] must equal chunk content for every chunk."""
    doc_texts: dict[str, str] = {}

    for _, row in corpus_df.iterrows():
        doc_id = row["doc_id"]
        if doc_id not in doc_texts:
            doc_path = CORPUS_DIR / doc_id
            if not doc_path.exists():
                pytest.fail(f"Corpus file missing: {doc_path}")
            with open(doc_path, encoding="utf-8") as f:
                doc_texts[doc_id] = f.read()

        start, end = row["start_end_idx"]
        expected = doc_texts[doc_id][start:end]
        actual = row["content"]
        assert actual == expected, (
            f"Offset mismatch in {doc_id} at [{start}:{end}]: "
            f"expected {expected!r:.80}, got {actual!r:.80}"
        )


# ==================================================================
# No duplicate chunk_ids
# ==================================================================


def test_no_duplicate_chunk_ids(corpus_df):
    """Every chunk_id must be unique."""
    chunk_ids = corpus_df["chunk_id"].tolist()
    dupes = {cid for cid, cnt in Counter(chunk_ids).items() if cnt > 1}
    assert len(dupes) == 0, f"Duplicate chunk_ids: {dupes}"


# ==================================================================
# Ground truth query coverage
# ==================================================================


def test_all_query_ids_resolvable(gt):
    """All query IDs must be resolvable via the ground truth adapter."""
    query_ids = gt.all_query_ids()
    assert len(query_ids) > 0, "No query IDs found"

    for qid in query_ids:
        spans = gt.get_spans(qid)
        assert len(spans) > 0, f"Query {qid} has no GT spans"

        doc_id = gt.get_doc_id(qid)
        assert doc_id, f"Query {qid} has no doc_id"

        text = gt.doc_text(doc_id)
        assert len(text) > 0, f"Doc {doc_id} is empty"


# ==================================================================
# GT span validity
# ==================================================================


def test_gt_spans_valid(gt):
    """All GT spans must satisfy: start < end, end <= len(doc_text)."""
    for qid in gt.all_query_ids():
        spans = gt.get_spans(qid)
        doc_id = gt.get_doc_id(qid)
        text = gt.doc_text(doc_id)
        doc_len = len(text)

        for start, end in spans:
            assert start < end, (
                f"Query {qid}: span ({start}, {end}) has start >= end"
            )
            assert end <= doc_len, (
                f"Query {qid}: span end {end} > doc length {doc_len}"
            )


# ==================================================================
# Checksum consistency
# ==================================================================


def test_checksum_consistency(corpus_df):
    """Reloading the same doc must produce the same checksum."""
    seen: dict[str, str] = {}

    for _, row in corpus_df.iterrows():
        doc_id = row["doc_id"]
        stored_checksum = row["checksum"]

        if doc_id not in seen:
            doc_path = CORPUS_DIR / doc_id
            if not doc_path.exists():
                pytest.fail(f"Corpus file missing: {doc_path}")
            with open(doc_path, encoding="utf-8") as f:
                text = f.read()
            computed = hashlib.sha256(text.encode("utf-8")).hexdigest()
            seen[doc_id] = computed

        assert stored_checksum == seen[doc_id], (
            f"Checksum mismatch for {doc_id}: "
            f"stored={stored_checksum}, computed={seen[doc_id]}"
        )


# ==================================================================
# Corpus schema
# ==================================================================


def test_corpus_schema(corpus_df):
    """corpus.parquet must have the required columns."""
    required = {"doc_id", "content", "start_end_idx", "chunk_id", "checksum"}
    actual = set(corpus_df.columns)
    missing = required - actual
    assert not missing, f"Missing columns in corpus.parquet: {missing}"
