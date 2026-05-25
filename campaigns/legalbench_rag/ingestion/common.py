"""Shared ingestion utilities for all LegalBench-RAG dataset loaders."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from urllib.parse import quote, unquote
from urllib.request import urlopen

import pandas as pd
from huggingface_hub import list_repo_tree


class IngestionError(Exception):
    """Raised when doc_text[start:end] != chunk_text (offset corruption)."""


def chunk_document(
    text: str, chunk_size: int = 512, overlap: int = 128
) -> list[tuple[int, int, str]]:
    """Split *text* into chunks with character offset tracking.

    Returns list of ``(start, end, chunk_text)`` tuples.
    Uses sentence-boundary-aware splitting: tries to break at sentence
    ends within the chunk window.  Half-open ``[start, end)`` intervals.
    """
    if not text:
        return []

    sentence_ends = [m.end() for m in re.finditer(r"[.!?]\s+", text)]

    chunks: list[tuple[int, int, str]] = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))

        if end < len(text):
            best_break = None
            for se in sentence_ends:
                if pos < se <= end:
                    best_break = se
            if best_break is not None:
                end = best_break

        chunk_text = text[pos:end]
        chunks.append((pos, end, chunk_text))

        if end >= len(text):
            break

        pos = end - overlap
        if pos <= chunks[-1][0]:
            pos = end

    return chunks


def sample_mini(queries: list[dict], max_queries: int = 194) -> list[dict]:
    """Sample queries matching LegalBench-RAG-mini logic.

    Sorts by document (seeded by file_path), then takes the first
    *max_queries* queries.
    """
    if len(queries) <= max_queries:
        return queries

    def _sort_key(q: dict) -> float:
        fp = q["snippets"][0]["file_path"]
        random.seed(fp)
        return random.random()

    sorted_queries = sorted(queries, key=_sort_key)
    return sorted_queries[:max_queries]


def verify_span_integrity(queries: list[dict], docs: dict[str, str]) -> None:
    """Verify spans are valid for all queries.

    Checks: file_path exists in docs, start < end, end <= len(doc).
    """
    for q in queries:
        for snippet in q["snippets"]:
            fp = snippet["file_path"]
            start, end = snippet["span"]
            if fp not in docs:
                raise IngestionError(
                    f"Query {q['query_id']}: file_path {fp!r} not in corpus"
                )
            doc_len = len(docs[fp])
            if start >= end:
                raise IngestionError(
                    f"Query {q['query_id']}: span [{start}, {end}) has start >= end"
                )
            if end > doc_len:
                raise IngestionError(
                    f"Query {q['query_id']}: span end {end} > doc length {doc_len} "
                    f"for {fp!r}"
                )


def build_corpus_df(
    raw_docs: dict[str, str],
    chunk_size: int,
    chunk_overlap: int,
    dataset_name: str,
) -> pd.DataFrame:
    """Chunk all documents and build a corpus DataFrame.

    Returns DataFrame with columns:
    ``doc_id, content, start_end_idx, chunk_id, checksum, dataset_name``.
    """
    rows: list[dict] = []
    chunk_counter = 0
    for doc_id, text in sorted(raw_docs.items()):
        doc_checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunks = chunk_document(text, chunk_size, chunk_overlap)

        for start, end, chunk_text in chunks:
            actual = text[start:end]
            if actual != chunk_text:
                raise IngestionError(
                    f"Offset mismatch in {doc_id} at [{start}:{end}]"
                )

            chunk_id = f"{doc_id}#chunk-{chunk_counter:05d}"
            chunk_counter += 1
            rows.append(
                {
                    "doc_id": doc_id,
                    "content": chunk_text,
                    "start_end_idx": (start, end),
                    "chunk_id": chunk_id,
                    "checksum": doc_checksum,
                    "dataset_name": dataset_name,
                }
            )

    return pd.DataFrame(rows)


def write_benchmark_json(
    queries: list[dict], benchmarks_dir: Path, dataset_name: str
) -> None:
    """Write benchmark JSON for a dataset."""
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    benchmark_json = {
        "tests": [
            {
                "query_id": q["query_id"],
                "query": q["query"],
                "snippets": q["snippets"],
            }
            for q in queries
        ]
    }
    path = benchmarks_dir / f"{dataset_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(benchmark_json, f, indent=2)


def download_hf_corpus(
    hf_repo: str,
    hf_prefix: str,
    corpus_dir: Path,
    dataset_prefix: str,
    query_file_paths: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Download corpus .txt files from a HuggingFace dataset repo.

    Uses direct URL downloads to avoid Windows MAX_PATH issues.

    Returns
    -------
    raw_docs : keyed by raw HF filename for chunking
    lookup : keyed by query file_path (handles URL encoding)
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    raw_docs: dict[str, str] = {}

    repo_files = list(
        list_repo_tree(hf_repo, repo_type="dataset", path_in_repo=hf_prefix)
    )
    txt_files = [f for f in repo_files if hasattr(f, "path") and f.path.endswith(".txt")]

    hf_raw_base = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"

    for repo_file in txt_files:
        filename = repo_file.path.removeprefix(hf_prefix + "/")
        url_path = quote(repo_file.path, safe="/")
        url = f"{hf_raw_base}/{url_path}"
        text = urlopen(url).read().decode("utf-8")  # noqa: S310

        with open(corpus_dir / filename, "w", encoding="utf-8") as f:
            f.write(text)

        doc_id = f"{dataset_prefix}/{filename}"
        raw_docs[doc_id] = text

    # Build normalized lookup for query file_path matching
    def _normalize(s: str) -> str:
        s = s.removeprefix(f"{dataset_prefix}/")
        s = unquote(s)
        s = s.replace("/", "_")
        return s.lower()

    norm_to_text: dict[str, str] = {}
    for doc_id, text in raw_docs.items():
        norm_to_text[_normalize(doc_id)] = text

    lookup: dict[str, str] = {}
    for qfp in query_file_paths:
        norm = _normalize(qfp)
        if norm in norm_to_text:
            lookup[qfp] = norm_to_text[norm]

    unmatched = query_file_paths - set(lookup.keys())
    if unmatched:
        print(f"  WARNING: {len(unmatched)} query file_paths not matched to corpus:")
        for fp in sorted(unmatched)[:5]:
            print(f"    {fp}")

    print(f"  Downloaded {len(raw_docs)} corpus files from {hf_repo}")
    return raw_docs, lookup


def download_hf_queries(
    hf_dataset: str, dataset_prefix: str
) -> list[dict]:
    """Download queries from an orgrctera HF dataset.

    Returns list of dicts with query_id, query, snippets.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split="default")
    queries: list[dict] = []
    for i, row in enumerate(ds):
        eo = json.loads(row["expected_output"])
        snippets = [
            {"file_path": s["file_path"], "span": s["span"]}
            for s in eo
        ]
        queries.append(
            {
                "query_id": f"{dataset_prefix}-{i:04d}",
                "query": row["input"],
                "snippets": snippets,
            }
        )
    print(f"  Downloaded {len(queries)} queries from {hf_dataset}")
    return queries
