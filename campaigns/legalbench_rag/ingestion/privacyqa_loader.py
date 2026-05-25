"""PrivacyQA ingestion from raw PrivacyQA_EMNLP data.

Downloads the raw CSV from GitHub and generates queries with character
spans. No LLM calls required — purely algorithmic.

Ported from legalbenchrag/generate/generate_privacy_qa.py.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from campaigns.legalbench_rag.ingestion.common import (
    IngestionError,
    build_corpus_df,
    sample_mini,
    verify_span_integrity,
    write_benchmark_json,
)

PRIVACYQA_URL = (
    "https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP/"
    "archive/refs/heads/master.zip"
)
DATASET_NAME = "privacy_qa"


def _sort_and_merge_spans(
    spans: list[tuple[int, int]], max_bridge_gap: int = 2
) -> list[tuple[int, int]]:
    """Merge adjacent/overlapping spans."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start <= merged[-1][1] + max_bridge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _download_and_parse(
    data_dir: Path,
) -> tuple[dict[str, str], list[dict]]:
    """Download PrivacyQA CSV, build documents and queries."""
    raw_dir = data_dir / "raw_data" / "privacy_qa"
    check_path = raw_dir / "PrivacyQA_EMNLP-master"

    if not check_path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading PrivacyQA from GitHub...")
        resp = urlopen(PRIVACYQA_URL)  # noqa: S310
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            zf.extractall(raw_dir)
        print(f"  Extracted to {check_path}")

    csv_path = check_path / "data" / "policy_test_data.csv"
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    # Build per-document sentence maps
    doc_sent_map: dict[str, dict[int, str]] = {}
    query_map: dict[tuple[str, int], str] = {}
    annotation_map: dict[tuple[str, int], list[dict]] = {}

    for row in rows:
        doc_id = row["DocID"].split("_")[0].strip()
        query_id = int(row["QueryID"].split("_")[-1])
        sent_id = int(row["SentID"].split("_")[-1])

        if doc_id not in doc_sent_map:
            doc_sent_map[doc_id] = {}
        doc_sent_map[doc_id][sent_id] = row["Segment"]
        query_map[(doc_id, query_id)] = row["Query"].strip()

        # Count annotator votes
        relevants = 0
        irrelevants = 0
        no_responses = 0
        for i in range(1, 7):
            ann = str(row.get(f"Ann{i}", "nan")).lower()
            if "irrelevant" in ann:
                irrelevants += 1
            elif "relevant" in ann:
                relevants += 1
            else:
                no_responses += 1

        key = (doc_id, query_id)
        if key not in annotation_map:
            annotation_map[key] = []
        annotation_map[key].append(
            {
                "sent_id": sent_id,
                "relevants": relevants,
                "irrelevants": irrelevants,
                "no_responses": no_responses,
            }
        )

    # Build document texts with span tracking
    doc_texts: dict[str, str] = {}
    doc_sent_spans: dict[str, dict[int, tuple[int, int]]] = {}

    for doc_id, sent_map in doc_sent_map.items():
        doc_sent_spans[doc_id] = {}
        total_text = ""
        for sent_id, sentence in sorted(sent_map.items()):
            sentence += "\n"
            doc_sent_spans[doc_id][sent_id] = (
                len(total_text),
                len(total_text) + len(sentence),
            )
            total_text += sentence
        doc_texts[doc_id] = total_text

    # Build queries
    queries: list[dict] = []
    used_doc_ids: set[str] = set()
    counter = 0

    for (doc_id, query_id), annotations in annotation_map.items():
        # Skip if too many non-responses
        if any(a["no_responses"] > 2 for a in annotations):
            continue

        scored = [
            (
                a["sent_id"],
                a["relevants"] / (a["relevants"] + a["irrelevants"])
                if (a["relevants"] + a["irrelevants"]) > 0
                else 0.0,
            )
            for a in annotations
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        spans = []
        for sent_id, score in scored:
            if score >= 0.5:
                span = doc_sent_spans[doc_id][sent_id]
                spans.append(span)

        spans = _sort_and_merge_spans(spans)
        if not spans:
            continue

        used_doc_ids.add(doc_id)
        question = query_map[(doc_id, query_id)]
        queries.append(
            {
                "query_id": f"privacy_qa-{counter:04d}",
                "query": f'Consider "{doc_id}"\'s privacy policy; {question}',
                "snippets": [
                    {"file_path": f"privacy_qa/{doc_id}.txt", "span": list(s)}
                    for s in spans
                ],
            }
        )
        counter += 1

    # Write corpus files
    corpus_dir = data_dir / "corpus" / DATASET_NAME
    corpus_dir.mkdir(parents=True, exist_ok=True)
    raw_docs: dict[str, str] = {}
    for doc_id in used_doc_ids:
        text = doc_texts[doc_id]
        with open(corpus_dir / f"{doc_id}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        raw_docs[f"privacy_qa/{doc_id}.txt"] = text

    return raw_docs, queries


def ingest_privacyqa(
    data_dir: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    max_queries: int | None = 194,
) -> tuple[pd.DataFrame, list[dict]]:
    """Full ingestion pipeline for PrivacyQA."""
    data_dir = Path(data_dir)

    raw_docs, queries = _download_and_parse(data_dir)

    # Build lookup for span verification (file_paths match doc_ids directly)
    lookup = dict(raw_docs)
    verify_span_integrity(queries, lookup)
    print(f"  Span integrity verified for all {len(queries)} queries")

    if max_queries is not None:
        queries = sample_mini(queries, max_queries)

    write_benchmark_json(queries, data_dir / "benchmarks", DATASET_NAME)

    corpus_df = build_corpus_df(raw_docs, chunk_size, chunk_overlap, DATASET_NAME)
    corpus_df.to_parquet(data_dir / "corpus_privacy_qa.parquet", index=False)
    print(
        f"Ingested {len(raw_docs)} documents, "
        f"{len(corpus_df)} chunks, {len(queries)} queries"
    )

    return corpus_df, queries
