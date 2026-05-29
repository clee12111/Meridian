"""Build SAC indexes for all corpora on voyage-4.

Generates SAC summaries (if not cached), builds SAC-augmented parquets,
embeds with voyage-4, indexes into Qdrant, and builds routing indexes.

Usage:
    python scripts/build_corpus_v4.py privacyqa
    python scripts/build_corpus_v4.py cuad
    python scripts/build_corpus_v4.py maud
    python scripts/build_corpus_v4.py all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)

from core.ingestion.embed_index import embed_and_upsert

EMBED_MODEL = "voyage-4"

CORPUS_CONFIG = {
    "contractnli": {
        "parquet": Path("data/corpus_contractnli.parquet"),
        "sac_parquet": Path("data/corpus_contractnli_sac.parquet"),
        "summaries": Path("data/sac_summaries.json"),
        "collection": "contractnli_sac_v4",
        "routing_index": Path("data/routing_index_v4.npz"),
        "dataset_name": "contractnli",
        "doc_type": "NDA",
    },
    "privacyqa": {
        "parquet": Path("data/corpus_privacy_qa.parquet"),
        "sac_parquet": Path("data/corpus_privacy_qa_sac.parquet"),
        "summaries": Path("data/sac_summaries_privacyqa.json"),
        "collection": "privacyqa_sac_v4",
        "routing_index": Path("data/routing_index_privacyqa_v4.npz"),
        "dataset_name": "privacy_qa",
        "doc_type": "privacy policy",
    },
    "cuad": {
        "parquet": Path("data/corpus_cuad.parquet"),
        "sac_parquet": Path("data/corpus_cuad_sac.parquet"),
        "summaries": Path("data/sac_summaries_cuad.json"),
        "collection": "cuad_sac_v4",
        "routing_index": Path("data/routing_index_cuad_v4.npz"),
        "dataset_name": "cuad",
        "doc_type": "commercial contract",
    },
    "maud": {
        "parquet": Path("data/corpus_maud.parquet"),
        "sac_parquet": Path("data/corpus_maud_sac.parquet"),
        "summaries": Path("data/sac_summaries_maud.json"),
        "collection": "maud_sac_v4",
        "routing_index": Path("data/routing_index_maud_v4.npz"),
        "dataset_name": "maud",
        "doc_type": "merger agreement",
    },
}

SUMMARY_PROMPT_TEMPLATE = """\
You are a legal document analyzer. Given the opening sections \
of a {doc_type}, extract a concise identification \
summary in exactly this format:

Parties: [Party A] and [Party B]
Date: [date or "undated" if not found]
Type: [{doc_type}]
Subject: [brief description of what the document covers]

Return ONLY this 4-line summary. No explanation, no preamble."""


# ── Phase 1: SAC Summaries ─────────────────────────────────────────────────


def generate_summaries(corpus_df: pd.DataFrame, config: dict) -> dict[str, str]:
    """Generate SAC summaries for each document."""
    summaries_path = config["summaries"]
    if summaries_path.exists():
        summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
        print(f"  Loaded {len(summaries)} cached summaries from {summaries_path}")
        return summaries

    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    doc_ids = sorted(corpus_df["doc_id"].unique())
    total = len(doc_ids)
    print(f"  Generating summaries for {total} documents...")

    system_prompt = SUMMARY_PROMPT_TEMPLATE.format(doc_type=config["doc_type"])
    summaries: dict[str, str] = {}
    lock = threading.Lock()

    def summarize_one(doc_id: str) -> None:
        chunks = corpus_df[corpus_df["doc_id"] == doc_id]
        chunks = chunks.sort_values("chunk_id").head(3)
        sample = "\n\n".join(chunks["content"].tolist())

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Document opening:\n{sample[:4000]}"},
                    ],
                    max_tokens=150,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                summary = (response.choices[0].message.content or "").strip()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    summary = f"(summary generation failed: {e})"

        with lock:
            summaries[doc_id] = summary
            n = len(summaries)
            if n % 20 == 0 or n == total:
                print(f"    [{n}/{total}] summaries generated")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(summarize_one, did): did for did in doc_ids}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                print(f"  ERROR: {futures[future]} - {exc}", file=sys.stderr)

    summaries_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved {len(summaries)} summaries to {summaries_path}")
    return summaries


# ── Phase 2: SAC-augmented corpus ──────────────────────────────────────────


def build_sac_corpus(corpus_df: pd.DataFrame, summaries: dict[str, str],
                     config: dict) -> pd.DataFrame:
    """Prepend document summary to each chunk's content for embedding."""
    sac_path = config["sac_parquet"]
    if sac_path.exists():
        df = pd.read_parquet(sac_path)
        if "sac_content" in df.columns:
            print(f"  Loaded cached SAC corpus from {sac_path}")
            return df

    corpus_df = corpus_df.copy()
    corpus_df["sac_content"] = corpus_df.apply(
        lambda row: (
            f"[Document: {summaries.get(row['doc_id'], '')}]\n\n{row['content']}"
            if summaries.get(row["doc_id"])
            else row["content"]
        ),
        axis=1,
    )
    corpus_df.to_parquet(sac_path, index=False)
    print(f"  Saved SAC corpus ({len(corpus_df)} chunks) to {sac_path}")
    return corpus_df


# ── Lockfile guard ────────────────────────────────────────────────────────

LOCK_DIR = Path("data")


def _lock_path(collection: str) -> Path:
    return LOCK_DIR / f".indexing_lock_{collection}"


def _acquire_lock(collection: str) -> None:
    lp = _lock_path(collection)
    if lp.exists():
        pid = lp.read_text().strip()
        print(f"  ERROR: Lock file {lp} exists (PID {pid}).", file=sys.stderr)
        print(f"  Another indexing job may be active. Delete the lock "
              f"manually if the prior job crashed.", file=sys.stderr)
        sys.exit(1)
    lp.write_text(str(os.getpid()))


def _release_lock(collection: str) -> None:
    lp = _lock_path(collection)
    if lp.exists():
        lp.unlink()


# ── Phase 3: Embed and index (resumable) ─────────────────────────────────


def _get_existing_chunk_ids(client: QdrantClient, collection: str) -> set[str]:
    """Scroll the collection and collect all chunk_ids already present."""
    existing: set[str] = set()
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        for point in results:
            cid = point.payload.get("chunk_id")
            if cid:
                existing.add(cid)
        if offset is None:
            break
    return existing


def embed_and_index_corpus(corpus_df: pd.DataFrame, config: dict) -> None:
    """Embed SAC content with voyage-4 and upsert into Qdrant.

    Resumable: if the collection exists with partial data, only
    embeds and upserts the missing chunks.
    """
    collection = config["collection"]

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    total_chunks = len(corpus_df)
    collection_exists = False
    existing_count = 0

    try:
        info = client.get_collection(collection)
        collection_exists = True
        existing_count = info.points_count or 0
        if existing_count == total_chunks:
            print(f"  Collection {collection} already complete: "
                  f"{existing_count} points, skipping")
            return
        print(f"  Collection {collection} exists with {existing_count}/{total_chunks} "
              f"points — resuming")
    except Exception:
        pass

    _acquire_lock(collection)
    try:
        if not collection_exists:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            client.create_payload_index(
                collection_name=collection,
                field_name="dataset_name",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=collection,
                field_name="chunk_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"  Created collection {collection}")

        # Find which chunks are already indexed
        if existing_count > 0:
            print(f"  Scanning existing chunk_ids...")
            existing_ids = _get_existing_chunk_ids(client, collection)
            print(f"  Found {len(existing_ids)} existing chunks")
            pending_df = corpus_df[~corpus_df["chunk_id"].isin(existing_ids)]
        else:
            pending_df = corpus_df

        pending_count = len(pending_df)
        if pending_count == 0:
            print(f"  All chunks already indexed")
            _release_lock(collection)
            return

        chunks = pending_df.to_dict("records")
        embed_and_upsert(
            chunks=chunks,
            client=client,
            collection=collection,
            model=EMBED_MODEL,
            dataset_name=config["dataset_name"],
        )
    finally:
        _release_lock(collection)


# ── Phase 4: Routing index ────────────────────────────────────────────────


def build_routing(summaries: dict[str, str], config: dict) -> None:
    """Build hybrid routing index (dense + BM25) for this corpus."""
    from core.retrieval.routing import build_routing_index

    output_path = config["routing_index"]
    if output_path.exists():
        data = np.load(output_path, allow_pickle=True)
        print(f"  Routing index already exists: {len(data['doc_ids'])} docs at {output_path}")
        return

    # Write a temp summaries file for build_routing_index
    tmp_summaries = config["summaries"]
    n = build_routing_index(
        summaries_path=tmp_summaries,
        output_path=output_path,
        model=EMBED_MODEL,
    )
    print(f"  Built routing index: {n} docs -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────


def build_one(corpus_name: str) -> None:
    """Build all SAC artifacts for one corpus."""
    config = CORPUS_CONFIG[corpus_name]
    print(f"\n{'='*60}")
    print(f"Building {corpus_name} on {EMBED_MODEL}")
    print(f"{'='*60}")

    corpus_df = pd.read_parquet(config["parquet"])
    n_docs = corpus_df["doc_id"].nunique()
    print(f"  Corpus: {len(corpus_df)} chunks, {n_docs} docs")

    # Phase 1: summaries
    summaries = generate_summaries(corpus_df, config)

    # Phase 2: SAC-augmented corpus
    corpus_df = build_sac_corpus(corpus_df, summaries, config)

    # Phase 3: embed and index
    embed_and_index_corpus(corpus_df, config)

    # Phase 4: routing index
    build_routing(summaries, config)

    print(f"  {corpus_name} COMPLETE")


def main() -> None:
    p = argparse.ArgumentParser(description="Build SAC indexes on voyage-4")
    p.add_argument("corpora", nargs="+",
                   choices=list(CORPUS_CONFIG.keys()) + ["all"],
                   help="Which corpora to build")
    args = p.parse_args()

    targets = list(CORPUS_CONFIG.keys()) if "all" in args.corpora else args.corpora

    for corpus in targets:
        build_one(corpus)

    # Final verification
    print(f"\n{'='*60}")
    print("FINAL COLLECTION STATUS")
    print(f"{'='*60}")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    for name in sorted(col.name for col in client.get_collections().collections):
        try:
            info = client.get_collection(name)
            print(f"  {name}: {info.points_count} points")
        except Exception:
            pass


if __name__ == "__main__":
    main()
