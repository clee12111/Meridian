"""Embed section-aware CUAD SAC parquet and index into a NEW Qdrant collection.

Does NOT touch the baseline collection (cuad_sac_v4). Reuses cached SAC
summaries. Writes a fingerprint with chunk_strategy metadata.

Usage:
    python scripts/build_section_index.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)

from core.ingestion.embed_index import embed_and_upsert

COLLECTION = "cuad_section_sac_v4"
SAC_PARQUET = Path("data/corpus_cuad_section_sac.parquet")
EMBED_MODEL = "voyage-4"


def main() -> None:
    corpus_df = pd.read_parquet(SAC_PARQUET)
    total_chunks = len(corpus_df)
    print(f"Loaded {total_chunks} chunks from {SAC_PARQUET}")
    print(f"  Docs: {corpus_df['doc_id'].nunique()}")
    print(f"  Collection: {COLLECTION}")
    print(f"  Model: {EMBED_MODEL}")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Verify baseline is untouched
    baseline_info = client.get_collection("cuad_sac_v4")
    baseline_count = baseline_info.points_count
    print(f"  Baseline cuad_sac_v4 intact: {baseline_count} points")

    # Check if collection already exists (resumable)
    collection_exists = False
    existing_count = 0
    try:
        info = client.get_collection(COLLECTION)
        collection_exists = True
        existing_count = info.points_count or 0
        if existing_count == total_chunks:
            print(f"  Collection already complete: {existing_count} points, skipping")
            return
        print(f"  Exists with {existing_count}/{total_chunks} — resuming")
    except Exception:
        pass

    if not collection_exists:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="dataset_name",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="chunk_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"  Created collection {COLLECTION}")

    # Find already-indexed chunks for resume
    if existing_count > 0:
        print(f"  Scanning existing chunk_ids...")
        existing_ids: set[str] = set()
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=COLLECTION, limit=1000,
                offset=offset, with_payload=["chunk_id"], with_vectors=False,
            )
            for pt in results:
                cid = pt.payload.get("chunk_id")
                if cid:
                    existing_ids.add(cid)
            if offset is None:
                break
        print(f"  Found {len(existing_ids)} existing chunks")
        pending_df = corpus_df[~corpus_df["chunk_id"].isin(existing_ids)]
    else:
        pending_df = corpus_df

    pending_count = len(pending_df)
    if pending_count == 0:
        print("  All chunks already indexed")
        return

    chunks = pending_df.to_dict("records")
    stats = embed_and_upsert(
        chunks=chunks,
        client=client,
        collection=COLLECTION,
        model=EMBED_MODEL,
        dataset_name="cuad",
    )

    # Confirm baseline untouched
    baseline_after = client.get_collection("cuad_sac_v4")
    print(f"  Baseline cuad_sac_v4 still intact: {baseline_after.points_count} points "
          f"({'OK' if baseline_after.points_count == baseline_count else 'MISMATCH!'})")

    # Write fingerprint with chunk_strategy tag
    from core.evaluation.fingerprint import write_fingerprint
    write_fingerprint(
        Path("data"),
        collection_name=COLLECTION,
        target_dataset="cuad",
        chunk_size=512,
        chunk_overlap=0,
    )
    fp_path = Path("data/index_fingerprint.json")
    fp_data = json.loads(fp_path.read_text(encoding="utf-8"))
    fp_data[COLLECTION]["chunk_strategy"] = "section"
    fp_path.write_text(json.dumps(fp_data, indent=2), encoding="utf-8")
    print(f"  Fingerprint written with chunk_strategy='section'")


if __name__ == "__main__":
    main()
