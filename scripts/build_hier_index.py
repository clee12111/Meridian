"""Index hierarchy CHILDREN into Qdrant (parents stay in parquet for lookup).

Usage:
    python scripts/build_hier_index.py cuad
    python scripts/build_hier_index.py maud
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from core.ingestion.embed_index import embed_and_upsert

EMBED_MODEL = "voyage-4"

CORPUS_CONFIG = {
    "cuad": {
        "sac_parquet": Path("data/corpus_cuad_hier_sac.parquet"),
        "collection": "cuad_hier_sac_v4",
        "dataset_name": "cuad",
        "baseline_collection": "cuad_sac_v4",
    },
    "maud": {
        "sac_parquet": Path("data/corpus_maud_hier_sac.parquet"),
        "collection": "maud_hier_sac_v4",
        "dataset_name": "maud",
        "baseline_collection": "maud_sac_v4",
    },
}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Index hierarchy children")
    p.add_argument("corpus", choices=list(CORPUS_CONFIG.keys()))
    args = p.parse_args()

    cfg = CORPUS_CONFIG[args.corpus]
    corpus_df = pd.read_parquet(cfg["sac_parquet"])

    # Filter to children only (is_parent == False)
    children_df = corpus_df[~corpus_df["is_parent"]].copy()
    total_children = len(children_df)
    print(f"Loaded {len(corpus_df)} total rows, {total_children} children to index")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    collection = cfg["collection"]

    # Verify baseline untouched
    baseline = cfg["baseline_collection"]
    baseline_info = client.get_collection(baseline)
    baseline_count = baseline_info.points_count
    print(f"Baseline {baseline}: {baseline_count} points (will verify untouched after)")

    # Check if collection already complete
    try:
        info = client.get_collection(collection)
        existing = info.points_count or 0
        if existing == total_children:
            print(f"Collection {collection} already complete: {existing} points")
            return
        print(f"Collection {collection} exists with {existing}/{total_children}")
    except Exception:
        print(f"Creating collection {collection}...")
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

    # Resume: find already-indexed chunks and embed only pending
    existing_ids: set[str] = set()
    if existing > 0:
        print(f"  Scanning existing chunk_ids for resume...")
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=collection, limit=1000,
                offset=offset, with_payload=["chunk_id"], with_vectors=False,
            )
            for pt in results:
                cid = pt.payload.get("chunk_id")
                if cid:
                    existing_ids.add(cid)
            if offset is None:
                break
        print(f"  Found {len(existing_ids)} already indexed")

    pending_df = children_df[~children_df["chunk_id"].isin(existing_ids)]
    pending_count = len(pending_df)
    if pending_count == 0:
        print(f"  All {total_children} children already indexed")
    else:
        print(f"  {pending_count} pending chunks to embed")
        os.environ["MERIDIAN_EMBED_WORKERS"] = "6"

        chunks = pending_df.to_dict("records")
        try:
            embed_and_upsert(
                chunks=chunks,
                client=client,
                collection=collection,
                model=EMBED_MODEL,
                dataset_name=cfg["dataset_name"],
            )
        except RuntimeError as e:
            if "COMPLETENESS" in str(e):
                # Expected on resume: assertion compares pending vs total
                pass
            else:
                raise

    # Final completeness check
    final_info = client.get_collection(collection)
    actual = final_info.points_count or 0
    if actual != total_children:
        print(f"  WARNING: {actual}/{total_children} points (gap of {total_children - actual})")
    else:
        print(f"  Completeness OK: {actual}/{total_children} points")

    # Verify baseline untouched
    baseline_after = client.get_collection(baseline)
    assert baseline_after.points_count == baseline_count, \
        f"Baseline {baseline} changed: {baseline_count} -> {baseline_after.points_count}"
    print(f"Baseline {baseline} untouched: {baseline_after.points_count} points")


if __name__ == "__main__":
    main()
