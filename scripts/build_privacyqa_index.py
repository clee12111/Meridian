"""Build PrivacyQA baseline index in Qdrant.

Embeds corpus_privacy_qa.parquet with voyage-4-large and indexes
into collection 'privacyqa_baseline'. No SAC prefix — baseline only.

Usage:
    python scripts/build_privacyqa_index.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd


CORPUS_PATH = Path("data/corpus_privacy_qa.parquet")
COLLECTION_NAME = "privacyqa_baseline"


def main() -> None:
    import voyageai
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PayloadSchemaType,
        PointStruct,
        VectorParams,
    )

    corpus_df = pd.read_parquet(CORPUS_PATH)
    print(f"Loaded corpus: {len(corpus_df)} chunks, "
          f"{corpus_df['doc_id'].nunique()} documents")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Verify existing collections are intact
    for coll in ["contractnli_baseline", "contractnli_sac"]:
        try:
            info = client.get_collection(coll)
            print(f"  {coll}: {info.points_count} points (intact)")
        except Exception:
            print(f"  {coll}: not found")

    # Create new collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing {COLLECTION_NAME}")
        time.sleep(2)
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,  # voyage-4-large
            distance=Distance.COSINE,
        ),
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="dataset_name",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"  Created collection {COLLECTION_NAME}")

    # Embed and upsert
    vo = voyageai.Client()
    BATCH_SIZE = 128
    chunks = corpus_df.to_dict("records")
    total = len(chunks)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, i in enumerate(range(0, total, BATCH_SIZE)):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        for attempt in range(5):
            try:
                result = vo.embed(
                    texts,
                    model="voyage-4-large",
                    input_type="document",
                )
                break
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    wait = 2 ** attempt
                    print(f"  Rate limit hit, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError("Max retries exceeded on Voyage embed")

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])),
                vector=embedding,
                payload={
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "dataset_name": c["dataset_name"],
                    "content": c["content"],
                },
            )
            for c, embedding in zip(batch, result.embeddings)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Indexed {min(i + BATCH_SIZE, total)}/{total} chunks "
              f"(batch {batch_num + 1}/{total_batches})")
        time.sleep(0.25)

    # Write fingerprint
    from core.evaluation.fingerprint import write_fingerprint

    write_fingerprint(
        Path("data"),
        collection_name=COLLECTION_NAME,
        target_dataset="privacy_qa",
        chunk_size=512,
        chunk_overlap=128,
    )

    # Verify
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"\n  PrivacyQA collection ready: {collection_info.points_count} points")
    print(f"  Expected: {total} points")

    # Confirm other collections intact
    for coll in ["contractnli_baseline", "contractnli_sac"]:
        try:
            info = client.get_collection(coll)
            print(f"  {coll}: {info.points_count} points (OK)")
        except Exception:
            print(f"  {coll}: MISSING!")


if __name__ == "__main__":
    main()
