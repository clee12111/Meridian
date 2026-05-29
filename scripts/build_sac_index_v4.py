"""Build SAC index on voyage-4 (standard tier) for tier-cost comparison.

Reuses the same SAC-augmented corpus and summaries, only changes the
embedding model from voyage-4-large to voyage-4. Creates a NEW collection
contractnli_sac_v4 — never touches contractnli_sac or contractnli_baseline.

Usage:
    python scripts/build_sac_index_v4.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

SAC_CORPUS_PATH = Path("data/corpus_contractnli_sac.parquet")
COLLECTION_NAME = "contractnli_sac_v4"
EMBED_MODEL = "voyage-4"


def main() -> None:
    corpus_df = pd.read_parquet(SAC_CORPUS_PATH)
    print(f"Loaded SAC corpus: {len(corpus_df)} chunks")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Verify existing collections are intact
    for name in ["contractnli_baseline", "contractnli_sac"]:
        info = client.get_collection(name)
        print(f"  {name}: {info.points_count} points (untouched)")

    # Create new collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing {COLLECTION_NAME}")
        time.sleep(2)
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="dataset_name",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chunk_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"  Created collection {COLLECTION_NAME} (voyage-4, 1024-dim)")

    # Embed and upsert
    vo = voyageai.Client()
    BATCH_SIZE = 128
    chunks = corpus_df.to_dict("records")
    total = len(chunks)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, i in enumerate(range(0, total, BATCH_SIZE)):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["sac_content"] for c in batch]

        for attempt in range(5):
            try:
                result = vo.embed(texts, model=EMBED_MODEL, input_type="document")
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

        if (batch_num + 1) % 5 == 0 or (batch_num + 1) == total_batches:
            print(f"  Indexed {min(i + BATCH_SIZE, total)}/{total} chunks "
                  f"(batch {batch_num + 1}/{total_batches})")
        time.sleep(0.25)

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n  {COLLECTION_NAME}: {info.points_count} points (voyage-4)")

    # Confirm originals untouched
    for name in ["contractnli_baseline", "contractnli_sac"]:
        info = client.get_collection(name)
        print(f"  {name}: {info.points_count} points (still intact)")


if __name__ == "__main__":
    main()
