"""Build SAC index on voyage-4 (standard tier) for tier-cost comparison.

Reuses the same SAC-augmented corpus and summaries, only changes the
embedding model from voyage-4-large to voyage-4. Creates a NEW collection
contractnli_sac_v4 — never touches contractnli_sac or contractnli_baseline.

Usage:
    python scripts/build_sac_index_v4.py
"""

from __future__ import annotations

import os
import sys
import time
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

    chunks = corpus_df.to_dict("records")
    embed_and_upsert(
        chunks=chunks,
        client=client,
        collection=COLLECTION_NAME,
        model=EMBED_MODEL,
        dataset_name="contractnli",
    )

    # Confirm originals untouched
    for name in ["contractnli_baseline", "contractnli_sac"]:
        info = client.get_collection(name)
        print(f"  {name}: {info.points_count} points (still intact)")


if __name__ == "__main__":
    main()
