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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd


CORPUS_PATH = Path("data/corpus_privacy_qa.parquet")
COLLECTION_NAME = "privacyqa_baseline"


def main() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PayloadSchemaType,
        VectorParams,
    )
    from core.ingestion.embed_index import embed_and_upsert
    from core.evaluation.fingerprint import write_fingerprint

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

    chunks = corpus_df.to_dict("records")
    embed_and_upsert(
        chunks=chunks,
        client=client,
        collection=COLLECTION_NAME,
        model="voyage-4-large",
        dataset_name="privacy_qa",
        text_key="content",  # No SAC for this baseline
    )

    write_fingerprint(
        Path("data"),
        collection_name=COLLECTION_NAME,
        target_dataset="privacy_qa",
        chunk_size=512,
        chunk_overlap=128,
    )

    # Confirm other collections intact
    for coll in ["contractnli_baseline", "contractnli_sac"]:
        try:
            info = client.get_collection(coll)
            print(f"  {coll}: {info.points_count} points (OK)")
        except Exception:
            print(f"  {coll}: MISSING!")


if __name__ == "__main__":
    main()
