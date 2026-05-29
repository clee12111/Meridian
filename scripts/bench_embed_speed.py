"""Benchmark embed speed: old (sequential+sleep) vs new (parallel, no sleep).

Uses first 2000 chunks from the section CUAD SAC parquet into temporary
collections that are deleted after each run. Costs ~2M tokens total.

Usage:
    python scripts/bench_embed_speed.py
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
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from core.ingestion.embed_index import embed_and_upsert

PARQUET = Path("data/corpus_cuad_section_sac.parquet")
N_CHUNKS = 2000
EMBED_MODEL = "voyage-4"


def setup_collection(client: QdrantClient, name: str) -> None:
    try:
        client.delete_collection(name)
        time.sleep(1)
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=name,
        field_name="dataset_name",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=name,
        field_name="chunk_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def cleanup(client: QdrantClient, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        pass


def run_bench(chunks: list[dict], client: QdrantClient,
              collection: str, workers: int, sleep: float) -> dict:
    os.environ["MERIDIAN_EMBED_WORKERS"] = str(workers)
    os.environ["MERIDIAN_EMBED_SLEEP"] = str(sleep)

    setup_collection(client, collection)

    t0 = time.perf_counter()
    stats = embed_and_upsert(
        chunks=chunks,
        client=client,
        collection=collection,
        model=EMBED_MODEL,
        dataset_name="cuad",
    )
    elapsed = time.perf_counter() - t0

    # Spot-check 5 payloads
    mismatches = 0
    check_ids = [chunks[i]["chunk_id"] for i in range(0, min(len(chunks), 5))]
    for cid in check_ids:
        results, _ = client.scroll(
            collection_name=collection, limit=1,
            scroll_filter={"must": [{"key": "chunk_id", "match": {"value": cid}}]},
            with_payload=True, with_vectors=True,
        )
        if not results:
            print(f"  MISSING point for {cid}")
            mismatches += 1
        else:
            pt = results[0]
            orig = next(c for c in chunks if c["chunk_id"] == cid)
            if pt.payload["content"] != orig["content"]:
                print(f"  PAYLOAD MISMATCH for {cid}")
                mismatches += 1
            if pt.vector is None or len(pt.vector) != 1024:
                print(f"  VECTOR MISSING/WRONG for {cid}")
                mismatches += 1

    stats["wall_s"] = round(elapsed, 1)
    stats["payload_checks"] = len(check_ids)
    stats["payload_mismatches"] = mismatches

    cleanup(client, collection)
    return stats


def main() -> None:
    df = pd.read_parquet(PARQUET)
    chunks = df.head(N_CHUNKS).to_dict("records")
    print(f"Benchmarking with {len(chunks)} chunks\n")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Run A: old behavior (sequential, sleep=0.25)
    print("=" * 60)
    print("RUN A: workers=1, sleep=0.25 (old sequential behavior)")
    print("=" * 60)
    stats_a = run_bench(chunks, client, "_bench_old", workers=1, sleep=0.25)
    print(f"\n  Result: {stats_a}")

    # Brief pause between runs
    time.sleep(2)

    # Run B: new behavior (parallel, no sleep)
    print("\n" + "=" * 60)
    print("RUN B: workers=4, sleep=0.0 (new parallel behavior)")
    print("=" * 60)
    stats_b = run_bench(chunks, client, "_bench_new", workers=4, sleep=0.0)
    print(f"\n  Result: {stats_b}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'Old (1w/0.25s)':<18} {'New (4w/0.0s)':<18}")
    print(f"  {'Wall time':<25} {stats_a['wall_s']:>12.1f}s {stats_b['wall_s']:>12.1f}s")
    print(f"  {'429 retries':<25} {stats_a['retries']:>12} {stats_b['retries']:>12}")
    print(f"  {'Points indexed':<25} {stats_a['indexed']:>12} {stats_b['indexed']:>12}")
    print(f"  {'Payload mismatches':<25} {stats_a['payload_mismatches']:>12} {stats_b['payload_mismatches']:>12}")
    speedup = stats_a['wall_s'] / stats_b['wall_s'] if stats_b['wall_s'] > 0 else 0
    print(f"  {'Speedup':<25} {'':>12} {speedup:>11.1f}x")
    completeness = "PASS" if stats_a['indexed'] == N_CHUNKS and stats_b['indexed'] == N_CHUNKS else "FAIL"
    print(f"  {'Completeness gate':<25} {completeness:>30}")


if __name__ == "__main__":
    main()
