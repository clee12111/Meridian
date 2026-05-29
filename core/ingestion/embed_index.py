"""Shared embed-and-upsert loop for all index-building scripts.

Parallelizes Voyage embed calls with a configurable thread pool.
Keeps the 5-retry exponential backoff per batch (the only active retry
layer — SDK retries are disabled at max_retries=0).

Env vars:
    MERIDIAN_EMBED_WORKERS  — thread pool size (default 4)
    MERIDIAN_EMBED_SLEEP    — per-batch sleep in seconds (default 0.0)
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

BATCH_SIZE = 128
_MAX_RETRIES = 5


def _embed_with_retry(
    vo: voyageai.Client,
    texts: list[str],
    model: str,
    sleep: float,
) -> tuple[list, int]:
    """Embed a single batch with 5-attempt exponential backoff.

    Returns (embeddings_list, retry_count).
    """
    retries = 0
    for attempt in range(_MAX_RETRIES):
        try:
            result = vo.embed(texts, model=model, input_type="document")
            if sleep > 0:
                time.sleep(sleep)
            return result.embeddings, retries
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                retries += 1
                wait = 2 ** attempt
                print(f"  Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Max retries ({_MAX_RETRIES}) exceeded on Voyage embed")


def embed_and_upsert(
    chunks: list[dict],
    client: QdrantClient,
    collection: str,
    model: str,
    dataset_name: str,
    text_key: str = "sac_content",
) -> dict:
    """Embed chunks and upsert into Qdrant with parallel workers.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must have ``chunk_id``, ``doc_id``, ``content``,
        and a text field (default ``sac_content``) to embed.
    client : QdrantClient
        Connected Qdrant client.
    collection : str
        Target collection name.
    model : str
        Voyage embedding model name.
    dataset_name : str
        Fallback dataset_name if not in chunk dict.
    text_key : str
        Key in chunk dict for the text to embed (default: sac_content).

    Returns
    -------
    dict with keys: total, indexed, retries, elapsed_s
    """
    workers = int(os.environ.get("MERIDIAN_EMBED_WORKERS", "4"))
    sleep = float(os.environ.get("MERIDIAN_EMBED_SLEEP", "0.0"))

    total = len(chunks)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"  Embedding {total} chunks ({total_batches} batches, "
          f"{workers} workers, sleep={sleep}s)")

    vo = voyageai.Client()
    progress_lock = threading.Lock()
    completed = [0]
    total_retries = [0]
    t0 = time.perf_counter()

    def process_batch(batch_num: int, batch: list[dict]) -> None:
        texts = [c[text_key] for c in batch]

        embeddings, retries = _embed_with_retry(vo, texts, model, sleep)

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])),
                vector=embedding,
                payload={
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "dataset_name": c.get("dataset_name", dataset_name),
                    "content": c["content"],
                },
            )
            for c, embedding in zip(batch, embeddings)
        ]
        client.upsert(collection_name=collection, points=points)

        with progress_lock:
            completed[0] += 1
            total_retries[0] += retries
            done = completed[0]
            if done % 20 == 0 or done == total_batches:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_batches - done) / rate if rate > 0 else 0
                print(f"  Batch {done}/{total_batches} "
                      f"({done * BATCH_SIZE:,}/{total:,} chunks) "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    # Build batch list
    batches = []
    for batch_num, i in enumerate(range(0, total, BATCH_SIZE)):
        batches.append((batch_num, chunks[i : i + BATCH_SIZE]))

    # Execute
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_batch, bn, b): bn
            for bn, b in batches
        }
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                raise exc

    elapsed = time.perf_counter() - t0

    # ── Completeness assertion (non-negotiable) ─────────────────────────
    info = client.get_collection(collection)
    actual = info.points_count or 0
    if actual != total:
        raise RuntimeError(
            f"COMPLETENESS FAILURE: {collection} has {actual} points, "
            f"expected {total}. Index is incomplete — do not use."
        )
    print(f"  Completeness OK: {actual}/{total} points")

    stats = {
        "total": total,
        "indexed": actual,
        "retries": total_retries[0],
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  Done in {elapsed:.1f}s ({total_retries[0]} retries)")
    return stats
