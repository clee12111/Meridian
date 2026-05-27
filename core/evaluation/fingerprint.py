"""Index fingerprint: track which (dataset, chunk_size, chunk_overlap) is
live in each Qdrant collection.

Written by evaluate_config() immediately after qdrant.index() completes.
Read by any caller that wants to assert the live index matches a family's
index-time params before trusting skip_index=True results.

Format: data/index_fingerprint.json
  { "<collection_name>": { "target_dataset": ..., "chunk_size": ...,
                            "chunk_overlap": ..., "indexed_at": "<iso>" } }

One file, multiple keys — supports both single-dataset and multi-dataset
collections in the same repo.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_FILENAME = "index_fingerprint.json"


def write_fingerprint(
    data_dir: Path,
    collection_name: str,
    target_dataset: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Record that *collection_name* now holds chunks for the given params."""
    path = data_dir / _FILENAME
    existing: dict = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing[collection_name] = {
        "target_dataset": target_dataset,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def read_fingerprint(
    data_dir: Path,
    collection_name: str,
) -> dict | None:
    """Return the stored fingerprint for *collection_name*, or None if absent."""
    path = data_dir / _FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(collection_name)


def fingerprint_matches(
    data_dir: Path,
    collection_name: str,
    target_dataset: str,
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    """Return True iff the live fingerprint matches the given index-time params.

    Returns False (not raises) when no fingerprint exists — the caller
    decides whether that is an error.
    """
    fp = read_fingerprint(data_dir, collection_name)
    if fp is None:
        return False
    return (
        fp["target_dataset"] == target_dataset
        and fp["chunk_size"] == chunk_size
        and fp["chunk_overlap"] == chunk_overlap
    )
