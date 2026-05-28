"""Build SAC (Summary-Augmented Chunking) index for ContractNLI.

Generates document-level summaries, prepends them to each chunk's
embedding input, and indexes into a new Qdrant collection
(contractnli_sac). The baseline collection is never touched.

Reference: Reuter et al., arXiv:2510.06999

Usage:
    python scripts/build_sac_index.py
"""

from __future__ import annotations

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

import pandas as pd

CORPUS_PATH = Path("data/corpus_contractnli.parquet")
SUMMARIES_PATH = Path("data/sac_summaries.json")
SAC_CORPUS_PATH = Path("data/corpus_contractnli_sac.parquet")
COLLECTION_NAME = "contractnli_sac"

SUMMARY_SYSTEM_PROMPT = """\
You are a legal document analyzer. Given the opening sections \
of a Non-Disclosure Agreement, extract a concise identification \
summary in exactly this format:

Parties: [Party A] and [Party B]
Date: [date or "undated" if not found]
Type: [mutual NDA / one-way NDA / NDCSC / other]
Subject: [brief description of what the NDA covers]

Return ONLY this 4-line summary. No explanation, no preamble."""


# ── Phase 1: Generate document summaries ─────────────────────────────────


def get_doc_sample(doc_id: str, corpus_df: pd.DataFrame,
                   n_chunks: int = 3) -> str:
    """Concatenate first n chunks of a document as context for summarization."""
    chunks = corpus_df[corpus_df["doc_id"] == doc_id]
    chunks = chunks.sort_values("chunk_id").head(n_chunks)
    return "\n\n".join(chunks["content"].tolist())


def generate_summary(doc_id: str, doc_sample: str, api_key: str) -> str:
    """Generate a 4-line identification summary via DeepSeek-flash."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"NDA opening sections:\n{doc_sample}"},
        ],
        max_tokens=150,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (response.choices[0].message.content or "").strip()


def generate_all_summaries(corpus_df: pd.DataFrame) -> dict[str, str]:
    """Generate summaries for all documents, 8 concurrent threads."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    doc_ids = corpus_df["doc_id"].unique().tolist()
    total = len(doc_ids)
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generating summaries for {total} documents")
    print(f"{'='*60}")

    summaries: dict[str, str] = {}
    lock = threading.Lock()

    def summarize_one(doc_id: str) -> None:
        sample = get_doc_sample(doc_id, corpus_df)
        summary = generate_summary(doc_id, sample, api_key)
        with lock:
            summaries[doc_id] = summary
            n = len(summaries)
            first_line = summary.split("\n")[0] if summary else "(empty)"
            print(f"  [{n}/{total}] {doc_id} — {first_line}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(summarize_one, did): did for did in doc_ids}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                doc_id = futures[future]
                print(f"  ERROR: {doc_id} — {exc}", file=sys.stderr)

    # Save summaries
    SUMMARIES_PATH.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(summaries)} summaries to {SUMMARIES_PATH}")
    return summaries


# ── Phase 2: Build SAC-augmented corpus ──────────────────────────────────


def build_sac_corpus(corpus_df: pd.DataFrame,
                     summaries: dict[str, str]) -> pd.DataFrame:
    """Prepend document summary to each chunk's content for embedding."""
    print(f"\n{'='*60}")
    print(f"PHASE 2: Building SAC-augmented corpus")
    print(f"{'='*60}")

    def build_sac_content(row: pd.Series) -> str:
        summary = summaries.get(row["doc_id"], "")
        if summary:
            return f"[Document: {summary}]\n\n{row['content']}"
        return row["content"]

    corpus_df = corpus_df.copy()
    corpus_df["sac_content"] = corpus_df.apply(build_sac_content, axis=1)

    corpus_df.to_parquet(SAC_CORPUS_PATH, index=False)
    print(f"  Saved SAC corpus ({len(corpus_df)} chunks) to {SAC_CORPUS_PATH}")

    # Show 2 examples
    for i, idx in enumerate(corpus_df.index[:2]):
        row = corpus_df.iloc[idx]
        print(f"\n  --- Example {i+1}: {row['chunk_id']} ---")
        preview = row["sac_content"][:300]
        print(f"  {preview}...")

    return corpus_df


# ── Phase 3: Embed and index into Qdrant ─────────────────────────────────


def embed_and_index(corpus_df: pd.DataFrame) -> None:
    """Embed SAC content with voyage-4-large and upsert into Qdrant."""
    import voyageai
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PayloadSchemaType,
        PointStruct,
        VectorParams,
    )

    print(f"\n{'='*60}")
    print(f"PHASE 3: Embedding and indexing into Qdrant")
    print(f"{'='*60}")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Verify baseline is intact before we start
    baseline_info = client.get_collection("contractnli_baseline")
    baseline_points = baseline_info.points_count
    print(f"  Baseline collection intact: {baseline_points} points")

    # Create new collection (delete + wait if exists)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing {COLLECTION_NAME}")
        time.sleep(2)  # allow Qdrant to finalize deletion
    except Exception:
        pass  # collection didn't exist

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,  # voyage-4-large dimension
            distance=Distance.COSINE,
        ),
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="dataset_name",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"  Created collection {COLLECTION_NAME}")

    # Embed and upsert in batches
    vo = voyageai.Client()
    BATCH_SIZE = 128
    chunks = corpus_df.to_dict("records")
    total = len(chunks)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, i in enumerate(range(0, total, BATCH_SIZE)):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["sac_content"] for c in batch]

        # Embed with retry
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
                    "content": c["content"],  # original, not SAC
                },
            )
            for c, embedding in zip(batch, result.embeddings)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)

        if (batch_num + 1) % 5 == 0 or (batch_num + 1) == total_batches:
            print(
                f"  Indexed {min(i + BATCH_SIZE, total)}/{total} chunks "
                f"(batch {batch_num + 1}/{total_batches})"
            )

        time.sleep(0.25)  # rate limit buffer

    # Write fingerprint
    from core.evaluation.fingerprint import write_fingerprint

    write_fingerprint(
        Path("data"),
        collection_name=COLLECTION_NAME,
        target_dataset="contractnli",
        chunk_size=512,
        chunk_overlap=128,
    )

    # Verify
    collection_info = client.get_collection(COLLECTION_NAME)
    sac_points = collection_info.points_count
    print(f"\n  SAC collection ready: {sac_points} points")
    print(f"  Expected: {total} points")

    # Confirm baseline untouched
    baseline_info_after = client.get_collection("contractnli_baseline")
    baseline_after = baseline_info_after.points_count
    print(f"  Baseline still intact: {baseline_after} points "
          f"({'OK' if baseline_after == baseline_points else 'MISMATCH!'})")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    corpus_df = pd.read_parquet(CORPUS_PATH)
    print(f"Loaded corpus: {len(corpus_df)} chunks, "
          f"{corpus_df['doc_id'].nunique()} documents")

    # Phase 1: summaries
    if SUMMARIES_PATH.exists():
        print(f"\nSummaries already exist at {SUMMARIES_PATH}, loading...")
        summaries = json.loads(SUMMARIES_PATH.read_text(encoding="utf-8"))
        print(f"  Loaded {len(summaries)} summaries")
    else:
        summaries = generate_all_summaries(corpus_df)

    # Phase 2: augment corpus
    corpus_df = build_sac_corpus(corpus_df, summaries)

    # Phase 3: embed and index
    embed_and_index(corpus_df)

    print(f"\n{'='*60}")
    print("SAC INDEX BUILD COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
