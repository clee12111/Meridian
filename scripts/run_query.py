"""Thin runner: send one hardcoded query through the v2 pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)

from core.supervisor.context import PipelineContext
from core.supervisor.graph import compile_graph


def main() -> None:
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    print("Building PipelineContext...", flush=True)
    context = PipelineContext.build(
        corpus_path=Path("data/corpus_contractnli.parquet"),
        qdrant_url=qdrant_url,
        collection_name="contractnli_baseline",
        dataset_name="contractnli",
        top_k=50,
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    print(f"  corpus: {len(context.corpus_df)} chunks", flush=True)
    print(f"  qdrant: {qdrant_url} / contractnli_baseline", flush=True)

    print("Compiling graph...", flush=True)
    compiled, saver = compile_graph(
        db_path="data/pipeline.sqlite",
        context=context,
    )

    query = "Does the contract include a non-compete clause?"
    print(f"\nQuery: {query}\n", flush=True)

    initial_state = {
        "raw_query": query,
        "iteration": 1,
        "max_iterations": 3,
        "loop_complete": False,
    }

    config = {"configurable": {"thread_id": "test-run-1"}}
    result = compiled.invoke(initial_state, config=config)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE RESULT")
    print("=" * 70)

    # Phase 3
    print(f"\nrewritten_query: {result.get('rewritten_query', '<not set>')}")

    # Phase 4
    bundle = result.get("retrieval_bundle", {})
    for channel in ("dense", "sparse"):
        ch = bundle.get(channel, {})
        n = len(ch.get("ids", []))
        top3 = ch.get("ids", [])[:3]
        print(f"\nretrieval_bundle.{channel}: {n} results, top 3 ids: {top3}")

    # Phase 5
    fused = result.get("fused_result", {})
    print(f"\nfused_result: {len(fused.get('ids', []))} results, ids: {fused.get('ids', [])}")

    # Phase 6
    reranked = result.get("reranked_result", {})
    print(f"\nreranked_result: {len(reranked.get('ids', []))} results (passthrough)")

    # Phase 7
    ctx_ids = result.get("context_ids", [])
    print(f"\ncontext_ids: {ctx_ids}")
    chunks = result.get("context_chunks", [])
    print(f"context_chunks: {len(chunks)} chunks, first 80 chars each:")
    for i, c in enumerate(chunks[:8]):
        print(f"  [{i}] {c[:80]}...")

    # Phase 8
    answer = result.get("answer", "<not set>")
    print(f"\nanswer ({len(answer)} chars):")
    print(answer[:500])
    if len(answer) > 500:
        print("...")

    # Phase 9
    vr = result.get("verification_result", {})
    print(f"\nverification_result: {json.dumps(vr)}")

    print("\n" + "=" * 70)
    print("END-TO-END: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
