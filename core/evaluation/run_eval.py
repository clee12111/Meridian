"""Reusable evaluation pipeline: ingest -> retrieve -> measure -> aggregate.

Called by both scripts/run_baseline.py and the Supervisor run_eval node.
No generation API calls -- retrieval and measurement only.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from core.measurement.metrics import MetricResult, compute_all_k
from core.measurement.taxonomy import FailureType, classify
from core.retrieval.bm25_retriever import BM25Retriever

K_VALUES = [1, 2, 4, 8, 16, 32, 64]

DATASET_NAMES = ["contractnli", "cuad", "maud", "privacy_qa"]


def evaluate_config(
    config: dict,
    dataset_name: str | None = None,
    data_dir: str | Path | None = None,
    skip_index: bool = False,
) -> tuple[MetricResult, dict[str, int]]:
    """Run the full retrieval evaluation pipeline for a given config.

    Parameters
    ----------
    config : dict
        Must contain keys: chunk_size, chunk_overlap, bm25_top_k,
        dense_top_k, fusion_top_n, retrieval_mode.
        retrieval_mode must be "bm25" or "hybrid".
    dataset_name : str or None
        If None, evaluate all available datasets. Otherwise one of:
        "contractnli", "cuad", "maud", "privacy_qa".
    data_dir : path or None
        Defaults to env MERIDIAN_DATA_DIR or "data".

    Returns
    -------
    (MetricResult, failure_counts)
        MetricResult with mean P@k and R@k across all queries.
        failure_counts maps FailureType.name -> count.

    Raises
    ------
    ValueError
        If retrieval_mode is missing or not one of ("bm25", "hybrid").
    EnvironmentError
        If retrieval_mode is "hybrid" but VOYAGE_API_KEY is not set.
    """
    from campaigns.legalbench_rag.ground_truth_adapter import LegalBenchGroundTruth
    from campaigns.legalbench_rag.ingestion.contractnli_loader import ingest_contractnli
    from campaigns.legalbench_rag.ingestion.full_loader import ingest_all

    # --- Validate retrieval_mode (mandatory, no default) ---
    retrieval_mode = config.get("retrieval_mode")
    if retrieval_mode not in ("bm25", "hybrid"):
        raise ValueError(
            f"config['retrieval_mode'] must be 'bm25' or 'hybrid', "
            f"got {retrieval_mode!r}"
        )

    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]
    bm25_top_k = config["bm25_top_k"]
    dense_top_k = config["dense_top_k"]
    fusion_top_n = config["fusion_top_n"]

    if data_dir is None:
        data_dir = Path(os.environ.get("MERIDIAN_DATA_DIR", "data"))
    else:
        data_dir = Path(data_dir)

    # --- Ingest (per-dataset to avoid unnecessary work) ---
    if dataset_name == "contractnli":
        corpus_df, _ = ingest_contractnli(
            data_dir, chunk_size, chunk_overlap, max_queries=194
        )
    elif dataset_name == "maud":
        from campaigns.legalbench_rag.ingestion.maud_loader import ingest_maud
        corpus_df, _ = ingest_maud(
            data_dir, chunk_size, chunk_overlap, max_queries=194
        )
    elif dataset_name == "cuad":
        from campaigns.legalbench_rag.ingestion.cuad_loader import ingest_cuad
        corpus_df, _ = ingest_cuad(
            data_dir, chunk_size, chunk_overlap, max_queries=194
        )
    elif dataset_name == "privacy_qa":
        from campaigns.legalbench_rag.ingestion.privacyqa_loader import ingest_privacyqa
        corpus_df, _ = ingest_privacyqa(
            data_dir, chunk_size, chunk_overlap, max_queries=194
        )
    else:
        corpus_df, _ = ingest_all(
            data_dir, chunk_size, chunk_overlap, max_queries_per_dataset=194
        )

    # --- Ground truth ---
    if dataset_name is not None:
        benchmark_paths = [data_dir / "benchmarks" / f"{dataset_name}.json"]
    else:
        benchmark_paths = [
            data_dir / "benchmarks" / f"{ds}.json"
            for ds in DATASET_NAMES
            if (data_dir / "benchmarks" / f"{ds}.json").exists()
        ]
    gt = LegalBenchGroundTruth(benchmark_paths, data_dir / "corpus")
    query_ids = gt.all_query_ids()

    # --- Build retrievers ---
    bm25 = BM25Retriever(corpus_df, top_k=bm25_top_k)

    qdrant = None
    if retrieval_mode == "hybrid":
        voyage_key = os.environ.get("VOYAGE_API_KEY")
        if not voyage_key:
            raise EnvironmentError(
                "retrieval_mode is 'hybrid' but VOYAGE_API_KEY is not set"
            )

        from core.retrieval.qdrant_retriever import QdrantRetriever
        from qdrant_client import QdrantClient

        qdrant_url = os.environ.get("QDRANT_URL", "")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
        if qdrant_url:
            client_kwargs: dict = {"url": qdrant_url}
            if qdrant_api_key:
                client_kwargs["api_key"] = qdrant_api_key
            qdrant_client = QdrantClient(**client_kwargs)
        else:
            qdrant_client = QdrantClient(":memory:")

        is_multi = dataset_name is None
        collection = "legalbench_rag_full" if is_multi else "contractnli_baseline"

        qdrant = QdrantRetriever(
            client=qdrant_client,
            collection_name=collection,
            corpus_df=corpus_df,
            top_k=dense_top_k,
            multi_dataset=is_multi,
        )
        if skip_index:
            point_count = qdrant.collection_point_count()
            if point_count == 0:
                raise RuntimeError(
                    f"skip_index=True (query_time experiment) but collection "
                    f"'{collection}' has 0 points. Run an ingestion_time "
                    f"experiment first to populate the index."
                )
        else:
            qdrant.index()

    # --- Per-query eval loop ---
    all_p: dict[int, list[float]] = {k: [] for k in K_VALUES}
    all_r: dict[int, list[float]] = {k: [] for k in K_VALUES}
    failure_counts: Counter[FailureType] = Counter()

    for qid in query_ids:
        query_text = gt.get_query_text(qid)
        gt_spans = gt.get_spans(qid)
        gt_doc_id = gt.get_doc_id(qid)
        ds = gt.dataset_for_query(qid)

        # Retrieve
        if qdrant is not None:
            from core.retrieval.fusion import rrf

            sparse_result = bm25.retrieve(query_text, top_k=bm25_top_k, dataset_name=ds)
            dense_result = qdrant.retrieve(query_text, top_k=dense_top_k, dataset_name=ds)
            retrieval_result = rrf(sparse_result, dense_result, top_n=fusion_top_n)
        else:
            retrieval_result = bm25.retrieve(query_text, top_k=fusion_top_n, dataset_name=ds)

        # Measure
        result = compute_all_k(retrieval_result.spans, gt_spans, k_values=K_VALUES)
        for k in K_VALUES:
            all_p[k].append(result.p_at_k[k])
            all_r[k].append(result.r_at_k[k])

        # Classify failure
        retrieved_with_docs = [
            (cid.split("#")[0], s[0], s[1])
            for cid, s in zip(retrieval_result.ids, retrieval_result.spans)
        ]
        gt_with_docs = [(gt_doc_id, s[0], s[1]) for s in gt_spans]
        failure = classify(retrieved_with_docs, gt_with_docs)
        failure_counts[failure] += 1

    # --- Aggregate ---
    n = len(query_ids)
    mean_p = {k: sum(all_p[k]) / n for k in K_VALUES}
    mean_r = {k: sum(all_r[k]) / n for k in K_VALUES}

    metric_result = MetricResult(
        p_at_k=mean_p,
        r_at_k=mean_r,
        eval_mode="SPAN_OVERLAP",
    )

    return metric_result, {ft.name: count for ft, count in failure_counts.items()}
