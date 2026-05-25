"""LegalBench-RAG ground truth adapter.

Implements the GroundTruth Protocol from core/measurement/ground_truth.py.
Supports loading one or multiple benchmark JSONs for multi-dataset evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote


class LegalBenchGroundTruth:
    """Ground truth provider for LegalBench-RAG queries.

    Satisfies the ``GroundTruth`` Protocol.  Loads from one or more
    benchmark JSONs and the corpus text files on disk.

    Parameters
    ----------
    benchmark_paths : list of paths
        Paths to benchmark JSON files (e.g., contractnli.json, cuad.json).
    corpus_dir : path
        Path to the ``corpus/`` directory containing subdirs per dataset.
    """

    def __init__(
        self,
        benchmark_paths: list[str | Path] | str | Path,
        corpus_dir: str | Path,
    ) -> None:
        self._corpus_dir = Path(corpus_dir)
        self._queries: dict[str, dict] = {}

        if isinstance(benchmark_paths, (str, Path)):
            benchmark_paths = [benchmark_paths]

        for bp in benchmark_paths:
            with open(bp, encoding="utf-8") as f:
                data = json.load(f)
            for test in data["tests"]:
                qid = test["query_id"]
                self._queries[qid] = test

        self._doc_texts: dict[str, str] = {}

    def get_spans(self, query_id: str) -> list[tuple[int, int]]:
        """Return ground-truth character spans for *query_id*."""
        entry = self._queries[query_id]
        return [tuple(s["span"]) for s in entry["snippets"]]

    def get_doc_id(self, query_id: str) -> str:
        """Return the document ID for *query_id*."""
        entry = self._queries[query_id]
        return entry["snippets"][0]["file_path"]

    def all_query_ids(self) -> list[str]:
        """Return all query IDs (deterministic order)."""
        return sorted(self._queries.keys())

    def doc_text(self, doc_id: str) -> str:
        """Return the full document text for *doc_id*."""
        if doc_id not in self._doc_texts:
            doc_path = self._corpus_dir / unquote(doc_id)
            with open(doc_path, encoding="utf-8") as f:
                self._doc_texts[doc_id] = f.read()
        return self._doc_texts[doc_id]

    def dataset_for_query(self, query_id: str) -> str:
        """Return the dataset name for *query_id*.

        Derived from the query_id prefix (e.g., ``contractnli-0001`` → ``contractnli``).
        """
        return query_id.rsplit("-", 1)[0]

    def query_ids_for_dataset(self, dataset_name: str) -> list[str]:
        """Return all query IDs belonging to *dataset_name*."""
        return sorted(
            qid for qid in self._queries
            if self.dataset_for_query(qid) == dataset_name
        )


# Backwards-compatible alias
ContractNLIGroundTruth = LegalBenchGroundTruth
