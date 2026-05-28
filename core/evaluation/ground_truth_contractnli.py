"""ContractNLI ground truth adapter.

Implements the GroundTruth protocol from core/measurement/ground_truth.py.
Reads data/benchmarks/contractnli.json and data/corpus/contractnli/*.txt.
"""

from __future__ import annotations

import json
from pathlib import Path


class ContractNLIGroundTruth:
    """Ground truth provider for the ContractNLI dataset.

    Parameters
    ----------
    benchmark_path : Path
        Path to contractnli.json (e.g. data/benchmarks/contractnli.json).
    corpus_dir : Path
        Path to the corpus root (e.g. data/corpus). Doc files are at
        corpus_dir / "contractnli" / "<filename>.txt".
    """

    def __init__(self, benchmark_path: Path, corpus_dir: Path) -> None:
        self._corpus_dir = corpus_dir
        self._queries: dict[str, str] = {}
        self._snippets: dict[str, list[dict]] = {}
        self._doc_cache: dict[str, str] = {}

        data = json.loads(benchmark_path.read_text(encoding="utf-8"))
        for test in data["tests"]:
            qid = test["query_id"]
            self._queries[qid] = test["query"]
            self._snippets[qid] = test["snippets"]

    def get_spans(self, query_id: str) -> list[tuple[int, int]]:
        """Return ground-truth character spans for *query_id*."""
        if query_id not in self._snippets:
            raise KeyError(f"Unknown query_id: {query_id!r}")
        return [tuple(s["span"]) for s in self._snippets[query_id]]

    def get_doc_id(self, query_id: str) -> str:
        """Return the document ID (file_path) for *query_id*.

        All ContractNLI snippets reference a single document per query.
        """
        if query_id not in self._snippets:
            raise KeyError(f"Unknown query_id: {query_id!r}")
        return self._snippets[query_id][0]["file_path"]

    def all_query_ids(self) -> list[str]:
        """Return sorted list of all query IDs."""
        return sorted(self._queries.keys())

    def get_query_text(self, query_id: str) -> str:
        """Return the query text for *query_id*."""
        if query_id not in self._queries:
            raise KeyError(f"Unknown query_id: {query_id!r}")
        return self._queries[query_id]

    def doc_text(self, doc_id: str) -> str:
        """Return the full document text for *doc_id*.

        Lazy-loads from corpus_dir / doc_id and caches.
        doc_id is e.g. "contractnli/ceii-and-nda.txt".
        """
        if doc_id not in self._doc_cache:
            path = self._corpus_dir / doc_id
            self._doc_cache[doc_id] = path.read_text(encoding="utf-8")
        return self._doc_cache[doc_id]
