"""PrivacyQA ground truth adapter.

Implements the GroundTruth protocol from core/measurement/ground_truth.py.
Reads data/benchmarks/privacy_qa.json and data/corpus/privacy_qa/*.txt.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote


class PrivacyQAGroundTruth:
    """Ground truth provider for the PrivacyQA dataset.

    Parameters
    ----------
    benchmark_path : Path
        Path to privacy_qa.json (e.g. data/benchmarks/privacy_qa.json).
    corpus_dir : Path
        Path to the corpus root (e.g. data/corpus). Doc files are at
        corpus_dir / "privacy_qa" / "<filename>.txt".
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
            # Normalize URL-encoded file_paths (%20 → space) at load time
            for snippet in test["snippets"]:
                snippet["file_path"] = unquote(snippet["file_path"])
            self._snippets[qid] = test["snippets"]

    def get_spans(self, query_id: str) -> list[tuple[int, int]]:
        """Return ground-truth character spans for *query_id*."""
        if query_id not in self._snippets:
            raise KeyError(f"Unknown query_id: {query_id!r}")
        return [tuple(s["span"]) for s in self._snippets[query_id]]

    def get_doc_id(self, query_id: str) -> str:
        """Return the document ID (file_path) for *query_id*.

        All PrivacyQA snippets reference a single document per query.
        """
        if query_id not in self._snippets:
            raise KeyError(f"Unknown query_id: {query_id!r}")
        return unquote(self._snippets[query_id][0]["file_path"])

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
        doc_id is e.g. "privacy_qa/Fiverr.txt".
        """
        if doc_id not in self._doc_cache:
            path = self._corpus_dir / doc_id
            self._doc_cache[doc_id] = path.read_text(encoding="utf-8")
        return self._doc_cache[doc_id]
