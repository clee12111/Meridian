"""RetrievalResult — the universal retrieval output type.

All retrievers must return this. spans must be populated from a
corpus.parquet join on doc_id — never reconstructed from text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Result from any retriever.

    Attributes
    ----------
    contents : list[str]
        The text content of each retrieved chunk.
    ids : list[str]
        The chunk_id of each retrieved chunk.
    scores : list[float]
        The relevance score of each retrieved chunk.
    spans : list[tuple[int, int]]
        Half-open ``[start, end)`` character offsets into the source
        document, joined from ``corpus.parquet`` ``start_end_idx``.
        Never ``None``, never reconstructed from text.
    """

    contents: list[str]
    ids: list[str]
    scores: list[float]
    spans: list[tuple[int, int]]

    def __post_init__(self) -> None:
        n = len(self.contents)
        if len(self.ids) != n or len(self.scores) != n or len(self.spans) != n:
            raise ValueError(
                f"All fields must have the same length, got "
                f"contents={len(self.contents)}, ids={len(self.ids)}, "
                f"scores={len(self.scores)}, spans={len(self.spans)}"
            )
        for span in self.spans:
            if span is None:
                raise ValueError("spans must be populated — got None entry")
