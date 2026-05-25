# Architecture Notes

## Qdrant dataset-filtering invariant

Filtering on `dataset_name` in Qdrant depends on two independent facts:

1. **Must filter** — whether the collection is multi-dataset (contamination
   risk). Multi-dataset collections (e.g. `legalbench_rag_full`, 4 datasets,
   293k points) MUST filter per-query to prevent cross-dataset contamination.
   Single-dataset collections (e.g. `contractnli_baseline`, 1 dataset, 3797
   points) MUST NOT filter — there is nothing to filter and applying one
   triggers a 400 if no payload index exists.

2. **Can filter** — whether a `dataset_name` keyword payload index physically
   exists on that Qdrant collection. Without it, Qdrant rejects filter
   queries with HTTP 400.

### Enforcement

The requirement is set explicitly at retriever construction via the
`multi_dataset` flag (`core/retrieval/qdrant_retriever.py: __init__`),
NOT inferred from `corpus_df.columns`. The retriever checks the actual
collection's payload schema at construction
(`QdrantRetriever._check_dataset_index`).

Retrieve-time logic (`core/retrieval/qdrant_retriever.py: retrieve`):

- `dataset_name` given + `multi_dataset=False` (single-dataset collection):
  **skip** the filter. Nothing to filter. Reproduces locked baseline 8.84%.

- `dataset_name` given + `multi_dataset=True` + index exists:
  **apply** the filter. Required to prevent cross-dataset contamination.

- `dataset_name` given + `multi_dataset=True` + index MISSING:
  **RAISE RuntimeError**. Never silently skip on multi-dataset collections.
  This is Hard Rule 15: contamination must halt, not degrade silently.

### Why not infer from corpus_df?

`ingest_contractnli` adds a `dataset_name` column to the DataFrame even
for single-dataset corpora. Using `"dataset_name" in corpus_df.columns`
as the gate would incorrectly trigger filtering on `contractnli_baseline`
(which has no payload index). The decision must come from the collection's
actual state, not the ingestion DataFrame's schema.

### BM25Retriever

BM25 decides filtering from its own state: `self._dataset_indexes` is
non-empty only when the corpus had multiple datasets indexed
(`core/retrieval/bm25_retriever.py:52-57`). If `dataset_name` is passed
but `_dataset_indexes` is empty, it falls back to the full-corpus index
(line 75). This is correct for single-dataset corpora.
