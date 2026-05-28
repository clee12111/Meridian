> Decision log: see docs/DECISIONS.md

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

---

## Failure classifier: single-document-ground-truth precondition

**HARD PRECONDITION**: The failure classifier (`core/measurement/taxonomy.py:
classify`) does NOT support multi-document ground truth.

`_per_span_coverage` (taxonomy.py:34) operates on raw `(start, end)` tuples
after doc_ids are stripped at line 112. When ground-truth spans from different
documents share character offsets, retrieved characters from one document
falsely satisfy another document's span — the per-span coverage merges
character positions into a single flat namespace with no per-document
isolation.

**Example**: gt = `[("doc_a", 0, 100), ("doc_b", 0, 100)]`, retrieved =
`[("doc_a", 0, 100)]`. The classifier reports OK (both spans "covered")
even though doc_b's span is never retrieved. Correct answer: SGP.

**Why this is acceptable today**: All four LegalBench-RAG datasets
(ContractNLI, CUAD, MAUD, PrivacyQA) have zero multi-document queries
(verified 2025-05-25: 0/776 queries across all benchmarks). The
`GroundTruth.get_doc_id()` protocol returns a single doc_id per query,
and `run_eval.py:184` constructs gt_with_docs using that single doc_id.

**Gate for future datasets**: ANY multi-document or multi-hop dataset
(e.g. HotpotQA, FEVER, BEIR-multihop) requires fixing this with
per-document span grouping BEFORE it can be measured correctly. Until
fixed, only single-document-ground-truth datasets produce valid
classifications. The fix: group gt_spans by doc_id in `classify()` and
run `_per_span_coverage` separately per document, then aggregate the
per-span verdicts.

Discovered: measurement audit 2025-05-25, test_multi_doc_char_offset_collision.

---

## Fusion: RRF, CC, and weighted RRF

Three fusion functions are available (`core/retrieval/fusion.py`):

- **RRF** (`rrf_fuse`) — reciprocal rank fusion (k=60). Rank-based,
  discards score magnitude. Production-robust default for uncalibrated
  scores across heterogeneous systems.

- **CC** (`cc_fuse`) — convex combination: `α·norm(BM25) + (1-α)·norm(dense)`,
  min-max normalized. Preserves score magnitude — a BM25 gap of 0.95 vs 0.55
  survives into the final ranking. Largest single improvement in the campaign
  (+24.5pp P@1 over RRF). Best for structured benchmark corpora where scores
  are calibrated. Per-dataset α required: ContractNLI α=0.3, PrivacyQA α=0.1.

- **Weighted RRF** (`weighted_rrf_fuse`) — rank-based with per-channel weights.
  Strictly less expressive than CC (Bruch et al.): any wRRF config has a
  rank-equivalent or better CC config.

Selection: set `MERIDIAN_CC_ALPHA` or `MERIDIAN_WRRF_SPARSE` env vars, or
pass `--cc-alpha` / `--wrrf-sparse` CLI flags. If neither is set, RRF is used.

---

## Cited-span measurement (secondary)

The harness runs two parallel classifications per query:

1. **Chunk-span** (primary) — uses full chunk boundaries from the parquet
   `start_end_idx`. Always valid. Measures retrieval-region quality.

2. **Cited-span** (secondary) — uses Phase 8's `cited_text` field, located
   in the source document via `extract_cited_spans()` (deterministic substring
   search, no LLM). Measures end-to-end evidence-use precision.

~83% of OVR was a measurement artifact of chunk boundaries (Finding 19).
Cited-span unmasked +5 CBF and +9 SGP invisible under chunk-span.

**Domain boundary:** cited-span works only on extractive corpora where
`cited_text` is verbatim-findable via `str.find()`. Extraction failure was
7.7% (non-DRM) on ContractNLI. Never make cited-span primary where
extraction failure exceeds ~25%.
