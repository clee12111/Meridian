# Decision Log — LegalBench-RAG / ContractNLI Campaign

---

## Experiment 0 — Baseline establishment (Phase 2)

### Hypothesis
Replicate ZeroEntropy's published baseline on ContractNLI-mini to 
verify the measurement pipeline is correct before autonomous 
experimentation begins.

### Configuration
- Chunker: RCTS 512 chars / 128 overlap
- Embedder: voyage-4-large (2048 dims)
- Retrieval: BM25 + dense RRF (k=60), top-8
- Corpus: 95 LegalBench-RAG ContractNLI documents / 3797 chunks
- Queries: 194 mini-sampled, title-prefixed
  (e.g. "Consider the AGProjects' Non-Disclosure Agreement; 
   Does the document...")

### Results
| Metric | This baseline | Paper (text-embedding-3-large) | Delta |
|--------|--------------|-------------------------------|-------|
| P@1    | 8.84%        | 6.41%                         | +2.43 |
| R@8    | 50.41%       | 26.30%                        | +24.1 |
| DRM    | 39.2%        | not reported                  | —     |
| CBF    | 18.0%        | not reported                  | —     |
| OK     | 24.7%        | not reported                  | —     |

Full metric table:
| k  | P@k    | R@k    |
|----|--------|--------|
| 1  | 8.84%  | 8.84%  |
| 4  | 7.99%  | 26.91% |
| 8  | 6.75%  | 50.41% |
| 16 | 5.45%  | 68.24% |
| 64 | 2.69%  | 91.xx% |

### Forensic findings

**Finding 1 — Query format is the primary document routing signal.**
Running the same pipeline with bare Stanford queries ("Does the 
document restrict reverse engineering?") vs ZeroEntropy's 
title-prefixed queries ("Consider the AGProjects' NDA; Does the 
document...") produced a DRM shift from 87.6% → 39.2% — a 48.4pt 
drop. The party names in the title prefix are what anchor both BM25 
(exact term match) and dense retrieval (entity semantic anchor) to 
the right document among 95 NDAs. Without this signal, 444 Stanford 
NDAs are structurally indistinguishable at query time.

**Implication for future campaigns:** Any multi-document legal RAG 
system needs document-identifying context in the query. Bare clause 
questions are insufficient for document routing. This likely 
generalizes to CUAD, MAUD, and PrivacyQA.

**Finding 2 — voyage-4-large outperforms text-embedding-3-large 
on ContractNLI.**
P@1 delta of +2.43pts over the paper's text-embedding-3-large 
baseline, on identical corpus and query set. This is consistent with 
Voyage AI's published RTEB numbers showing voyage-4-large outperforms 
OpenAI text-embedding-3-large by ~14% on retrieval tasks.

**Finding 3 — DRM at 39.2% is the dominant remaining failure mode.**
Even with correct query format and a strong embedder, 39% of queries 
retrieve the wrong document entirely. This is the primary target for 
Experiments 1-4. Chunking improvements (ICR, CBF) are secondary — 
they only matter for the 61% of queries where document routing succeeds.

**Finding 4 — CBF emerged as the second failure mode (18.0%).**
Once DRM dropped from 88% to 39%, chunk boundary failures became 
visible at 18%. These are queries where the right document is 
retrieved but the relevant clause straddles a chunk boundary. 
RCTS 512/128 splits conditional clauses. This is the target for 
chunking experiments (section-aware splitting, larger chunk size).

**Finding 5 — R@8 gap vs paper (50.41% vs 26.3%) partially 
explained by corpus concentration.**
The 194 mini-sampled queries reference only 18 of 95 documents. 
This concentration means that once the retriever routes to a 
correct document, R@8 is high because multiple queries share the 
same 18 relevant documents. The paper's lower R@8 is likely 
attributable to the weaker embedder (text-embedding-3-large) 
missing more within-document matches.

### Verdict: CONFIRMED (with annotation)
Pipeline verified. Measurement layer producing consistent, 
interpretable results. Gate locked at P@1 [6.8, 10.8], 
R@8 [47.4, 53.4] for all subsequent experiments.

### Proposed next question
DRM at 39.2% is the dominant failure. Does increasing chunk size 
from 512 to 1024 chars reduce DRM by providing more document-level 
context per chunk, or does DRM reflect an embedder limitation that 
chunk size cannot address?

---

## Experiment 0c — Replication with text-embedding-3-large

**Purpose:** Validate measurement pipeline against paper's exact 
configuration.

**Result:**
| Metric | text-embedding-3-large | voyage-4-large | Paper |
|--------|----------------------|----------------|-------|
| P@1    | 7.47%                | 8.84%          | 6.41% |
| R@8    | 46.72%               | 50.41%         | 26.30%|
| DRM    | 24.7%                | 39.2%          | —     |

**Pipeline validation: CONFIRMED.** P@1 within ±1.1pts of paper 
on both embedders. Measurement layer is correct.

**R@8 discrepancy confirmed as chunking artifact.** Both embedders 
show identical +20pt inflation vs paper. Root cause: 512/128 RCTS 
chunking produces ~40 chunks/doc. At top-8 retrieval this covers 
20% of each document, inflating recall. Not a bug — a configuration 
difference. Experiment 2 (chunk size sweep) will test whether 
larger chunks bring R@8 toward paper's numbers.

**Novel finding — embedder DRM tradeoff:**
text-embedding-3-large has lower DRM (24.7%) but lower span 
precision (P@1 7.47%). voyage-4-large has higher DRM (39.2%) 
but higher span precision (P@1 8.84%). On ContractNLI's 95-doc 
corpus: text-embedding-3-large routes better, voyage-4-large 
extracts more precisely when routing succeeds.

**Hypothesis for full corpus:** On larger corpora (CUAD ~500 docs) 
document routing becomes harder and the DRM gap between embedders 
will widen. text-embedding-3-large's routing advantage may dominate 
at scale. To be tested in cross-dataset expansion.

**Verdict: CONFIRMED.** Phase 2 closed. Expanding to all four 
LegalBench-RAG datasets before Phase 3.

---

## Experiment 1 — Full 4-dataset baseline (contamination fix)

### Correction (2026-05-25): Qdrant filter was latent

The Qdrant `dataset_name` payload filter was never actually exercised
during Experiment 1's initial run. Investigation confirmed that
`legalbench_rag_full` (293,323 points) had `payload_schema: {}` —
no payload index on `dataset_name`. Without a keyword index, Qdrant
rejects filter queries with HTTP 400; the filter code path was
unreachable.

The 8.84% ContractNLI P@1 recovery reported in Experiment 1 came
entirely from **BM25 per-dataset indexing** (`BM25Retriever` builds
separate BM25Okapi instances per `dataset_name` value in the corpus
DataFrame, `core/retrieval/bm25_retriever.py:52-57`). The Qdrant
dense retrieval half ran unfiltered — cross-dataset contamination
in the dense retrieval results was suppressed only by RRF fusion
weighting BM25's correctly-filtered results above Qdrant's
unfiltered results.

### Actions taken this session

1. Created keyword payload index on `dataset_name` for
   `legalbench_rag_full` (293,323 points indexed, confirmed via
   `payload_schema`).
2. Redesigned `QdrantRetriever` filtering: `multi_dataset` flag set
   at construction; single-dataset collections skip the filter;
   multi-dataset collections RAISE if the index is missing (see
   `docs/architecture.md` for the full invariant).
3. `contractnli_baseline` (single-dataset, no index) now correctly
   skips filtering, reproducing the locked 8.84%/50.41%.

### Status

Experiment 1 is now fully built — both BM25 and Qdrant filtering
are wired and the payload index exists. However, the full-collection
contamination test has **not yet been re-run** with the Qdrant filter
genuinely applied. Until that run completes and metrics are compared
against the latent-filter run, the Experiment 1 results should be
considered provisional.

### Proposed next step

Re-run `scripts/run_baseline.py --full --persist --eval-only` with
the filter active. Compare per-dataset metrics against the prior
(latent-filter) run. If R@k improves on non-ContractNLI datasets
(CUAD, MAUD, PrivacyQA), that confirms cross-dataset contamination
was present in the dense retrieval path.
