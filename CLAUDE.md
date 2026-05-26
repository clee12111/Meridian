# Meridian — Project Spec

## What this is
Personal autonomous research platform. First campaign: LegalBench-RAG 
retrieval forensics on ContractNLI. Domain-agnostic by design.
Primary output is the forensic document — decision log, failure mode 
writeups, autopsy PDF. The agent loop is infrastructure, not the artifact.

## Hard boundary
The loop runs ONLY on retrieval experiments. Generation API calls are 
GATED — never called autonomously. Enforced at the Pydantic schema layer,
not by convention. AutoRAG nodes that call LLMs (HyDE, query decomposition,
LLM-as-reranker, RAGAS) are BLOCKED in the config schema.

---

## Stack (locked — do not substitute without explicit instruction)
- Vector store: Qdrant (Docker, self-hosted)
- Embeddings: voyage-4-large (Voyage SDK)
  voyage-4-large shares embedding space with voyage-4 and voyage-4-lite.
  Phase 3 optimization: embed corpus with voyage-4-large, query with
  voyage-4-lite to reduce per-query embedding cost.
- Sparse retrieval: rank_bm25.BM25Okapi (same engine AutoRAG wraps internally)
- Dense retrieval: core/retrieval/qdrant_retriever.py — hand-rolled (~50 LOC)
  AutoRAG's VectorDBRetrieval wraps ChromaDB — do NOT use it
- Hybrid fusion: hand-rolled RRF in core/retrieval/fusion.py (~10 LOC)
  AutoRAG's HybridRetrieval assumes ChromaDB — do NOT use it
- Tracing: Langfuse (self-hosted) — real-time, per-span, verbatim I/O
- Orchestration: LangGraph StateGraph + SqliteSaver
- HPO: Optuna TPE sampler (Phase 3+)
- Config validation: Pydantic v2 discriminated unions
- Notifications: Apprise (Gmail SMTP + Discord/Slack)
- Generation: GPT-5 Nano Batch API — GATED, never called by the loop
- Infra: VPS + systemd + Docker

## AutoRAG — Outcome B (resolved)
AutoRAG's eval loop does not expose character offsets. Use it as a 
component library only. Safe imports: BM25Retrieval. Do NOT use 
VectorDBRetrieval, HybridRetrieval, MetricInput, or @autorag_metric.
BM25Retrieval must be instantiated ONCE per experiment run, not per query.

## AutoRAG BM25 deviation (resolved)
autorag.nodes.retrieval.bm25.BM25Retrieval is not importable without
llama_index (~500MB). Use rank_bm25.BM25Okapi directly — the same
engine AutoRAG wraps internally. This is not a hand-roll deviation;
it is using the underlying library AutoRAG itself uses.

---

## The four components
1. Measurement library (core/measurement/) — pure functions, no side effects,
   no logging, no API calls. Called by the Supervisor. Never by the Proposer.
2. Proposer agent — reads ledger, outputs RagConfig + hypothesis + predicted delta.
3. Forensic Writer agent — ledger + traces → decision_log entry + nightly email.
4. Supervisor agent — LangGraph StateGraph, loop, checkpoints, kill-switch.

## GroundTruth Protocol (do not extend without a concrete second-domain use case)
```python
class GroundTruth(Protocol):
    def get_spans(self, query_id: str) -> list[tuple[int, int]]: ...
    def get_doc_id(self, query_id: str) -> str: ...
    def all_query_ids(self) -> list[str]: ...
    def doc_text(self, doc_id: str) -> str: ...
```

## RetrievalResult (all retrievers must return this)
```python
@dataclass
class RetrievalResult:
    contents: list[str]
    ids: list[str]
    scores: list[float]
    spans: list[tuple[int, int]]  # joined from corpus.parquet start_end_idx
                                   # never None, never reconstructed from text
```

---

## Measurement requirements
- Spans are half-open [start, end) — verify against LegalBench ground truth before implementing
- span_overlap raises MissingGroundTruthError on empty/None gt — never returns 0.0
- P@k/R@k property-tested with Hypothesis, min 10k examples
- Ingestion: doc_text[start:end] == chunk_text — HARD HALT on mismatch
- Sanity invariants (all thresholds configurable, never hardcoded):
  R@k non-decreasing in k; P@k non-increasing in k;
  no metric improves >SANITY_THRESHOLD pts over prior best (default 15.0)
- On invariant violation: QUARANTINE + flag + notify. Never silently accept.
- eval_mode must be explicit on every MetricResult: SPAN_OVERLAP or LLM_JUDGE
- Every MetricResult carries a Langfuse trace_id — no result without a trace

## Failure taxonomy
Classification uses PER-SPAN coverage analysis: each ground-truth
span is individually scored against retrieved spans.  This correctly
handles multi-span queries (43% of LegalBench-RAG).

Phase 1 (span-computable, precedence order — first match wins):
1. DRM — no retrieved doc matches any gt doc
2. CBF — correct doc, zero overlap on ALL gt spans
3. SGP — >=1 span covered (>=50%) AND >=1 span entirely missed (0%)
         "Found some, missed others." Fix: diversity/coverage in top-k.
4. ICR — total overlap > 0 but < 50% of total gt chars
5. OVR — total retrieved chars >= 3x total gt chars
6. OK  — none of the above

SGP requires parent-document reachability (DRM checked first).
For single-span queries SGP is impossible — behavior identical to
the pre-SGP classifier.

Phase 2 (chunk text parsing — stubs only until explicitly added):
DTM, XRF. Raise NotImplementedError in stubs.

---

## Tracing requirements
The Supervisor owns all Langfuse calls. The measurement library has zero 
Langfuse dependency. Every span captures verbatim JSON inputs and outputs —
never summarized. Per-query spans are mandatory. The Proposer's full prompt 
and full LLM response must be captured verbatim, not just the extracted config.

Span hierarchy: EXPERIMENT TRACE → PROPOSER → INDEX BUILD → EVAL LOOP 
→ (per query: RETRIEVAL + MEASUREMENT) → SANITY CHECK → SCRIBE.

---

## Documentation (parallel to all phases)
Inherits RagForensics structure. Scribe writes decision_log.md entries 
nightly. Human edits them next morning. Human writes autopsy.md at 
campaign end — Scribe never writes the autopsy.

Each decision_log entry: hypothesis (verbatim) → config diff → metric 
delta → failure type shift → forensic interpretation → verdict 
(CONFIRMED / PARTIALLY CONFIRMED / FALSIFIED) → proposed next question.

---

## Build order (phases are gates)
Phase 1 — measurement library + property tests. No agents, no Qdrant, 
no ingestion. Gate: 10k Hypothesis examples pass, known-answer tests pass.

Phase 2 — ContractNLI ingestion + baseline replication. GATE PASSED.
Locked baseline: P@1 8.84%, R@8 50.41% on 194 queries (95 docs, 3797 chunks).
Permanent gate: P@1 [6.8, 10.8], R@8 [47.4, 53.4].
If subsequent experiment regresses outside gate, QUARANTINE.

Phase 3 — autonomous loop: Supervisor → Measurement integration → 
Proposer → Writer → VPS deploy. Wired in that order.

Phase 4 — MCP forensic layer (after >20 runs exist worth querying).

Conditional (add when scope demands, not before):
RAGAS only if span-overlap saturates. Docling only for raw PDF datasets.
Qdrant native BM25 only when expanding past ~50k docs.
DTM/XRF only when they become the dominant unresolved failure type.

---

## Campaign vs. core boundary (non-negotiable)

Core (core/) is domain-agnostic. It has zero knowledge of:
- ContractNLI, LegalBench-RAG, or any specific dataset
- Legal text, NDAs, or any domain-specific structure
- Specific chunking strategies or retrieval techniques

All domain knowledge lives in campaigns/<campaign_name>/.
A new campaign is: one new folder + one GroundTruth adapter.
The core never changes when a new campaign is added.

The GroundTruth Protocol is the only interface between core and campaign.
The ChunkerConfig strategy enum is the only place domain-specific
chunking strategies are registered.

## Cross-dataset contamination (spine of this campaign)

ContractNLI P@1 dropped 8.84% (isolated index) → 3.38% (full 4-dataset
corpus) because a combined index lets ContractNLI queries retrieve CUAD
chunks. Hard Rule 15 (dataset filter mandatory on every Qdrant query)
exists because of this.

### Qdrant dataset filtering — two independent facts

Filtering depends on TWO independent facts, never collapsed into one
boolean:

1. **Must filter**: whether the collection is multi-dataset
   (contamination risk). Single-dataset collections skip the filter
   (nothing to separate). Multi-dataset collections filter-or-RAISE,
   never silently skip.

2. **Can filter**: whether a `dataset_name` keyword payload index
   physically exists on that Qdrant collection. Without it, Qdrant
   rejects filter queries with HTTP 400.

### Locked baseline values are GATES, not targets

ContractNLI hybrid (per-span taxonomy): P@1 8.84%, R@8 50.41%,
failure dist DRM 39.2% / CBF 18.0% / SGP 10.8% / ICR 3.1% /
OVR 10.8% / OK 18.0% (194 queries). Any refactor must reproduce
these exactly or it changed behavior.

Note: pre-SGP baseline had OK 24.2% — 12 false-OK queries were
reclassified to SGP (had a span entirely missed).

### Experiment 1 partial verification

Experiment 1 was only half-verified until 2026-05-25: the 8.84%
recovery came from BM25 per-dataset indexing; the Qdrant payload filter
could not execute until the `dataset_name` keyword index was created on
`legalbench_rag_full`. The full-collection contamination test has not
yet been run with the filter genuinely applied.

### Process rule

When reporting a fix, show the file content on disk and the real
terminal output, never a description of intended changes. Summaries
have diverged from actual file state in this session — always verify.

---

## Hard rules
1. Do not hand-roll anything in the BUY stack without explicit instruction.
2. Do not add dependencies without asking first.
3. Do not write Phase N+1 code before Phase N passes its gate.
4. Measurement functions are pure — no side effects of any kind.
5. No MetricResult without a trace_id.
6. span_overlap raises on empty gt — never fabricates a number.
7. Sanity violation → QUARANTINE, never silent acceptance.
8. Every Langfuse span: verbatim inputs + outputs, never summarized.
9. Per-query spans mandatory — aggregate-only logging is not sufficient.
10. RetrievalResult.spans populated from parquet join — never text reconstruction.
11. BM25Retrieval instantiated once per run, not per query.
12. Scribe writes decision_log entries. Human writes the autopsy. Never reversed.
13. Campaign vs. core boundary is non-negotiable (see section above).
14. No AutoRAG VectorDBRetrieval, HybridRetrieval, MetricInput, or @autorag_metric.
15. Dataset filter mandatory on every Qdrant query to a multi-dataset collection.
    Single-dataset collections skip. Multi-dataset collections RAISE if the
    payload index is missing — never silently skip.