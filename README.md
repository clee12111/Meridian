# Meridian

A forensic measurement framework for retrieval-augmented generation. Meridian evaluates RAG strategy choices across all 10 pipeline phases on a single axis: does a given strategy improve retrieval and answer quality, measured identically every time.

Most RAG evaluation uses either pure retrieval metrics (recall@k, precision@k) or pure LLM-judged answer scores. Meridian uses both in a two-layer design where **Layer 1 is deterministic — no LLM judges** — and Layer 2 is LLM-judged. The layers never contaminate each other. Layer 1 is the trust anchor: a deterministic span-overlap taxonomy that can't be gamed and makes measurement bugs distinguishable from model bugs. Layer 2 measures answer quality on top of that foundation.

The agent (a 10-phase RAG pipeline) is what gets measured. The measurement framework is the contribution.

---

## The measurement framework

### Layer 1 — deterministic retrieval taxonomy (no LLM judges)

Every retrieved span is classified against ground-truth character offsets into one of six failure types:

| Code | Meaning | What it tells you |
|------|---------|-------------------|
| **DRM** | Document-level retrieval miss | Retrieved from the wrong document entirely |
| **CBF** | Chunk boundary failure | Right document, but the answer straddles a chunk split |
| **SGP** | Span gap | Right document, but only part of a multi-span answer was found |
| **ICR** | Incorrect region | Right document, wrong section |
| **OVR** | Over-retrieval | Right region, but the retrieved span is much coarser than the evidence |
| **OK** | Correct | Retrieved span covers the ground-truth evidence |

This taxonomy is computed by character-span overlap — pure arithmetic on character offsets. No embeddings, no model calls, no judgment. Scoring is per-span (not merged-character-set) to correctly handle multi-span evidence.

**Why deterministic matters:** when a number moves, you know whether the pipeline changed or the measurement changed. LLM judges drift with model updates, temperature, and prompt wording. A deterministic anchor eliminates that variable.

Phase 9's deterministic citation-traceability (ENTAILED / CONTRADICTED / BASELESS per claim) also lives in Layer 1. Additional corpus-agnostic Tier A metrics (robustness, drift, latency, cost, trajectory) are designed but not yet built.

### Layer 2 — LLM-judged answer metrics (separate, clearly labeled)

Two metrics, both judged by a pinned model (DeepSeek-v4-flash, temperature 0, thinking disabled) held constant across all comparisons:

- **Correctness** — does the answer convey the same information as the ground-truth evidence? Span-informed: the judge sees both the system's answer and the golden evidence text. Semantic match, not string match.
- **Faithfulness** — is each answer claim entailed by the retrieved context? Holistic groundedness by default: each claim is judged against the *full* retrieved context (RAGAS definition, domain-agnostic). A separate strict citation-precision mode judges each claim against only its cited chunk — valid as a diagnostic on extractive/legal corpora, never the headline.

Layer 2 never contaminates Layer 1. They are reported separately and measure different things: Layer 1 measures the retrieval system, Layer 2 measures the reasoning/generation system.

---

## The 10-phase pipeline and what measurement found

The pipeline runs phases 1-2 once per corpus (indexing) and phases 3-10 per query. Each phase has been the subject of controlled A/B comparisons with the measurement framework evaluating the outcome. The framework's job is to make these calls on evidence, including saying no.

| Phase | What it does | Strategies compared | Measurement verdict |
|-------|-------------|---------------------|---------------------|
| **1. Chunking** | Split documents into retrievable units | Fixed-stride (2048 char) vs section-aware boundary detection | Section-aware is **corpus-dependent, rejected as general lever**. Marginal on CUAD (+3.6pp correctness), neutral on ContractNLI, harmful on MAUD (-22.8pp R@8) and PrivacyQA (-7.2pp correctness). Mechanism: section boundaries split multi-span evidence, collapsing recall on complex documents. |
| **2. Indexing** | Embed and store chunks | Raw chunks vs Summary-Augmented Chunking (SAC — document summary prepended before embedding, discarded after) | SAC **validated**. Bakes document identity into dense embeddings, reducing DRM by ~10pp. Additive with other retrieval improvements. |
| **3. Query rewriting** | Rewrite query for retrieval | DeepSeek-flash rewrite vs passthrough | Rewriter provides slight net positive. **Exonerated** as the cause of document discrimination failures (tested, DRM unchanged ±1.7pp). |
| **4. Retrieval** | Dense (Voyage) + sparse (BM25) | With and without document routing (hybrid dense+BM25 over document summaries, pre-filtering retrieval to top-N documents) | Routing **always-on** (domain-agnostic policy). Helps all four corpora at the answer level (+1.0 to +12.9pp, never hurts). Largest gain on highest-DRM corpus. The validated lever for document discrimination. |
| **5. Fusion** | Combine dense + sparse results | RRF vs convex combination (CC) fusion | CC **over RRF on structured benchmark corpora**. CC preserves score magnitude (a BM25 score gap of 0.95 vs 0.52 on a rare party-name token survives into the final ranking; RRF compresses it to near-zero rank difference). Per-corpus alpha: ContractNLI 0.2, CUAD 0.1, MAUD 0.2, PrivacyQA 0.1. All dense-heavy. |
| **6. Reranking** | Cross-encoder re-scoring | Voyage rerank-2.5 on vs off | Reranker **OFF on topically-homogeneous corpora**. Cross-encoder reranking by semantic relevance is blind to document identity — on 95 near-identical NDAs, it confidently promotes wrong-document chunks that are topically relevant. Produces ~79% DRM regardless of candidate quality. Confirmed three independent ways: aggregate, controlled (same result with CC and RRF input), and mechanistic (per-query Phoenix traces showing 0/8 correct-document chunks at reranker scores 0.91-0.95). |
| **7. Context** | Select chunks for the LLM | Top-8 from fused/reranked results | Top-8 captures all useful signal; R@8 = R@16 = R@64 across all corpora. |
| **8. Synthesis** | LLM generates answer with claims | DeepSeek-flash, structured output via instructor (claim + cited_chunk_id + cited_text per assertion) | Structured output enables both Phase 9 verification and faithfulness judging without an extra claim-extraction step. |
| **9. Verification** | Deterministic citation check | Three-way: ENTAILED / CONTRADICTED / BASELESS per claim, via normalized text matching against chunk content | No LLM calls. Provides the grounding signal for Phase 10's loop decision. |
| **10. Agentic loop** | Iterate retrieval if evidence insufficient | Single-shot vs multi-iteration (up to 3) | **Single-shot preferred.** Loop adds +1.8pp at +68% compute. It recovers some multi-span gaps but worsens document discrimination. Single-shot captures ~96% of loop performance at ~60% of the cost. |

**Best configuration:** SAC + no reranker + CC fusion (per-corpus alpha) + always-on hybrid document routing (top-3) + single-shot.

---

## Current state

Validated on four LegalBench-RAG corpora (ContractNLI, CUAD, MAUD, PrivacyQA) — all legal domain. 194 queries per corpus, all on voyage-4 embeddings.

### Retrieval (Layer 1)

| Corpus | P@1 | R@8 | Published baseline P@1 | Published baseline R@8 |
|--------|-----|-----|------------------------|------------------------|
| ContractNLI | 0.381 | 0.807 | 0.088 | 0.503 |
| MAUD | 0.247 | 0.732 | 0.027 | 0.062 |
| CUAD | 0.325 | 0.701 | — | — |
| PrivacyQA | 0.326 | 0.588 | — | — |

Published baselines are from the LegalBench-RAG benchmark (Pipitone & Alami, RCTS method with text-embedding-3-large). This is a system-vs-system comparison (full pipeline vs their baseline stack), not a single-component ablation.

### Answer quality (Layer 2)

**Correctness** (span-informed judge, routing ON):

| Corpus | Correct |
|--------|---------|
| ContractNLI | 75.3% |
| MAUD | 66.5% |
| CUAD | 63.9% |
| PrivacyQA | 61.9% |

The only clean before/after delta is ContractNLI: 25.8% → 75.3% (+49.5pp), with the same span-informed judge applied to both the v1 baseline and the best config. The other three corpora have absolute numbers only (no baseline answer-correctness run).

**Faithfulness** (holistic groundedness, full retrieved context):

| Corpus | Mean faithfulness |
|--------|-------------------|
| PrivacyQA | 97.2% |
| MAUD | 96.8% |
| CUAD | 95.5% |
| ContractNLI | 92.4% |

High faithfulness on legal text is a true finding (extractive domain, model mostly quotes), not a dud metric. Faithfulness is orthogonal to correctness — DRM queries score faithfulness 1.0 because the model faithfully reports what the *wrong* document says.

---

## What's not yet done

- **Non-legal transfer (the open question).** The measurement framework has been validated only on legal corpora. The transferability gate — Tier A measurement producing signal on a non-annotated corpus (FiQA or NFCorpus) — is not started. Until this is demonstrated, the framework's generality is a claim, not a result.
- **Hierarchical chunking.** The structural answer to section chunking's fragmentation problem: retrieve on small pure children, feed the parent section to the model. Separates the retrieval unit from the reading unit. In progress; a Tier B measurement-scope question (child-span vs parent-span scoring) must be locked before build.
- **Synthesis-gap fix.** The largest remaining error cluster: 18-43 queries per corpus where the answer is incorrect despite the model having correct, grounded evidence (INCORRECT + FAITHFUL). The model had the right evidence and reached the wrong conclusion. This is a reasoning-layer problem, not a retrieval problem — the measurement framework's failure decomposition is what makes it visible.
- **Reasoning-based loop gate.** The current loop uses a deterministic grounding threshold. A reasoning-aware gate (loop only when the failure type is recoverable by re-retrieval) is the agentic-loop frontier.

---

## Product direction

*Vision, not built:* the measurement layer surfaced as a standalone tool — a CLI or web interface where a user points it at any RAG pipeline's outputs and gets the two-layer diagnostic (Layer 1 taxonomy + Layer 2 answer quality) without adopting the full Meridian agent. The framework's value is in making strategy calls on evidence; the agent is one consumer of that value.

---

## Reproducing a strategy comparison

### Infrastructure

- **Qdrant** — vector database (cloud or local Docker)
- **Voyage AI** — embeddings (voyage-4) and optional reranking (rerank-2.5)
- **DeepSeek** — LLM calls (deepseek-v4-flash for query rewriting, synthesis, and judging)

```bash
cp .env.example .env
# Fill in: VOYAGE_API_KEY, DEEPSEEK_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

### Run an evaluation arm

```bash
# Run 194 queries on ContractNLI with the best config
python scripts/run_corpus_eval.py \
  --corpus contractnli \
  --chunk-alpha 0.2 \
  --routing-topk 3 \
  --routing-alpha 0.3 \
  --workers 12 \
  --output data/eval_contractnli.jsonl

# Judge answer correctness (Layer 2)
python scripts/judge_answers_v2.py \
  --corpus contractnli \
  --input data/eval_contractnli.jsonl

# Judge faithfulness (Layer 2, holistic groundedness)
python scripts/judge_faithfulness.py \
  --corpus contractnli \
  --input data/eval_contractnli.jsonl
```

### A/B comparison

Change one variable (e.g., chunking strategy, fusion method, routing on/off) and run both arms with the same judge. The eval harness outputs per-query JSONL with Layer 1 taxonomy (failure_type, P@1, R@8) and Layer 2 inputs (answer, claims, context_chunks). The judges add their verdicts to separate output files.

Environment flags for A/B toggles: `MERIDIAN_NO_REWRITE`, `MERIDIAN_NO_RERANK`, `MERIDIAN_CC_ALPHA`, `MERIDIAN_ROUTING_TOPK`. See `CLAUDE.md` for the full list.

### Key dependencies

Python 3.11+. Core: `qdrant-client`, `voyageai`, `openai`, `instructor`, `langgraph`, `rank-bm25`, `pandas`, `numpy`.

---

## Repository map

```
core/
  measurement/       Layer 1: taxonomy, metrics, span_overlap (deterministic, no LLM)
  retrieval/         Dense (Qdrant/Voyage), sparse (BM25), fusion (CC/RRF), routing
  supervisor/        LangGraph pipeline: nodes (phases 3-10), graph, state, context
  ingestion/         Chunking (fixed-stride, section-aware), embedding pipeline
  evaluation/        Ground-truth adapters (per-corpus), fingerprinting

scripts/
  run_corpus_eval.py         Cross-corpus eval harness (the main entry point)
  judge_answers_v2.py        Layer 2: span-informed answer correctness judge
  judge_faithfulness.py      Layer 2: holistic groundedness / strict citation-precision
  run_section_campaign.py    Multi-corpus A/B campaign runner (sequential, gated)
  build_sac_index.py         SAC indexing pipeline
  tune_cc_alpha.py           CC fusion alpha sweep

docs/
  DECISIONS.md       Decision log with 30 findings and corrections (the evidence trail)
  FourCorpus.md      Four-corpus retrieval sweep results

CLAUDE.md            Full project brief: architecture, hard rules, current state, workflow
```
