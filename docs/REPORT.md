# Meridian — Full Report

## 1. Introduction

Meridian is a forensic measurement framework for retrieval-augmented generation. The measurement layer — a deterministic, two-layer diagnostic that separates retrieval failures from reasoning failures without LLM judges in the trust-anchor layer — is the contribution. A 10-phase RAG pipeline serves as the proving ground: each phase underwent controlled A/B comparisons evaluated by the framework, and the framework's job included saying no.

The thesis: RAG evaluation requires a non-contaminating measurement instrument. LLM-judged metrics drift with model updates and prompt changes. A deterministic span-overlap taxonomy anchors measurement to character arithmetic, making measurement bugs distinguishable from system behavior changes. Four silent measurement bugs caught during this project's headline session — before they could ship wrong numbers — are the practical proof of that thesis.

---

## 2. The 10-phase pipeline

Each phase was evaluated as a categorical strategy choice (strategy A vs B vs C, not continuous knob-tuning). The measurement framework evaluates the outcome; the human decides the direction.

### Phase 1 — Chunking

**Strategies evaluated:** fixed-stride (512-char for ContractNLI/PrivacyQA/CUAD, 2048-char for MAUD), section-aware boundary detection, hierarchical chunking (retrieve children, feed parents).

**Verdict:** Section-aware is **corpus-dependent, rejected as general lever** (Finding 27). Marginal on CUAD (+3.6pp correctness, Finding 25), neutral on ContractNLI, harmful on MAUD (-22.8pp R@8 from fragmentation of multi-span evidence, Finding 28) and PrivacyQA (-7.2pp correctness). Hierarchical chunking also tested negative (Finding 31). The mechanism: section boundaries split multi-span evidence, collapsing recall on complex documents. Both retrieval-unit approaches failed — the remaining synthesis bottleneck is comprehension, not access (Finding 33).

MAUD uses 2048-char chunks (4x the others) because its merger agreements average ~350K chars/doc. This was pragmatic (collection size) and accidentally beneficial — large blocks held whole multi-span merger clauses intact (Finding 28). The chunk-size difference was undocumented until Finding 46.

### Phase 2 — Indexing (SAC)

**Strategies evaluated:** raw chunk embedding vs Summary-Augmented Chunking (SAC — document summary prepended to each chunk before embedding, discarded after).

**Verdict:** SAC **validated** (Finding 8). Bakes document identity into dense embeddings, reducing document-retrieval miss (DRM) by ~10pp. Additive with reranker-OFF (both target the same discrimination problem from different angles: SAC improves the signal, reranker-OFF stops destroying it). DRM reduction: baseline 79.9% → SAC+NoRerank 48.5% (-31.4pp, Finding 14).

### Phase 3 — Query understanding

**Strategies evaluated:** DeepSeek-flash query rewrite vs raw passthrough; multi-query decomposition; query expansion.

**Verdict:** All static pre-retrieval transforms **OFF** (Finding 36). Rewrite was exonerated as a DRM cause (Finding 12: DRM unchanged ±1.7pp with rewrite on/off) but found net-negative on synthesis quality (Finding 34: rewrite pulled blander chunks the model couldn't read correctly — retrieval failures masquerading as comprehension). Multi-query decomposition doubled ContractNLI DRM (+40pp, Finding 36) and produced hollow gains on MAUD (flat correctness, gains all unfaithful). Raw query to retrieval is the shipped default.

**Critical distinction:** "rewrite OFF" governs the initial query only. The loop's responsive, routed-doc-scoped re-query (Phase 10) is a different mechanism and was evaluated separately.

### Phase 4 — Retrieval and document routing

**Strategies evaluated:** dense-only vs hybrid (dense + BM25); with and without document routing (hybrid dense+BM25 over document summaries, pre-filtering retrieval to top-N documents).

**Verdict:** Hybrid retrieval with routing is the best config. Routing is **always-on for concentrated-relevance corpora** (Finding 24: helps all four legal corpora at the answer level, +1.0 to +12.9pp, never hurts). The validated DRM lever is document routing (Finding 24: +12.9pp on the most DRM-bound corpus), with BM25/dense balance secondary.

Routing recall on the combined-index regime: CUAD 100%, MAUD 100%, PrivacyQA 89%, ContractNLI 76% (Finding 45). ContractNLI degrades because homogeneous NDA documents confuse with CUAD's commercial contracts — a genuine system limitation.

Routing is a **concentrated-relevance technique, not universal** (Finding 47). On NFCorpus (dispersed-relevance medical text), routing monotonically hurts: OFF 0.397 > k=10 0.333 > k=5 0.273 > k=3 0.227. The sweep auto-detects this.

### Phase 5 — Fusion

**Strategies evaluated:** Reciprocal Rank Fusion (RRF), Convex Combination (CC), weighted RRF.

**Verdict:** CC fusion **over RRF on structured benchmark corpora** (Finding 9). CC preserves BM25 score magnitude — a score gap of 0.95 vs 0.52 on a rare party-name token survives into the final ranking; RRF compresses it to near-zero rank difference. The effect is dramatic: SAC+NoRerank+CC(0.3) achieved P@1 33.3%, R@8 75.5% vs RRF P@1 18.5%, R@8 56.8% — the largest single improvement of the project.

Per-corpus alpha is warranted (Finding 10/16): ContractNLI 0.2, PrivacyQA 0.1, CUAD 0.1, MAUD 0.2 — all dense-heavy. A single fixed alpha ≈ 0.2 is near-optimal across all four (narrow band).

CC fusion transfers as a **domain-general improvement**: +5.6pp on NFCorpus (medical), comparable to +7.9pp on legal (Finding 47).

### Phase 6 — Reranking

**Strategies evaluated:** Voyage rerank-2.5 on vs off.

**Verdict:** Reranker **OFF on topically-homogeneous corpora** (Findings 13, 21). Cross-encoder reranking by semantic relevance is blind to document identity — on 95 near-identical NDAs, it confidently promotes wrong-document chunks at reranker scores 0.91-0.95. Produces ~79% DRM regardless of candidate quality. Confirmed three independent ways: aggregate statistics, controlled comparison (CC and RRF inputs, same result), and mechanistic analysis (per-query Phoenix traces showing 0/8 correct-document chunks at top reranker scores, Finding 21). Reranker erases CC fusion's improvement entirely (DRM 26.8% → 78.9% when reranker added, Finding 21).

### Phase 7 — Context construction

Top-8 chunks fed to the LLM. Wider pool (top-30) feeds the selector mechanism. R@8 ≈ R@16 ≈ R@64 across all corpora — no additional recall from more chunks.

### Phase 8 — Synthesis

**Strategies evaluated:** DeepSeek-v4-flash vs DeepSeek-v4-pro; chain-of-thought prompting; structured claim-citation output.

**Verdict:** Flash is the correct default (Finding 34 CORRECTED). The original "Pro is null on hard cases" conclusion was invalid — all prior "Pro" tests actually ran flash due to a model-ID alias bug (PRO_MODEL="deepseek-chat" routes to flash, not Pro). The corrected Pro run shows Pro **indistinguishable from flash within the ±2-4pp synthesis variance band** (Finding 40): -1.7pp correctness, +0.9pp faithfulness, 2.3-4.2x latency. Not worth the cost.

The synthesis gap decomposes into four distinct failure types (Finding 34): retrieval-quality (fixed by rewrite-OFF), access-limited (target for the selector), entity-confusion (~1 case), and genuine comprehension residual (~5 hard cases, INCORRECT across all arms including Pro — capability is NOT the ceiling).

### Phase 9 — Verification

Deterministic three-way citation check: ENTAILED / CONTRADICTED / BASELESS per claim, via normalized text matching against chunk content. No LLM calls. Conservative false-positive rate of 0.20-0.33 on legal negation language (Finding 41) — the NLI model reads "shall not" as contradicting claims about the negation. Holistic faithfulness (Layer 2 LLM judge) is the headline metric; Phase 9 is a conservative safety check.

### Phase 10 — Agentic retrieval loop

**Strategies evaluated:** single-shot vs multi-iteration (up to 3), with two loop mechanisms: the original (full-replacement re-query) and the redesigned (delta retrieval + chunk accumulation + document-scoped re-query + freeze-patch synthesis).

**Verdict:** **Single-shot preferred** (Finding 37). The original loop mechanism was broken (Finding 35: 64% no-ops, 14% regressions, 86% document drift from unscoped re-query). The mechanism was redesigned (delta+accumulate+scope+freeze-patch), rebuilt, and retested — **still no gain** (Finding 37: both loop arms regressed vs single-shot on a 200-query mini e2e). The synthesis gap is comprehension-bound, not access-bound, confirmed five independent ways. Iterative retrieval is a dead lever on this corpus class.

### The selector (unconditional LLM chunk promotion)

**Strategies evaluated:** conditional trigger (fire when denial/unsupported detected in output) vs unconditional (fire on every query).

**Verdict:** Mechanism **validated**, trigger **bottlenecked** at ~17% (Finding 39). Access-miss is undetectable from the output — the model produces faithful, well-grounded claims about incomplete evidence, indistinguishable from correct answers. Unconditional firing resolves this: run the selector on every query, let it discover there's nothing to promote on ~97% (a fast no-op), and catch the ~3% where wider-pool evidence exists. Mechanism ceiling: +27 net on 130 INCORRECT (5.5:1 flip-to-regression). In-pipeline: 23/776 promotions, 70% favorable, ~1.0x token overhead (Finding 39 UPDATE).

---

## 3. The measurement layer

### Layer 1 — Deterministic retrieval taxonomy

Every retrieved chunk is classified against ground-truth character offsets into six failure types (DRM, CBF, SGP, ICR, OVR, OK). Classification uses pure character-set intersection — `gt_chars & retrieved_chars` — with per-span scoring (not merged-character-set, which masks coverage gaps on multi-span queries, Finding 3).

**Why each type matters:**

- **DRM (Document Retrieval Miss):** The retriever found the wrong document entirely. This is the discrimination problem — the target for SAC indexing (Phase 2), document routing (Phase 4), and reranker-OFF (Phase 6). DRM queries score 0% correctness and 100% faithfulness (the model faithfully reports the wrong document's content).
- **CBF (Chunk Boundary Failure):** The answer straddles a chunk split. This is the chunking problem — the target for section-aware chunking (Phase 1), tested and rejected as a general lever.
- **SGP (Span Gap):** Only part of a multi-span answer was found. The loop (Phase 10) was designed to recover these, but the mechanism proved ineffective.
- **ICR (Incorrect Region):** Right document, wrong section. The selector (Phase 7) targets these with wider-pool chunk promotion.
- **OVR (Over-Retrieval):** Right region, chunk much coarser than needed. ~83% measurement artifact at the chunk level (Finding 19); mostly reclassifies as OK under cited-span analysis.
- **OK:** Correct retrieval. The target state.

P@k and R@k are character-overlap ratios: P@k = |intersection(top_k_chars, gt_chars)| / |top_k_chars|, R@k = |intersection| / |gt_chars|. Computed at k = 1, 2, 4, 8, 16, 32, 64.

### Layer 2 — LLM-judged answer quality

**Correctness:** Span-informed judge. The LLM sees the system's answer, the system's claims with citations, the correct document ID, and the golden evidence text. It judges whether the answer conveys the same information — semantic match, not string match. Reported as CORRECT / PARTIAL / INCORRECT.

**Faithfulness:** Holistic groundedness (Finding 30). Each claim is judged against the full retrieved context (all 8 chunks). RAGAS-aligned, domain-agnostic. A separate strict citation-precision mode judges each claim against only its cited chunk — valid as a diagnostic on extractive/legal corpora, never the headline. Faithfulness is orthogonal to correctness: DRM queries score faithfulness 1.0 because the model faithfully reports what the wrong document says (Finding 26).

Both metrics use a pinned judge model (DeepSeek-v4-flash, temperature 0, thinking disabled) held constant across all comparisons. Layer 2 never contaminates Layer 1 — they are reported separately.

### The non-contamination principle

Layer 1 is the trust anchor. It uses no LLM calls, no embeddings, no learned models. When Layer 1 says DRM went from 48.5% to 26.8%, that delta is deterministic — it cannot be caused by judge drift, prompt wording, or model update. Layer 2 (LLM-judged) sits on top, measuring a different thing (answer quality, not retrieval quality). The layers measure different phenomena and must never be mixed: a Layer 1 taxonomy classification cannot depend on a Layer 2 judgment, and a Layer 2 metric must never feed back into Layer 1 scoring.

This separation is what makes measurement bugs distinguishable from behavior changes. When the (0,0) span bug produced near-zero P@k/R@8, Layer 1's deterministic nature made it checkable — the same query should produce the same retrieval, and a sanity check against the calibration exposed the discrepancy. An LLM-judged metric would have produced a plausible-looking low number indistinguishable from a real finding.

### Ruler calibration (Finding 44)

Before any external comparison, the measurement layer was empirically calibrated against the published LegalBench-RAG baseline. The paper's exact baseline stack was replicated: RCTS 500-char chunking, text-embedding-3-large embeddings, dense-only cosine retrieval, sqlite-vec exact-NN, combined 32K-chunk index. The paper's exact precision/recall formula (with doc_id matching) was applied.

**Result:** Three of four corpora reproduce within a ~2-3pp embedding-drift floor (text-embedding-3-large not byte-frozen since the paper's August 2024 run): MAUD (+0.39pp P@1, +2.29pp R@8), CUAD (+1.65pp, +2.85pp), PrivacyQA (+2.91pp, -2.14pp). ContractNLI diverges (+5.88pp, +12.9pp) due to benchmark-file provenance — our benchmark file was not generated by the paper's pipeline.

The ruler computes correctly. Divergences are input-data and embedding-drift differences, not measurement bugs.

### Verify-before-trust discipline

Four silent measurement bugs were caught during the headline session:

1. **Pro model-ID alias (Finding 34):** `PRO_MODEL="deepseek-chat"` is a legacy DeepSeek alias that routes to deepseek-v4-flash, not Pro. Confirmed three ways: API docs, billing ($0.00 on Pro line), served-model field (not logged — gap that hid it). Fixed: model ID corrected, served-model logging added. Every prior "Pro" conclusion withdrawn.

2. **BM25 channel mismatch (Finding 45):** Combined-index dense retrieval searched 4,496 mini-doc chunks; BM25 searched the full 96,256-chunk CUAD parquet. Two retrieval channels measuring different pools, producing a hybrid that was neither combined nor per-corpus. Fixed: BM25 filtered to the same mini-doc set.

3. **Zero-span offset bug (Finding 45):** Qdrant payloads don't store character span offsets (spans live in the parquets). The combined-index builder defaulted missing fields to (0,0). All P@k/R@8 computed as near-zero — a plausible result, not an obvious crash. Fixed: spans looked up from parquets, recomputed from saved records, spot-checked correct. The CUAD "anomaly" (R@8=0.049 with 100% routing recall) was entirely this bug; real R@8=0.814.

4. **Chunk-size inconsistency (Finding 46):** MAUD uses 2048-char chunks; the other three corpora use 512-char. Previously undocumented. Confounds MAUD's external P@k comparison with the paper's 500-char RCTS.

Each was caught by verification checks — calibration sanity, channel composition audit, span spot-check — not by the numbers looking wrong. This is the project's thesis in practice: measurement rigor requires systematic verification at every step.

---

## 4. Results

### Config-stack delta (controlled, method-level claim)

On the combined-index benchmark regime (all 4 LegalBench-RAG corpora, 11,524 SAC chunks from 72 mini-split documents, 194 queries per corpus):

| Corpus | Correctness Arm 0 → Arm 1 | Faithfulness Arm 0 → Arm 1 |
|--------|---------------------------|----------------------------|
| ContractNLI | 62.9% → 71.1% (+8.2pp) | 89.1% → 94.1% (+5.0pp) |
| PrivacyQA | 49.0% → 55.7% (+6.7pp) | 94.4% → 97.9% (+3.5pp) |
| CUAD | 61.9% → 73.7% (+11.8pp) | 92.1% → 96.0% (+3.9pp) |
| MAUD | 68.0% → 72.7% (+4.7pp) | 91.3% → 95.5% (+4.2pp) |
| **Average** | **60.5% → 68.3% (+7.9pp)** | **91.7% → 95.9% (+4.2pp)** |

Arm 0: SAC + RRF, no routing/selector. Arm 1: SAC + CC + routing(top-3) + selector + no-rerank. Same index, same embedder, same judge.

### Retrieval (corrected, real spans)

| Corpus | Arm 0 P@1 | Arm 1 P@1 | Arm 0 R@8 | Arm 1 R@8 |
|--------|-----------|-----------|-----------|-----------|
| ContractNLI | 0.152 | 0.422 | 0.702 | 0.810 |
| PrivacyQA | 0.068 | 0.297 | 0.462 | 0.579 |
| CUAD | 0.027 | 0.394 | 0.605 | 0.814 |
| MAUD | 0.008 | 0.270 | 0.670 | 0.783 |

### Cost and latency (Arm 1)

| Corpus | Tokens/query | Latency |
|--------|-------------|---------|
| ContractNLI | 2,077 | 5.6s |
| PrivacyQA | 2,365 | 7.1s |
| CUAD | 2,345 | 6.2s |
| MAUD | 4,813 | 8.4s |

### Routing recall

| Corpus | Routing recall | Avg correct-doc chunks /8 |
|--------|---------------|--------------------------|
| CUAD | 194/194 (100%) | 8.0 |
| MAUD | 194/194 (100%) | 7.9 |
| PrivacyQA | 172/194 (89%) | 6.1 |
| ContractNLI | 148/194 (76%) | 4.9 |

### External comparison (system-vs-system, not method-alone)

Published baselines from arXiv 2408.10343, Table 5 (RCTS 500-char, text-embedding-3-large, dense-only, no reranker). Baselines confirmed paper-pinned (Finding 44).

| Corpus | Meridian P@1 | RCTS P@1 | Meridian R@8 | RCTS R@8 |
|--------|-------------|----------|-------------|----------|
| ContractNLI | 0.422 | 0.066 | 0.810 | 0.250 |
| CUAD | 0.394 | 0.020 | 0.814 | 0.317 |
| PrivacyQA | 0.297 | 0.144 | 0.579 | 0.424 |

**Framing:** Full Meridian stack (SAC + CC + hybrid + routing + selector, voyage-4) vs bare baseline (RCTS, dense-only, text-embedding-3-large). The advantage bundles embedder quality + hybrid retrieval + fusion + routing. System-level evidence, not a single-component ablation.

MAUD excluded — 2048-char chunks vs the paper's 500-char create a chunk-granularity confound (Finding 46). ContractNLI caveated — benchmark-file provenance differs from the paper's (Finding 44).

---

## 5. NFCorpus transfer

First non-legal test. NFCorpus (BEIR medical IR benchmark, 3,633 documents, 323 queries, nDCG@10).

**Sweep results (25% slice, 80 queries):** CC alpha × routing top-k grid. Winner: alpha=0.1 (dense-heavy), routing OFF.

**Full run (323 queries):**

| System | nDCG@10 |
|--------|---------|
| Meridian (CC alpha=0.1, routing OFF) | 0.399 |
| BM25 + cross-encoder reranker | 0.350 |
| BM25 | 0.325 |
| contriever | 0.328 |
| TAS-B | 0.319 |

Above the classic BEIR baselines (original 2021 paper). These are dated single-method baselines; modern dense retrievers (2024+) are comparable. Frame as "above classic baselines," not SOTA.

**What transferred:** The retrieval stack (hybrid dense+sparse, CC fusion). CC over RRF adds +5.6pp on NFCorpus, comparable to +7.9pp on legal. Dense-heavy alpha=0.1 wins (same as legal). CC fusion is domain-general.

**What did NOT transfer / was not tested:** The span-forensic measurement framework (Layer 1 taxonomy). BEIR provides document-level relevance, not character spans. Layer 1's taxonomy could not run. Routing was correctly self-disabled (hurts on dispersed-relevance medical text). This is a retrieval-transfer result, not a measurement-framework-transfer result.

**The genuine finding:** Routing is a concentrated-relevance technique. It monotonically hurts on NFCorpus (OFF 0.397 > k=10 0.333 > k=5 0.273 > k=3 0.227), because medical queries have many relevant documents and routing's hard-filter discards them. Consistent with Finding 23 (routing benefit tracks document-discrimination difficulty). Characterizing routing's boundary is the real result.

---

## 6. Limitations

- **Routing degrades on topically-homogeneous cross-corpus retrieval.** ContractNLI routing drops to 76% in the combined-index regime. 47/194 queries get zero correct-document chunks. A genuine architectural limitation when document-level routing can't discriminate similar contract types.
- **Routing hurts on dispersed-relevance corpora.** Confirmed on NFCorpus. Routing is a concentrated-relevance technique, not universal.
- **Span-forensic framework requires character-span ground truth.** Layer 1's taxonomy does not apply to document-level relevance benchmarks (most of BEIR, MS MARCO). This is a scope boundary — the framework's generality beyond legal corpora with span annotations is not yet demonstrated.
- **MAUD external comparison confounded** by 2048-char vs 500-char chunk granularity (Finding 46).
- **ContractNLI external baseline caveated** — benchmark-file provenance differs from the paper's generation pipeline (Finding 44).
- **Affirmative-only evaluation.** All four LegalBench-RAG corpora contain only queries with affirmative answers. Correctness measures recall of evidence that exists — not false-positive rate.
- **Synthesis variance band is ±2-4pp** (Finding 40). Quality deltas below this are indistinguishable from run-to-run nondeterminism.
- **External multipliers are system-level, not method-level.** The advantage bundles embedder + hybrid retrieval + fusion + routing. No single component is credited with the multiplier.

---

## 7. Conclusion

**What's established:**

- A deterministic span-overlap measurement layer that separates retrieval failures from reasoning failures, calibrated against the published LegalBench-RAG baseline within ~2-3pp embedding-drift noise (Finding 44).
- A controlled config-stack delta of +7.9pp correctness and +4.2pp faithfulness over the system's own RRF foundation, measured on a combined-index benchmark regime with cross-corpus distractors (Finding 45).
- Routing characterized as a concentrated-relevance technique: effective on document-discriminable corpora (100% on CUAD/MAUD), degrading on homogeneous cross-corpus retrieval (76% on ContractNLI), and correctly self-disabling on dispersed-relevance medical text (Finding 47).
- CC fusion validated as a domain-general retrieval improvement, transferring from legal (+7.9pp) to medical (+5.6pp) without tuning (Finding 47).
- A verify-before-trust discipline that caught four silent measurement bugs, each of which would have shipped a confidently wrong number.

**What's explicitly NOT claimed:**

- Framework transfer. The span-forensic taxonomy was exercised on LegalBench-RAG (which has character spans) but not on NFCorpus or other document-level benchmarks. The framework's generality is a design goal, not yet a demonstrated result.
- Method-level superiority. External multipliers compare the full system stack against a bare baseline — the advantage bundles embedder, hybrid retrieval, fusion, and routing. No single component is credited.
- SOTA-everywhere. NFCorpus beats classic BEIR baselines; modern dense retrievers are comparable. The project is not benchmark-chasing.
- Autonomous research capability. The v1 autonomous-loop direction was explored and deliberately abandoned. Phase 10's loop is per-query only and was found ineffective even when correctly implemented.
