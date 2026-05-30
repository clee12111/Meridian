# CLAUDE.md — Meridian

The whole project, condensed. Read this first, every session.

---

## What Meridian is

A best-in-class agentic retrieval system instrumented by a deterministic,
domain-transferable forensic measurement layer. The agent is the proving
ground; the measurement layer is the product.

**Not** an autonomous research agent. **Not** another production RAG bot.
The autonomous-outer-loop direction from v1 was explored and abandoned —
do not bring it back.

**Positioning:** v1 (RAG Forensics) stays on resume as the shipped project
until v2 is demonstrably better. One gate remains:
  Tier A measurement produces signal on a non-annotated corpus (FiQA or
  NFCorpus) — NOT STARTED, the open transferability item.
(Prior conditions resolved: agent runs all 10 phases on LegalBench ✓
COMPLETE; Phase 10 vs single-shot measured at +1.8pp / +68% compute,
single-shot preferred — Finding 18.)
Until the transfer gate is met: v1 ships, v2 builds.

---

## The architecture

### The agent (Phases 1-10)

Phases 1-2 run once per corpus. Phases 3-10 run per query.

  1. Chunking              — fixed-size / semantic / section-aware / agentic
  2. Indexing              — Qdrant + voyage-4 + BM25 + HNSW (voyage-4-large
                            reserved for final headline run only)
  3. Query Understanding   — rewriting / expansion / decomposition / HyDE
  4. Retrieval             — dense + sparse channels
  5. Fusion                — RRF or convex combination
  6. Reranking             — cross-encoder over top 25-50
  7. Context Construction  — count, order, metadata, compression
  8. Synthesis             — LLM call; structured claim-citation output
  9. Verification          — deterministic citation-traceability
  10. Agentic Retrieval Loop — plan-retrieve-evaluate, PER QUERY

**Phase 10 is the research focus.** "Agentic" means the INNER per-query
loop only — the agent iterates retrieval until evidence is sufficient.
Phase 9's verification provides the deterministic stopping signal.
This is NOT a research-campaign loop. The agent answers ONE question by
retrieving multiple times. It does NOT decide what to test tomorrow.

### The measurement layer (two tiers)

Deterministic. No LLM judges in the deterministic span-overlap layer.

**Tier A — corpus-agnostic (work on any corpus):**
- Robustness — output stability under input perturbation
- Drift — output stability over repeated identical inputs
- Systems perf — latency, throughput, per-phase
- Economic — cost per query, per phase, per correct retrieval
- Trajectory — Phase 10 behavior: iterations, tool calls, termination reasons
- Groundedness (citation-traceable) — from Phase 9 output

**Tier B — span-forensic (where ground-truth spans exist):**
- Per-span taxonomy: DRM / CBF / SGP / ICR / OVR / OK
- Per-step failure attribution along agent trajectory
- Multi-span handling (per-span scoring, not merged — Finding 3)

### Where measurement lives

```
core/measurement/
  tier_a/     robustness, drift, systems, economic, trajectory, groundedness
  tier_b/     taxonomy, per_step, multi_span
```

All pure functions. No I/O. No Langfuse. No LLM calls.

### Dependency direction (bottom-to-top, clean)

```
measurement  ←  agent  ←  retrieval primitives  ←  corpus
```

Measurement never depends on the agent. Agent never bypasses measurement.

### Testbeds

- LegalBench-RAG — all four corpora (ContractNLI, PrivacyQA, CUAD,
  MAUD) indexed on voyage-4, swept, and answer-judged. COMPLETE.
- FiQA — Tier A only, validates the transferability bet
- NFCorpus — Tier A only, vocabulary mismatch stress
- MultiHop (HotpotQA) — BLOCKED on Finding 1 fix

---

## Hard rules (never violate)

1. **No LLM judges in the deterministic measurement layer (two-layer rule).**
   Layer 1 (deterministic, the trust anchor): span-overlap taxonomy (Tier B:
   DRM/CBF/SGP/ICR/OVR/OK) + corpus-agnostic metrics (Tier A). No LLM calls.
   Layer 2 (LLM-judged, separate): correctness (scripts/judge_answers_v2.py)
   + faithfulness (scripts/judge_faithfulness.py — holistic groundedness
   default, --strict for citation-precision diagnostic). Layer 2 never
   contaminates Layer 1. (Finding 20 established the boundary; Finding 30
   formalized the two-regime faithfulness design.)
2. **Per-span scoring, not merged-character-set.** Finding 3 fix; do not regress.
3. **Sub-floor deltas are noise.** R@8 has 0.50pp variance floor (Voyage
   embedding nondeterminism). Never narrate sub-floor changes as improvements.
4. **Dataset filter mandatory on every Qdrant query** (multi-corpus collection).
5. **Do not rebuild legalbench_rag_full** without budget approval (~20M tokens).
6. **No autonomous outer research loop.** Phase 10 is per-query only.
7. **Finding 1 is a hard gate on MultiHop.** Multi-doc classifier bug in
   `taxonomy.py` is latent on current data; must fix before MultiHop.
8. **Finding 2 — top_k is not a DRM lever.** DRM is discrimination-bound;
   the validated DRM lever is document routing (Finding 24, +12.9pp on the
   most DRM-bound corpus), with BM25/dense balance secondary. Query
   expansion was tested and scoped out (Finding 12) — not a DRM lever.
9. **Reranker OFF on topically-homogeneous corpora** without document-scoping.
   Cross-encoder reranking by semantic relevance is blind to document identity;
   produces ~79% DRM regardless of candidate quality. Three evidence lines:
   aggregate, controlled, mechanistic (Phoenix traces). (Findings 13, 21.)
10. **CC fusion over RRF on structured benchmark corpora.** Per-dataset chunk
    α (final sweep): ContractNLI 0.2, PrivacyQA 0.1, CUAD 0.1, MAUD 0.2 —
    all dense-heavy. A single fixed α≈0.2 is near-optimal across all four
    (narrow band). Routing α also per-corpus (CUAD 0.3, MAUD 0.7,
    ContractNLI 0.3). RRF remains the production-robust default for
    uncalibrated heterogeneous systems. (Findings 15, 16.)
11. **Cited-span is a secondary metric**, valid only on extractive corpora
    (where cited_text is verbatim-findable). Never make it primary where
    extraction failure exceeds ~25%. (Finding 19.)
12. **Single-shot preferred.** Loop adds +1.8pp at +68% compute. Use
    single-shot as default; loop only when SGP recovery justifies cost.
    (Finding 18.)
13. **Document routing ALWAYS-ON (domain-agnostic policy).** Routing
    helps all four corpora at the answer level (+1.0 to +12.9pp, never
    hurts); benefit tracks DRM rate (largest on ContractNLI). The system
    ships ONE fixed config and cannot detect corpus type at query time,
    so routing is always-on — no selective routing, no corpus detector.
    The frozen-default routing-index bug (DocumentRouter.__init__) is
    FIXED (index_path=None, resolved in body). (Finding 23, 24.)
14. **Hold the answer judge constant; span-informed is standard.** Two
    judges exist: affirmative-only (early, ~15% baseline) and span-informed
    (final, ~15-17pp higher by design). They are NOT comparable. Never
    quote a delta that mixes them. The honest ContractNLI delta is
    25.8%→75.3% (span-informed both ends). The old ~15%→75% figure mixed
    judges and overstated the gain. Span-informed is the standard going
    forward. (Finding 24.)

---

## Dual-Claude workflow

Three roles. The human bridges the two Claudes.

- **Human (operator, decider)** — holds the goal, decides interpretive
  questions, verifies relays, has final authority.
- **Claude.ai (advisor)** — interprets, frames, makes architectural calls.
  Has project context, NOT the live repo. Writes prompts FOR the engineer.
- **Claude Code (engineer)** — reads, writes, runs. Has live ground truth,
  lacks accumulated project framing. Solves what's asked, surfaces what's missing.

Two failure modes to actively counter:

- **Engineer context starvation** — Claude Code circles a problem 3+ times
  when missing the WHY. Counter: every non-trivial prompt carries the
  constraint/finding/gate inline. State the WHY, not just the task.
- **Advisor stale assumptions** — Claude.ai writes a confident spec from
  design intent that doesn't match code reality. Counter: recon round
  before any invasive build. Engineer reads the actual code and reports;
  advisor builds prompts against the report, not the docs.

### Standard cycle for non-trivial changes

1. Advisor proposes direction with tradeoffs
2. Human decides direction
3. Recon round — engineer READS ONLY and reports back
4. Advisor writes build prompt grounded in the recon
5. Engineer executes and reports (quoted code, executed checks vs reasoned)
6. Advisor verifies the report, flags silent-failure modes
7. Human approves or sends back

### Prompt rules (advisor → engineer)

- Tight, not dense — state goal, constraints, decisions, acceptance checks
- WHY inline — every meaningful instruction has a one-line reason
- Lock real decisions, leave implementation open
- Acceptance checks test the DETERMINISTIC part (winning knobs,
  classifications) — never raw noisy scores
- Don't ask the engineer to make interpretive calls — surface to human instead

### Reporting rules (engineer → advisor)

- Quote real code, not paraphrase
- Distinguish executed checks from reasoned ones — say which
- Flag assumption mismatches explicitly ("pre-existing mismatch found...")
- State what was REMOVED and what was KEPT, especially in surgical edits
- Stop and surface if circling 3+ times — don't write more code hoping it sticks

---

## Current view (update when focus shifts)

**Phase:** v2 — Stage 2 COMPLETE. Four-corpus retrieval + answer
correctness validated end-to-end. All four LegalBench-RAG corpora
indexed on voyage-4, swept, judged.

**Best config:** SAC + NoRerank + CC(per-corpus α) + always-ON
hybrid routing(top-3) + single-shot.

**Answer correctness (span-informed judge, routing-ON):**
  ContractNLI  75.3%   (v1 baseline 25.8%, honest delta +49.5pp)
  PrivacyQA    61.9%
  CUAD         63.9%
  MAUD         66.5%
  Average      66.9%

**Retrieval vs published baselines:**
  ContractNLI  P@1 0.381  R@8 0.807  (RCTS: P@1 0.088, R@8 0.503)
  MAUD         P@1 0.247  R@8 0.732  (RCTS: P@1 0.027, R@8 0.062)
  CUAD         P@1 0.325  R@8 0.701  (no published baseline)
  PrivacyQA    P@1 0.326  R@8 0.588  (no published baseline)

**What was built / validated:**
- Full 10-phase pipeline running end-to-end on all four corpora
- Parallel eval harness (16 workers)
- SAC indexing — scripts/build_sac_index.py, build_corpus_v4.py
- CC fusion (per-dataset α: CNL 0.2, PQA 0.1, CUAD 0.1, MAUD 0.2)
- Hybrid document routing (dense summary + BM25 filename tokens)
- Cited-span measurement (Phase 8 cited_text → taxonomy)
- Span-informed answer judge (scripts/judge_answers_v2.py)
- Faithfulness judge (scripts/judge_faithfulness.py) — holistic
  groundedness default, --strict for citation-precision diagnostic
- Section-aware chunker (core/ingestion/section_chunker.py) +
  build_section_chunks.py — tested across all four corpora, rejected
  as general lever (Finding 27); section collections retained for
  reference ({corpus}_section_sac_v4)
- Shared embed pipeline (core/ingestion/embed_index.py) with
  MERIDIAN_EMBED_WORKERS / MERIDIAN_EMBED_SLEEP env control
- Section A/B campaign runner (scripts/run_section_campaign.py) —
  sequential gated runner for multi-corpus section-chunking A/B
- Phoenix observability (localhost:6006)
- Collections on voyage-4: contractnli_sac_v4 (3797),
  privacyqa_sac_v4 (620), cuad_sac_v4 (96256), maud_sac_v4 (45324)
- Section collections: cuad_section_sac_v4 (75277),
  maud_section_sac_v4 (145601), contractnli_section_sac_v4 (2865),
  privacyqa_section_sac_v4 (464)
- Routing indexes: per-corpus hybrid dense+BM25 (.npz)
- voyage-4 budget: ~47M of 200M spent, ~153M remaining
- voyage-4-large: ~74M remaining, reserved for final headline run

**Chunking baseline (Phases 1-2):**
Fixed-size 2048-char stride (baseline collections). Section-aware
chunker built, wired, tested across all four corpora in a five-corpus
A/B, and REJECTED as a general lever (Finding 27): marginal on CUAD,
neutral on ContractNLI, harmful on MAUD (-22.8pp R@8, fragmentation —
Finding 28) and PrivacyQA (-7.2pp correctness). The mechanism: section
boundaries split multi-span evidence, collapsing recall. Hierarchical
chunking (retrieve tight children, feed parent context) was the
indicated next lever but also tested negative (Finding 31). Both
retrieval-unit approaches failed; the synthesis bottleneck is
comprehension, not access (Finding 33). See Findings 27-28, 31, 33.

**Faithfulness (Layer 2, LLM-judged):**
Holistic groundedness is the canonical default: each claim judged
against the FULL retrieved context (Finding 30). Domain-agnostic,
RAGAS-aligned. Strict cited-chunk mode (--strict flag on
judge_faithfulness.py) is a separate citation-precision diagnostic —
valid on extractive/legal corpora where citations are meaningful,
never the headline.
  Canonical holistic faithfulness (four corpora):
    PrivacyQA    97.2%
    MAUD         96.8%
    CUAD         95.5%
    ContractNLI  92.4%
A 6000-char context truncation bug was found and fixed (Finding 29
corrected). MAUD was 100% truncated under the old judge (~5 of 8
chunks clipped), producing corrupt 49.5% faithfulness. Fixed by
raising truncation limit to 20000. All pre-fix MAUD faithfulness
numbers are invalid.

**Resume swap conditions:**
1. Agent runs all 10 phases on LegalBench  ✓ COMPLETE
2. Tier A measurement on non-annotated corpus  — not started
3. Phase 10 measurably beats single-shot  — measured: +1.8pp at
   +68% compute (Finding 18). Marginal. Single-shot preferred.

**Next (in priority order):**
1. Phase 8 synthesis fix — INCORRECT+FAITHFUL cluster (grounded but
   wrong: right evidence, wrong conclusion). 18-43 queries per corpus
   under holistic judge (CUAD 34, ContractNLI 31, MAUD 18, PrivacyQA 43).
   Material cluster. Now confirmed as THE bottleneck: retrieval-unit
   changes (section chunking F28, hierarchy F31), cross-reference graph
   (F33), and the agentic loop have all been tested/analyzed and do NOT
   address it — the failure is comprehension, not access. Levers:
   chain-of-thought / structured reasoning prompt, or a stronger
   synthesis model, or both.
2. BEIR / non-legal transfer — Tier A measurement on FiQA or
   NFCorpus. The open transferability condition. Holistic faithfulness
   (Finding 30) is the metric designed for this regime.
3. Reasoning-based loop gate (vs current deterministic grounding
   gate) — the agentic-loop frontier piece.
4. Final headline run on voyage-4-large once config locked.

**Known issues / open flags:**
- INCORRECT+FAITHFUL synthesis failures — Phase 8 answers wrong
  despite having correct, grounded evidence (18-43 per corpus).
  The largest remaining error cluster. Reasoning-layer fix needed.
- Phase 9 CONTRADICTED: false positive rate on legal negation
  ("shall not") — conservative by design, NLI model fix deferred
- Cited-span extraction fails ~8% (non-DRM) — LLM paraphrases
  instead of verbatim quoting; tolerable on extractive corpora
- All four corpora are affirmative-only (no negative cases) —
  answer correctness measures recall, not false-positive rate
- Two answer judges exist (affirmative-only, span-informed) —
  do NOT mix numbers across judges (Finding 24)
- Section-aware chunking is a tested NEGATIVE result (Finding 27) —
  do not re-attempt as a general lever
- Hierarchical chunking is a tested NEGATIVE result (Finding 31) —
  retrieval-unit changes do not fix the synthesis bottleneck
- Cross-reference graph / DTGG precluded as synthesis fix (Finding
  33) — MAUD's grounded-but-wrong failures are comprehension, not
  access. The model already has the cross-referenced evidence and
  misreads it.
- Finding 1 (multi-doc classifier) is still a hard gate on MultiHop

**Env flags (for A/B experiments):**
  MERIDIAN_NO_REWRITE    — disable Phase 3 query rewriting
  MERIDIAN_NO_RERANK     — disable Phase 6 reranking
  MERIDIAN_CC_ALPHA      — CC fusion alpha (0.0-1.0); if unset, uses RRF
  MERIDIAN_WRRF_SPARSE   — weighted RRF sparse weight
  MERIDIAN_TOP_K         — override retriever top_k
  MERIDIAN_FUSION_TOP_N  — override fusion top_n
  MERIDIAN_EMBED_MODEL   — Voyage model (default: voyage-4; all v4
                            collections are voyage-4, must match)
  MERIDIAN_ROUTING_TOPK  — document routing top-k (unset = no routing)
  MERIDIAN_ROUTING_ALPHA — routing CC fusion alpha (default: 0.5)
  MERIDIAN_ROUTING_INDEX — routing index path (default: data/routing_index.npz)
  MERIDIAN_EMBED_WORKERS — thread pool for Voyage embed batches (default: 4)
  MERIDIAN_EMBED_SLEEP   — per-batch sleep in seconds (default: 0.0)
  PHOENIX_ENABLED        — enable Phoenix tracing (localhost:6006)

**CLI flags (scripts/run_corpus_eval.py — cross-corpus eval):**
  --corpus NAME          — {contractnli|privacyqa|cuad|maud}
  --chunk-alpha FLOAT    — CC fusion alpha (required)
  --output PATH          — output JSONL file (required)
  --collection STR       — override Qdrant collection (for A/B testing)
  --parquet PATH         — override corpus parquet (must match collection)
  --routing-topk INT     — document routing top-k (unset = no routing)
  --routing-alpha FLOAT  — routing CC fusion alpha
  --workers INT          — parallel workers (default: 8, max: 12)
  --limit INT            — run only first N queries

**CLI flags (scripts/run_eval.py — ContractNLI-only harness):**
  --collection NAME      — Qdrant collection (default: contractnli_baseline)
  --no-rerank            — disable reranker
  --no-rewrite           — disable query rewriter
  --single-shot          — max_iterations=1
  --cc-alpha FLOAT       — CC fusion alpha
  --wrrf-sparse FLOAT    — weighted RRF sparse weight
  --top-k INT            — override retriever top_k
  --fusion-top-n INT     — override fusion top_n
  --workers INT          — parallel workers (max 16, safe for single-shot)
  --limit INT            — run only first N queries
  --ids ID,ID,...        — run specific query IDs
  --routing-topk INT     — document routing top-k (unset = no routing)
  --fresh                — wipe output, start over
  --output PATH          — override output file

---

## Decision log

Decisions are recorded in docs/DECISIONS.md (Findings 1-30 + corrections).
Append new entries there. Format: date, decision, why, precludes. See that
file for full history and format instructions.