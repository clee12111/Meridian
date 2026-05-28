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
until v2 is demonstrably better. Three conditions: (1) agent runs all 10
phases on LegalBench, (2) Tier A measurement produces signal on a non-
annotated corpus (FiQA or NFCorpus), (3) Phase 10 measurably beats single-
shot retrieval with trajectories traced. Until all three: v1 ships, v2 builds.

---

## The architecture

### The agent (Phases 1-10)

Phases 1-2 run once per corpus. Phases 3-10 run per query.

  1. Chunking              — fixed-size / semantic / section-aware / agentic
  2. Indexing              — Qdrant + voyage-4-large + BM25 + HNSW
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

Deterministic. No LLM judges in scoring. Ever.

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

- LegalBench-RAG — Tier A + Tier B, 293k chunks indexed (starting testbed)
- FiQA — Tier A only, validates the transferability bet
- NFCorpus — Tier A only, vocabulary mismatch stress
- MultiHop (HotpotQA) — BLOCKED on Finding 1 fix

---

## Hard rules (never violate)

1. **No LLM judges in scoring.** Measurement is deterministic. The agent
   uses LLMs (rewriter, synthesizer) — those are PART of what's measured.
2. **Per-span scoring, not merged-character-set.** Finding 3 fix; do not regress.
3. **Sub-floor deltas are noise.** R@8 has 0.50pp variance floor (Voyage
   embedding nondeterminism). Never narrate sub-floor changes as improvements.
4. **Dataset filter mandatory on every Qdrant query** (multi-corpus collection).
5. **Do not rebuild legalbench_rag_full** without budget approval (~20M tokens).
6. **No autonomous outer research loop.** Phase 10 is per-query only.
7. **Finding 1 is a hard gate on MultiHop.** Multi-doc classifier bug in
   `taxonomy.py` is latent on current data; must fix before MultiHop.
8. **Finding 2 — top_k is not a DRM lever.** DRM is discrimination-bound;
   real DRM levers are query expansion, BM25/dense ratio, hybrid weighting.

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

**Phase:** v2 — Stage 2 complete. Full 10-phase pipeline running
end-to-end on LegalBench-RAG (ContractNLI corpus).

**What was built in Stage 2:**
- Phase 3: DeepSeek-flash query rewriting
- Phase 4: Qdrant dense + BM25 sparse retrieval (top_k=50 each)
- Phase 5: RRF fusion (top_n=50)
- Phase 6: Voyage rerank-2.5 (50→8 candidates)
- Phase 7: Context construction (top 8 chunks)
- Phase 8: DeepSeek-flash structured synthesis
  (instructor + Pydantic claim-citation pairs)
- Phase 9: Three-way deterministic verification
  (ENTAILED / CONTRADICTED / BASELESS)
- Phase 10: Score-gated agentic loop
  (threshold=0.75, max_iterations=3, loops back to Phase 4)

**Chunking baseline (Phases 1-2):**
Fixed-size 512 tokens, 128 overlap. Pre-indexed as
contractnli_baseline in Qdrant. RagForensics section-aware
chunker transplanted to core/ingestion/chunker.py but not yet
wired. Switch deferred until CBF failure rates from eval harness
justify a deliberate re-index. See docs/DECISIONS.md.

**Resume swap conditions (from CLAUDE.md positioning statement):**
1. Agent runs all 10 phases on LegalBench  ✓ COMPLETE
2. Tier A measurement on non-annotated corpus  — not started
3. Phase 10 measurably beats single-shot  — not measured yet

**Stage 3 — next (in order):**
1. Wire eval harness — connect harness.py to v2 pipeline entry
   point, replacing unresolved run_query import from RagForensics
2. Baseline comparison — run max_iterations=1 (single-shot) vs
   max_iterations=3 (agentic) on ContractNLI query set, compare
   P@1 and R@8 against locked baseline (8.84% / 50.29%)
3. Tier A measurement — port FiQA or NFCorpus testbed, run
   corpus-agnostic metrics (robustness, drift, economic, systems)
4. NLI model for Phase 9 — DeBERTa-MNLI local model, fixes
   CONTRADICTED false positive rate on legal negation patterns
5. Section-aware chunker — swap when CBF is dominant failure type,
   re-index with voyage-law-2 simultaneously (one deliberate pass)
6. Phase 3 query decomposition — builds on rewriting pattern,
   directly attacks SGP failures (43% of queries are multi-span)

**Known issues / open flags:**
- Phase 9 CONTRADICTED: false positive rate on legal negation
  ("shall not") — conservative by design, NLI model fixes in Stage 3
- harness.py line 161: unresolved run_query import from RagForensics,
  blocks batch eval until Stage 3 wiring pass
- state.py fields: iteration/max_iterations not in initial state
  schema — set in run_query.py directly, acceptable for now
- graph.py: loop-back path tested structurally but not yet exercised
  on a real query that scores below 0.75

---

## Decision log

Decisions are recorded in docs/DECISIONS.md. Append new entries there.
Format: date, decision, why, precludes. See that file for full history
and format instructions.