# Decision log

Every meaningful decision goes here, dated. Format: date, decision, why,
precludes. Append, don't edit history. The engineer updates this for
technical decisions; the advisor updates for design decisions the human
approved.

What to log: design choices between real options, non-obvious bug fixes,
constraints discovered that weren't documented, surprising test results,
advisor requests overriding default behavior.

What NOT to log: routine refactors, formatting, code reading without changes.

---

### 2026-05-27 — v2 slate refresh

**Decision:** Reframed Meridian from v1's autonomous-research-loop direction
to v2's best-in-class-agent + transferable-measurement direction. Docs
fully rewritten (CLAUDE.md folds scope/architecture/workflow into one file).

**Why:** v1's autonomous Proposer was explored across many sessions and
deliberately abandoned. The measurement layer is what's defensible and
transferable; the agent exists to give it something worth measuring.

**Precludes:** Bringing back an LLM that chooses experiments overnight.
Phase 10's agentic loop is per-query only — different beast.

---

### 2026-05-27 — Findings carried as constraints

**Decision:** v1's four findings carry forward as hard rules (above).
F1 (multi-doc classifier bug) is a latent gate on MultiHop. F2 (top_k is
not a DRM lever) constrains experiment design. F3 (per-span scoring) is
already in measurement code and must not regress. F4 (Voyage nondeterminism)
sets the 0.50pp R@8 variance floor.

**Why:** These are empirical results from v1 runs, not design preferences.
Re-litigating them wastes runs.

**Precludes:** Experiments that test top_k as a DRM lever, scoring that
merges character sets across multi-span queries, assertions on raw R@8
across runs.

---

### 2026-05-27 — Embedder locked, re-index deferred

**Decision:** Stay on voyage-4-large for all embedding. Do not switch to
voyage-law-2 until a planned re-index is explicitly approved.

**Why:** Corpus is already indexed at 124M tokens spent. voyage-law-2 would
require a full re-index (~150M tokens) which exceeds its 50M free tier.
Switch deferred until a deliberate re-index is warranted by a chunking
strategy change (e.g. section-aware chunker landing in Stage 2).

**Precludes:** Switching embedders without explicit budget approval and a
planned re-index pass. voyage-4-large stays until that gate is met.

---

### 2026-05-27 — Fingerprint check is a hard rule

**Decision:** The ingestion path must check fingerprint_matches() before
any embedding call. If fingerprint matches, skip embedding entirely and
log a warning. No script or entry point may bypass this check.

**Why:** 124M of 200M free voyage-4-large tokens were consumed by accidental
re-indexing during v1 development. The Qdrant collection already exists and
is valid. Re-embedding costs ~150M tokens and must never happen implicitly.

**Precludes:** Any ingestion script that embeds unconditionally. Re-indexing
requires explicit --force flag and human approval.

---

### 2026-05-27 — Chunking baseline locked at 512/128

**Decision:** Keep fixed-size 512 token / 128 overlap chunking as
the Phase 1-2 baseline. Do not switch to section-aware chunking
until CBF failure rates from the eval harness justify a re-index.

**Why:** The locked baseline (P@1 8.84%, R@8 50.29%) is the
measurement anchor. Changing chunking before running batch eval
loses the ability to attribute improvements. CBF failures in the
taxonomy will signal when boundary cuts are the bottleneck.

**Precludes:** Re-indexing the corpus without CBF data justifying
it. When the switch happens, voyage-law-2 replaces voyage-4-large
simultaneously — one re-index, two upgrades, measured together.

---

### 2026-05-27 — Stage 2 complete, pipeline structurally done

**Decision:** All 10 pipeline phases are implemented and running
end-to-end. Phase 10 conditional loop-back is wired. No further
structural additions until eval harness produces numbers.

**Why:** The pipeline is the proving ground. Measurement is the
product. Adding more sophistication before measurement produces
signal is premature optimization.

**Precludes:** Adding new pipeline phases or swapping
implementations before Stage 3 eval harness runs and produces
comparative numbers.
