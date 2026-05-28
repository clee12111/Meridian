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

---

### 2026-05-27 — Finding 5 (v2): DRM is discrimination-bound, reproduced and worsened

**Decision:** Document that the v2 pipeline reproduces v1's Finding 2
("DRM is discrimination-bound, not coverage-bound") and appears to
WORSEN it. Treat the 80.3% DRM rate as a real finding, not a
measurement bug. Diagnosis precedes any fix.

**The data (full 194-query ContractNLI run, 193 completed):**
- v2 DRM: 155/193 = 80.3%
- v1 DRM (locked baseline): 76/194 = 39.2%
- DRM roughly doubled despite v2 adding a reranker and query rewriting
- All other failure types collapsed (CBF 18%→1%, OVR 10.8%→2.6%,
  SGP 10.8%→3.6%) because DRM is checked first in the priority chain
  and starves the later branches
- Verification score 0.85 is NOT contradictory: the pipeline produces
  well-cited answers about the WRONG document. ContractNLI queries
  name a specific NDA but ask generic legal questions; the retriever
  matches topical legal language and returns clause-perfect chunks
  from other NDAs.

**Investigation confirmed (recon):**
- NOT a doc_id format bug. The #chunk suffix is stripped correctly
  (cid.split("#")[0]); GT and retrieved doc_ids match format.
  Hypothesis tested and refuted.
- The DRM labels are genuine document-level retrieval failures.
- Verified on contractnli-0134: query asks about Seeed's NDA,
  retriever returned chunks from NSK/Aspiegel/TabunKitchen/ONSemi/
  NCDG/ADVANIDE — none from NDA-Seeed.txt.

**Hypotheses for WHY v2 worsened DRM (ranked, each testable A/B):**
1. Query rewriting (Phase 3) erases discriminating signal —
   normalizes party-specific terms toward generic legal vocabulary,
   strengthening topical match and washing out document identity.
   v1 had no rewriter. LEADING hypothesis. Test: eval with Phase 3
   passthrough vs active.
2. BM25/dense balance buries exact-match document signal — BM25
   catches party names as tokens; if RRF weights dense too heavily,
   the discriminating signal drowns. This is the lever Finding 2
   named directly. Test: shift fusion weighting.
3. Reranker optimizes relevance over discrimination — scores
   query-chunk semantic relevance, ranks topically-perfect
   wrong-document chunks above right-document chunks. Counterintuitive
   (reranker doing its job well makes DRM worse). Test: eval with
   reranker on vs off.
4. Phase 10 loop amplifies but does not cause — 74% of queries were
   single-shot, so the loop cannot produce 80% DRM. All 5 slowest
   queries were iteration-3 DRM (loop tried, failed to fix
   discrimination). Ruled out as primary cause.

**Why this matters:** A more sophisticated pipeline produced WORSE
document discrimination on the metric that matters most. If confirmed,
this is a genuine finding: relevance-optimizing components (rewriter,
reranker) can degrade document discrimination when corpus documents
are topically homogeneous (many similar NDAs). The headline deltas
(P@1 +5.66pp, R@8 +2.11pp) are misleading — they average over a
pipeline answering 80% of queries about the wrong document.

**Precludes:** Trusting the v2 headline metrics until DRM is
diagnosed. Treating top_k as a DRM lever (Finding 2 — confirmed,
top_k=50 did not help). "Fixing" the pipeline before isolating which
component caused the regression via single-variable A/B runs.

**Next step:** A/B run with Phase 3 rewriting disabled (passthrough)
vs enabled, holding everything else constant. If DRM drops with
rewriting off → hypothesis 1 confirmed. If DRM holds at ~80% →
rewriter exonerated, test reranker next.

---

### 2026-05-28 — DRM measurement scope clarified: top-8 not top-64

**Decision:** V2 measures DRM at top-8 (final context window). 
V1 measured DRM at top-64 (full fused candidate set). Both are 
correct measurements of different questions. V2's top-8 DRM is 
the operationally meaningful metric.

**Why:** A document outside the top-8 context window cannot 
contribute to the LLM's answer regardless of retrieval rank. 
The right question is "did the right document reach the LLM" 
not "did the retriever find it somewhere in top-64."

**Evidence:** V2 single-shot (top_k=32, no reranker, no loop) 
shows DRM 61.9% vs V1's ~24.7% (encoding-corrected). R@8 matches 
within 0.25pp — retrieval recall is equivalent. The gap is purely 
the top-8 vs top-64 scope difference. 72 of 120 V2 DRM queries 
had the right document retrieved but ranked 9th or lower.

**Implication for SAC:** SAC must improve top-8 document 
discrimination, not just top-64 recall. Document identity baked 
into embeddings at index time is the mechanism that keeps 
right-document chunks at the top of the ranking through 
RRF fusion and reranking.

**Precludes:** Comparing V1 and V2 DRM rates as equivalent 
measurements. V1 DRM is a recall metric; V2 DRM is a precision 
metric. They measure different things.

---

### 2026-05-28 — Finding 8 (v2): SAC + reranker-OFF is additive, best configuration found

**Results (194 queries, ContractNLI):**
  SAC + reranker OFF:  P@1=18.5%, R@8=56.8%, DRM=48.5%, OK=16.0%
  Baseline (urlfix):   P@1=12.2%, R@8=50.5%, DRM=79.9%, OK=10.8%
  Delta:               P@1+6.3pp, R@8+6.3pp, DRM-31.4pp, OK+5.2pp

**What this proves:**
SAC and reranker-OFF are genuinely additive. SAC improves dense
channel discrimination (~10pp DRM reduction) by baking document
identity into chunk embeddings. Removing the reranker preserves
that improvement (~10pp more) by preventing semantic relevance
scoring from re-promoting topically-identical wrong-document chunks.

**The confirmed finding:**
On a topically-homogeneous legal corpus (95 NDAs), cross-encoder
reranking by semantic relevance worsens document discrimination
because topically-similar wrong-document chunks outscore
right-document chunks on clause-level semantic similarity. SAC
partially mitigates this at the embedding level but the reranker
reverses most of the gain. Optimal configuration omits the reranker.

**Failure modes now visible (unmasked by DRM drop):**
CBF: 0→13 (6.7%), SGP: 7→23 (11.9%), OVR: 6→23 (11.9%)
These were hidden behind DRM in prior runs. Now the targets.

**Next experiments in priority order:**
1. CC fusion (RRF → convex combination) on SAC+NoRerank config
2. Document-scoped reranking (entity extract → filter → rerank)
   to recover CBF/OVR quality without reintroducing DRM
3. Section-aware + conditional-clause chunking re-index to
   attack CBF and SGP directly

**Precludes:**
Using the reranker without document-scoping on topically-
homogeneous corpora. Treating reranker as universally beneficial.

---

### 2026-05-28 — Finding 9: CC fusion over RRF — score preservation produces largest single improvement of session

**Results (194 queries, ContractNLI, SAC+NoRerank):**
  SAC+NoRerank+CC(0.3): P@1=33.3%, R@8=75.5%, DRM=28.9%, OK=28.4%
  SAC+NoRerank+RRF:     P@1=18.5%, R@8=56.8%, DRM=48.5%, OK=16.0%
  vs v1 baseline:       P@1+24.5pp, R@8+25.2pp

**Mechanism:**
RRF compresses score gaps into rank positions before combining
channels. CC fusion preserves score magnitude — a BM25 score gap
of 0.95 vs 0.55 between right/wrong documents survives into the
final ranking. With SAC-improved dense embeddings already
providing slight document discrimination, CC's score preservation
amplifies that signal dramatically.

**CC vs weighted RRF (score preservation vs rank weighting):**
CC a=0.3:    R@8=0.767, P@1=0.329 (50-query slice)
wRRF 0.25:   R@8=0.746, P@1=0.238 (50-query slice)
Gap: R@8 +2.1pp, P@1 +9.1pp in CC's favor.
Score preservation matters most for P@1 (top-1 precision).
For R@8 (top-8 recall) rank position captures most signal.

**Per-dataset a confirmed:**
ContractNLI: a=0.3 (30% BM25 / 70% dense)
PrivacyQA:   a=0.1 (10% BM25 / 90% dense)
Per-dataset routing warranted — corpora prefer different balance.

**New dominant failure mode visible after DRM reduction:**
OVR: 6.7% -> 26.8% (+20.1pp)
OVR was masked by DRM. CC fusion's document discrimination
improvement revealed it. Retrieved chunks are correct document
but 3x larger than GT span. Next target: LLM-as-span-extractor.

**Precludes:**
Using RRF as default fusion without A/B testing CC.
Treating score normalization as irrelevant to retrieval quality.

---

### 2026-05-28 — Finding 10: Per-dataset CC a is warranted

**Decision:** Use per-dataset a for CC fusion, not a single
global value. ContractNLI a=0.3, PrivacyQA a=0.1.

**Why:** 0.2pp R@8 difference between a=0.2 and a=0.3 on
ContractNLI, but 16.4pp R@8 gap between a=0.1 and a=0.5 on
PrivacyQA. Dataset-level heterogeneity is real and measurable.

**Precludes:** Using a=0.5 (equal weight) as a universal default.

---

### 2026-05-28 — Finding 11: Loop adds marginal value on current best config, single-shot preferred

**Comparison (194 queries, SAC+NoRerank+CC(0.3)):**
  Single-shot:  P@1=31.5%, R@8=73.7%, SGP=14.4%, DRM=26.8%
  With loop:    P@1=33.3%, R@8=75.5%, SGP=10.3%, DRM=28.9%
  Loop cost:    +68% compute (1.00->1.68 avg iterations)
  Net gain:     +1.8pp on P@1 and R@8

**What the loop is doing:**
Primarily recovers SGP failures (+4.1pp) -- loop finds missing
spans on multi-span queries. But worsens DRM (-2.1pp) -- refined
queries sometimes retrieve from wrong documents. Single-shot
captures ~96% of loop performance at ~60% of the cost.

**Recommended fix for loop design:**
Only trigger loop for SGP/MISSING_EVIDENCE failures.
Do not loop on DRM failures -- document-scoped retrieval not
query refinement is the fix. Implement failure-type-aware
loop trigger per the typed failure classification pattern.

**Precludes:**
Using the loop unconditionally on all low-scoring queries.
Assuming more iterations always improve results.
