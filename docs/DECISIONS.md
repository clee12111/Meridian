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

---

### 2026-05-28 — Session summary: SAC + CC fusion campaign

This session ran a full diagnostic + optimization campaign on
ContractNLI (194 queries). Best config improved P@1 from 8.84%
(v1) to 31.5% and R@8 from 50.4% to 73.7%. Findings below are
ordered by the experiment sequence.

NOTE: Findings 8-11 above were logged individually mid-session.
Findings 12-19 below are the consolidated authoritative versions,
renumbered to avoid collision. Where content overlaps (14=8,
15=9, 16=10, 18=11), the consolidated entry is more complete.

---

### Finding 6 — URL encoding measurement bug (18% false DRM)

**Bug:** ContractNLI ground truth JSON stored some file_paths
URL-encoded (VELCO%20NDA) while the corpus parquet used spaces.
The DRM classifier did exact string comparison, so 28 queries
across 3 documents (VELCO, Evelozcity, Grindrod) were classified
DRM despite the right document being retrieved (dense rank 1 in
most cases).

**Fix:** urllib.parse.unquote() applied in ContractNLIGroundTruth
__init__ and get_doc_id(). 10 of 28 reclassified correctly; 18
were genuine DRM masked by the encoding bug.

**Precludes:** Trusting DRM rates without verifying doc_id format
consistency between ground truth and corpus.

---

### Finding 7 — DRM measurement scope: top-8 vs top-64

**Decision:** V2 measures DRM at top-8 (final context window).
V1 measured at top-64 (full fused candidate set). V2's top-8 is
the operationally meaningful metric — a document outside the
context window cannot contribute to the answer.

**Evidence:** V2 single-shot at v1 config (top_k=32, no reranker,
no loop) showed DRM 61.9% vs v1's ~24.7% (encoding-corrected).
R@8 matched within 0.25pp — retrieval recall equivalent. 72 of
120 V2 DRM queries had the right document retrieved but ranked
9th or lower. The gap is purely measurement scope, confirmed via
the v1 ground_truth_adapter (git show d40daa6): v1 passed all 64
fused candidates to classify(), v2 passes only the top 8.

**Precludes:** Comparing v1 and v2 DRM as equivalent. V1 DRM is
a recall metric (top-64); v2 DRM is a precision metric (top-8).

---

### Finding 12 — Rewriter exonerated as DRM cause

**Experiment:** Full 194-query A/B with Phase 3 rewriting disabled.
  Rewriter ON:  DRM 80.3%
  Rewriter OFF: DRM 82.0% (+1.7pp, within nondeterminism margin)

**Conclusion:** Query rewriting is not the DRM cause. The
discrimination failure is downstream in retrieval ranking, not
query transformation. Rewriter provides slight net positive — keep it.

**Note on query expansion:** Scoped OUT of the project. The DRM
bottleneck is document discrimination on a topically-homogeneous
corpus (indexing/ranking problem), not query insufficiency.
Expansion helps when input is insufficient (production); these
benchmark queries already name the document.

---

### Finding 13 — Reranker harms DRM on topically-homogeneous corpora

**Experiment:** Full 194-query A/B with Voyage rerank-2.5 disabled.
  Reranker ON:  DRM 79.9%, R@8 50.5%, CBF 0%
  Reranker OFF: DRM 59.3%, R@8 45.9%, CBF 12.4%

**Mechanism:** On 95 near-identical NDAs, the cross-encoder scores
topically-perfect wrong-document chunks as highly relevant and
promotes them above right-document chunks in the 50->8 cut. The
reranker is working correctly — semantic relevance scoring is just
orthogonal to document identity. This matches the LegalBench-RAG
paper finding (Cohere v3.0 hurt) and the literature warning to
A/B test rerankers on legal, never assume benefit.

**Tradeoff:** Reranker causes ~40 DRM but fixes ~24 CBF. It does
genuine within-document chunk selection but can't discriminate
between documents. Fix is document-scoped reranking (defer).

---

### Finding 14 — SAC + reranker-OFF is additive

**Experiment:** SAC (Summary-Augmented Chunking) — 150-char
document summary prepended to each chunk before embedding, summary
discarded after (LLM reads clean clause text, only the vector is
influenced). Built scripts/build_sac_index.py, collection
contractnli_sac (3797 points).

  SAC alone (rerank ON):    DRM 76.8% (reranker reverses SAC gain)
  Reranker OFF alone:       DRM 59.3%
  SAC + reranker OFF:       DRM 48.5%, P@1 18.5%, R@8 56.8%

**Mechanism:** SAC improves dense-channel discrimination (~10pp) by
baking document identity into embeddings. The reranker reverses
most of it by re-scoring on semantic relevance. Removing the
reranker preserves SAC's improvement. The effects compound:
SAC + no-reranker beats either alone.

**Precludes:** Using the reranker with SAC on topically-homogeneous
corpora without document-scoping. SAC's published >95%->19% DRM
result (Reuter et al.) was not replicated — likely because their
measurement was at deeper k and their baseline DRM was higher.

---

### Finding 15 — CC fusion over RRF: largest single improvement

**Experiment:** Convex combination fusion replacing RRF k=60.
CC_score = a*norm(BM25) + (1-a)*norm(dense), min-max normalized.

  SAC+NoRerank+RRF:      P@1 18.5%, R@8 56.8%, DRM 48.5%
  SAC+NoRerank+CC(0.3):  P@1 33.3%, R@8 75.5%, DRM 28.9%
  vs v1 baseline:        P@1 +24.5pp, R@8 +25.2pp

**Mechanism:** RRF compresses score gaps into rank positions
(1/(60+rank)), discarding magnitude. BM25 gives party-name matches
(rare tokens, e.g. "Seeed") a large score gap over wrong documents
(0.95 vs 0.52). RRF collapses that to near-zero rank difference.
CC preserves the gap. With SAC-improved dense embeddings providing
slight discrimination, CC's score preservation amplifies it.

**Why CC suits this corpus, not production:** CC works when scores
are calibrated and comparable (single index, same corpus every run)
and one channel has strong magnitude signal (BM25 on rare tokens).
RRF is more robust for noisy production queries with uncalibrated
scores across heterogeneous systems. CC is a benchmark/structured-
corpus optimization, not a universal production default.

**Precludes:** Using RRF as default without A/B testing CC on
structured benchmark corpora. Treating score normalization as
irrelevant to retrieval quality.

---

### Finding 16 — Per-dataset CC a is warranted

**Experiment:** Two-corpus a sweep (ContractNLI SAC + PrivacyQA
baseline, 50 queries each, a in [0.1..0.9]).

  ContractNLI best: a=0.3 (30% BM25 / 70% dense)
  PrivacyQA best:   a=0.1 (10% BM25 / 90% dense)

**Mechanism:** ContractNLI tolerates more BM25 because party names
provide lexical discrimination. PrivacyQA is near-pure-dense —
lay-language queries against varied privacy-policy vocabulary make
semantic similarity dominant. Per-dataset routing materially
outperforms a single global a.

**Built:** core/evaluation/ground_truth_privacyqa.py, collection
privacyqa_baseline (620 points). MAUD and CUAD deferred (budget —
~98M and ~49M Voyage tokens respectively).

---

### Finding 17 — CC beats weighted RRF (score preservation vs rank weighting)

**Experiment:** weighted_rrf (sparse_weight applied to rank scores)
vs cc_fusion (a applied to normalized scores), 50-query slice.

  CC a=0.3:    R@8 0.767, P@1 0.329
  wRRF 0.25:   R@8 0.746, P@1 0.238

**Conclusion:** CC beats best weighted RRF by +2.1pp R@8 and
+9.1pp P@1 at similar DRM. Score preservation matters most for P@1
(top-1 precision benefits from knowing the magnitude gap between
#1 and #2). For R@8, rank position captures most of the signal.
Confirms Bruch et al.: CC is strictly more expressive than weighted
RRF (any weighted RRF config has a rank-equivalent or better CC).

---

### Finding 18 — Loop adds marginal value; single-shot preferred

**Experiment:** Single-shot vs 3-iteration loop on best config.
  Single-shot:  P@1 31.5%, R@8 73.7%, SGP 14.4%, DRM 26.8%, iter 1.00
  With loop:    P@1 33.3%, R@8 75.5%, SGP 10.3%, DRM 28.9%, iter 1.68

**Conclusion:** Loop adds +1.8pp at +68% compute. It helps SGP
(+4.1pp — finds missing spans) but hurts DRM (-2.1pp — refined
queries retrieve from wrong documents). Single-shot captures ~96%
of loop performance at ~60% cost. Single-shot is the preferred
default at current quality.

**Decision:** Loop stays as a simple verification-threshold (0.75)
with no failure-type gating. Rejected coupling loop behavior to
the failure classifier — the classifier is a downstream symptom
that shifts with config (reranker pushed everything to DRM;
removing it pushed everything out). Coupling control flow to a
shifting classification would make a measurement bug
indistinguishable from a behavior bug. Measurement observes; it
does not steer control flow.

---

### Finding 19 — OVR was ~83% measurement artifact (chunk vs cited-span)

**Experiment:** Parallel classification using Phase 8's cited_text
spans (the verbatim sub-span the LLM quoted per claim) instead of
full chunk spans. Deterministic substring search, no new LLM calls.

  OVR (chunk-span):   47 queries (24.2%)
  OVR (cited-span):    4 queries (3.1% of 129 matched)
  39 of 43 extractable OVR queries reclassified (27->OK, 7->SGP,
  3->CBF, 1->ICR)

**Conclusion:** ~83% of OVR was a measurement artifact — chunks
(~512 chars) are coarser than the LLM's actual citations (~150
chars). The LLM cites tightly; the chunk-span metric measured the
whole chunk. Chunk-span measures retrieval-region quality;
cited-span measures evidence-use quality. The gap reveals coarse
chunking, not imprecise evidence use.

**Secondary finding:** Cited-span unmasked +5 CBF and +9 SGP —
cases where the right chunk was retrieved but the LLM cited the
wrong part of it. A Phase 8 citation-quality signal invisible
under chunk-span.

**Domain boundary (critical):** Cited-span works ONLY for
extractive domains (legal). It relies on cited_text being a
verbatim span findable via str.find(). On inferential domains
(finance: "is this company healthy?" has no verbatim span),
extraction fails and the metric collapses. ContractNLI extraction
failure was 7.7% (excluding DRM queries where wrong-doc citations
naturally don't match). On inferential corpora this rate would
spike — and the spike is itself a signal of how extractive the
domain is.

**Decision:** Keep cited-span as a SECONDARY metric on extractive
corpora. Report three numbers: chunk-span (always valid, retrieval
quality), cited-span (valid when extraction succeeds, end-to-end
precision), and extraction-failure-rate (meta-signal of domain
extractiveness). Never make cited-span primary on a corpus where
extraction failure exceeds ~25%.

---

### Best configuration (end of session)

SAC + NoRerank + CC(a=0.3) + single-shot:
  P@1 31.5% (v1: 8.84%), R@8 73.7% (v1: 50.4%)
  DRM 26.8%, OK 28.9% (chunk-span)
  OVR 3.1% under cited-span (was 24.2% chunk-span)

Remaining levers (no re-index):
  - Document-scoped retrieval for residual DRM (26.8%)
  - Document-scoped reranking to recover CBF without DRM cost

Deliberate re-index (paired):
  - Section-aware + conditional-clause boundary chunking (CBF/SGP)
  - Cross-reference graph (E4, CUAD)
  - Defined-term glossary graph (DTGG, MAUD — highest-conviction)

Cross-corpus validation still open: best config tuned on
ContractNLI only. PrivacyQA indexed but not fully evaluated.
MAUD/CUAD deferred on budget. Overfitting risk acknowledged —
validate transferability before further ContractNLI-specific work.

---

### Finding 20 — End-to-end answer correctness validates the campaign; synthesis failures now isolated

**Method:** LLM-judged answer correctness (DeepSeek-flash judge,
separate eval step, NOT in the deterministic measurement layer).
Judged the saved answers from four configs against the implicit
ground truth (ContractNLI is affirmative-only — correct answer
is always YES, the queried provision exists).

**Results — answer correctness tracks retrieval:**
  Config                  OK%     R@8     CORRECT%
  baseline                10.8%   50.5%   14.9%
  SAC (rerank ON)         13.9%   51.4%   19.1%
  SAC+NoRerank (RRF)      16.0%   56.8%   29.4%
  SAC+NoRerank+CC(0.3)    28.9%   73.7%   45.4%

CORRECT% tripled (14.9%->45.4%), closely tracking retrieval gains.
The retrieval optimization translated to genuinely better answers —
not a metric artifact.

**Correctness by retrieval failure type (best config):**
  DRM (52):  0% correct — wrong document = wrong answer, zero
             exceptions across 776 judgments. DRM is an accurate
             end-to-end failure predictor, not pessimistic.
  OK  (56):  70% correct (39/56) — 12 had perfect retrieval but
             WRONG answer. Pure Phase 8 synthesis failures.
  OVR (50):  54% correct — over-retrieval rarely prevents correct
             answers (confirms OVR is mostly chunk-boundary artifact).
  SGP (28):  71% correct — partial span coverage still often correct.

**The newly isolated problem — synthesis failures:**
12 OK+INCORRECT queries had the right document, right chunk, right
spans, but Phase 8 reached the wrong conclusion (answered NO or
"cannot determine" when the cited text supported YES). Examples:
0452, 0851, 0958. These need a Phase 8 prompt/model fix, NOT
retrieval work. Invisible until answer correctness was measured.

**Failure decomposition (best config, 194 queries):**
  ~88 CORRECT (45.4%)
  ~85 INCORRECT: 42 DRM (retrieval), 12 OK+INCORRECT (synthesis),
                 ~31 other failure types
  ~12 PARTIAL, ~9 judge parse errors (4.6% noise floor)

**Limitation:** ContractNLI is affirmative-only. This measures
correctness on provisions that EXIST. It does not test false-
positive rate (correctly saying NO when a provision is absent) —
the benchmark has no negative cases. Real deployment correctness
would differ.

**Decision on LLM judges:** The "no LLM judges" hard rule applies
to the deterministic span-overlap measurement layer (the trust
anchor), NOT to answer-correctness evaluation. Answer correctness
legitimately requires a judge and lives in a separate, clearly-
labeled eval step. The span taxonomy stays deterministic; the
correctness metric is LLM-judged and labeled as such. They are
reported separately and never contaminate each other.

**Next levers, now cleanly separated:**
  - Retrieval: document-scoping for the 42 DRM queries
  - Synthesis: Phase 8 prompt/model fix for the 12 OK+INCORRECT

---

### Finding 21 — Reranker harm is robust to candidate quality (Finding 9 confirmed with Phoenix mechanism)

**Experiment:** SAC + CC(0.3) + reranker ON — the one untested
three-way config. Tests whether CC's cleaner candidate set changes
the reranker's effect.

  SAC+CC+NoRerank:  DRM 26.8%, P@1 31.5%, R@8 73.7%  (best)
  SAC+CC+Rerank:    DRM 78.9%, P@1 11.1%, R@8 52.0%
  SAC+Rerank(noCC): DRM 76.8%  (RRF input, earlier)

**Conclusion:** The reranker produces ~77-79% DRM regardless of
whether fed CC's clean candidates or RRF's noisy ones. CC's
improvement is entirely erased. 36 queries flipped OK->DRM when the
reranker was added. The reranker's failure is orthogonal to
candidate-set quality — it discards the fusion ordering and
re-scores all candidates on semantic relevance from scratch.

**Phoenix mechanistic evidence (per-query, observed not inferred):**
For 3 queries that were OK without the reranker, after reranking
the top 8 were 0/8 correct-document chunks:
- contractnli-0832 (CEII): 8 wrong docs at scores 0.91-0.95
- contractnli-0405 (Motorola): 8 wrong docs at 0.62-0.67
- contractnli-0586 (Inventor-PDE): 8 wrong docs at 0.80-0.86
The right document's chunks were present in CC's fused input but
scored below 8 wrong-document chunks. High reranker confidence
(0.91-0.95) on wrong documents = the signature of a model
optimizing the wrong objective (semantic relevance, orthogonal
to document identity).

**Refined finding:** Cross-encoder reranking by semantic relevance
is fundamentally blind to document identity. On topically-
homogeneous corpora, it confidently promotes wrong-document chunks
that are topically relevant. Better fusion does not help — the
reranker discards fusion ordering. Three independent evidence
lines: aggregate (27%->79%), controlled (same with CC and RRF
input), mechanistic (Phoenix per-query traces).

**Only path to reranker use here:** document-scoped reranking —
filter to the right document first, then rerank within it. The
reranker's within-document chunk selection (CBF benefit) is real;
its cross-document discrimination is absent. Requires entity
extraction / document-scoping at retrieval time.

**Precludes:** Any reranker use on topically-homogeneous corpora
without document-scoping, regardless of fusion method.

---

### Finding 22 — Hybrid document routing: best config, e2e correctness 56.7%, the OVR artifact confirmed harmless

**Config:** SAC + NoRerank + CC(0.3) + hybrid-routing(top-3)
+ single-shot. Routing = standalone SAC summaries as a document
index, scored by CC fusion of dense (summary semantics) + BM25
(filename/abbreviation tokens, NDA-structural stopwords removed).

**Full campaign progression (e2e answer correctness, LLM-judged):**
  baseline            DRM 79.9%  R@8 50.5%  CORRECT 14.9%
  SAC (rerank ON)     DRM 76.8%  R@8 51.4%  CORRECT 19.1%
  SAC+NoRerank(RRF)   DRM 48.5%  R@8 56.8%  CORRECT 29.4%
  SAC+NoRerank+CC     DRM 26.8%  R@8 73.7%  CORRECT 45.4%
  + hybrid routing    DRM 17.5%  R@8 80.6%  CORRECT 56.7%
Routing added +11.3pp correctness — largest single-step gain.
Rescued 19 queries from DRM+INCORRECT to CORRECT.

**Routing mechanism (two-level CC fusion):**
SAC put document identity into chunk vectors but diluted it with
clause content (weak). Routing embeds the SAME summaries STANDALONE
(undiluted document fingerprints) + BM25 on filename tokens to catch
query abbreviations (CEII, SE_NDCA) that dense embeddings can't
bridge to full party names. Same CC insight as chunk fusion
(Findings 11-13), reapplied at document granularity. Routing recall
73.7% (dense only) -> 84.5% (hybrid).

**OVR artifact CONFIRMED harmless (closes Finding 19):**
Routing concentrated retrieval -> OVR rose to 54.6%. But OVR queries
convert at 70% correct (74/106) — HIGHER than OK queries did without
routing. The chunk-span taxonomy's OVR alarm is a measurement
artifact; the ground-truth answer correctness is what matters.
Verified: judge the output, not the diagnostic.

**Correctness by retrieval failure type:**
  DRM 0%, ICR 11%, OK 73%, SGP 88%, OVR 70%, CBF 50%
DRM = wrong doc = 0% correct, no exceptions (confirms Finding 20).

**Remaining failures decompose cleanly:**
  - 30 DRM = routing misses -> routing-alpha tuning + remaining
    abbreviation cases (Seeed, Kenway, ResConnect, FNHA)
  - ICR cluster -> conditional-clause boundary chunking (re-index)
  - ~7 OK+INCORRECT -> Phase 8 synthesis quality

**Principle confirmed:** The measurement layer diagnoses; the e2e
answer-correctness rate is the ground truth. Retrieval metrics
overpredict and the chunk-span taxonomy can false-alarm (OVR);
only the judged output is the observable that matters.
