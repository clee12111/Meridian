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

---

### Finding 23 — Four-corpus routing (corrected): routing recall is general, routing BENEFIT is conditional on DRM presence

**Bug context:** Stage 2 initially showed 0% routing recall on
CUAD/MAUD/PrivacyQA, read as "routing is ContractNLI-specific."
That was WRONG — a frozen-default-argument bug (DocumentRouter
.__init__ evaluated index_path = _get_routing_index_path() at
import, freezing it to ContractNLI's index). All three corpora
searched ContractNLI's 95 NDAs. Fixed: index_path: Path | None =
None, resolved in body. Audit confirmed this was the only frozen-
default bug in the codebase.

**Corrected results (full 194 queries, per-corpus routing index):**
  Corpus        Route recall   R@8 impact   DRM before->after
  ContractNLI   81%            +3.8pp       20.0% -> 20.6%
  CUAD          100%           +3.2pp        0.5% ->  0.0%
  MAUD          100%           +3.3pp        1.0% ->  0.0%
  PrivacyQA     92-100%        -1.1pp        2.1% ->  1.5%

**The precise finding (recall vs benefit are different):**
Routing RECALL is excellent on all four corpora once the correct
index loads (81-100%) — routing reliably finds the right document
everywhere. Routing BENEFIT is conditional: it helps where document
discrimination is a problem (ContractNLI's near-identical NDAs) and
is neutral-to-slightly-negative where discrimination is already easy
(PrivacyQA, 7 documents, dense already discriminates). The benefit
tracks the DRM rate, a measurable corpus property — not the corpus
identity.

**Correction to prior reading:** "Routing is ContractNLI-specific"
was a bug artifact. The accurate claim: routing is a general
capability whose benefit is proportional to the document-
discrimination difficulty of the corpus.

**OPEN — mechanism of CUAD/MAUD R@8 gain (flagged for verification):**
+3.2/+3.3pp R@8 on corpora with near-zero DRM cannot come from
DRM-elimination alone (can't gain 3pp R@8 by fixing 0.5% DRM).
Likely a recall-concentration effect: hard-filtering to routed
documents pulls more right-document chunks into top-8. To be
confirmed (recon below). "Routing fixed DRM" and "routing
concentrated within-doc recall" are different stories.

**Precludes:** Claiming routing helps universally; claiming it's
ContractNLI-specific. Both are wrong. Benefit proportional to DRM rate.

---

### Finding 24 — Four-corpus answer-correctness validation: routing helps ALL corpora; honest single-judge deltas; always-ON policy

**Method:** Span-informed answer judge (DeepSeek-flash, gives credit
for capturing golden-span information, semantic not string match)
applied UNIFORMLY across all four corpora and both routing states.
Separate, clearly-labeled LLM-judged metric — does not touch the
deterministic span taxonomy.

**Routing comparison (all span-informed judge, full 194 each):**
  Corpus        routing-OFF   routing-ON   Delta
  ContractNLI   62.4%         75.3%        +12.9pp
  PrivacyQA     57.2%         61.9%        +4.7pp
  CUAD          62.4%         63.9%        +1.5pp
  MAUD          65.5%         66.5%        +1.0pp
  Average       61.9%         66.9%        +5.0pp

**Corrects Finding 23's prediction:** Finding 23 (retrieval-level)
predicted routing would be neutral-to-negative on low-DRM corpora
(PrivacyQA -1.1pp R@8). At the ANSWER level, routing helped EVERY
corpus and hurt none (+1.0 to +12.9pp). PrivacyQA's +4.7pp answer
gain despite only 7 documents was the surprise — routing helps even
where dense discrimination was thought sufficient. The magnitude
still tracks DRM (largest gain on highest-DRM ContractNLI), but the
floor is "always helps, never hurts," not "conditional on DRM."

**Always-ON routing policy (domain-agnostic decision):**
The system ships ONE fixed config (cannot detect corpus type at
query time). Always-ON beats always-OFF by +5.0pp average and wins
or ties on all four corpora. Decision: routing ALWAYS-ON, single
dense-heavy chunk-alpha. No corpus detector, no selective routing,
no pre-query filter needed.

**Honest baseline delta (judge held constant — CRITICAL CORRECTION):**
Prior "~15% -> 75%" claims MIXED two judges (affirmative-only baseline
vs span-informed best-config) and overstated the gain. Re-judging the
v1 baseline with the SAME span-informed judge:
  v1 baseline (span-judged):   25.8%
  Best config (span-judged):   75.3%
  Honest delta:                +49.5pp (~3x), judge held constant
The 25.8% -> 75.3% is the defensible number. Do NOT quote the old
~15% baseline alongside span-informed numbers — different judge.
(v2-urlfix scored 19.6% span-judged, lower than v1's 25.8%, because
v2 measures DRM at top-8 vs v1's top-64 — Finding 7, not a regression.)

**Honest limitation — three corpora have no answer baseline:**
PrivacyQA/CUAD/MAUD were never run with the v1 baseline config, so
no answer-delta exists for them — only absolute best-config numbers
(61.9/63.9/66.5%) and the published-retrieval-baseline comparison.
The ContractNLI delta (25.8->75.3) is the only clean answer before/after.

**Discipline note:** Two answer judges exist — affirmative-only (early
session, "did the system affirm YES") and span-informed (final, "did
the answer capture golden-span info"). They are NOT comparable;
span-informed runs ~15-17pp higher by design. All reported deltas
must hold the judge constant. The span-informed judge is the standard
going forward (works for extraction queries; affirmative-only could not).

**DRM-is-the-answer-killer thesis (confirmed at answer level):**
Answer correctness tracks LOW DRM, not high R@8. MAUD (DRM 1.0%,
R@8 0.715) -> 65.5% beats ContractNLI (DRM 20.6%, R@8 0.807) -> 56.7%
routing-off. Document discrimination matters more for answers than
recall. R@8 overpredicts correctness where DRM is present.

**Precludes:** Mixing affirmative-only and span-informed judge
numbers in any comparison. Quoting the ~15% baseline. Selective
per-corpus routing (ship always-ON, one config).

---

### 2026-05-29 — Finding 25: Section-aware chunking helps on CUAD (+3.6pp correctness, attributable)

**Experiment:** Section-aware / conditional-clause chunker (cascading boundary
detector: Article/numbered-section/ALL-CAPS/lettered-subsection patterns, terminal
fallback to fixed-stride on unstructured docs). Char-based offsets, half-open
[start,end), cap ~512 / floor ~200 to keep granularity comparable to the 512/128
fixed baseline. A/B on CUAD, single variable = chunk boundaries, everything else held
(SAC + NoRerank + CC(0.1) + always-on routing + single-shot).

**Offset integrity gate (the load-bearing check):** source_text[start:end] == content
for ALL 75,277 chunks, zero failures. Tier B character-set overlap depends on this;
gate passed before any embedding spent.

**Results (194 queries):**
  Metric          Run A (fixed 512/128)   Run B (section)   Delta
  CBF             43 (22.2%)              33 (17.0%)        -10 (-5.2pp)
  R@8             0.6709                  0.7243            +5.34pp (above noise floor)
  P@1             0.3054                  0.2980            -0.74pp (within noise)
  Correctness     121 (62.4%)            128 (66.0%)       +3.6pp
  INCORRECT       56 (28.9%)             41 (21.1%)        -15

**Attribution (why this is real, not a metric wobble):** Of 18 queries freed from
CBF, 9 went 0/18->9/18 CORRECT (R@8 0.00->>=0.99 — span went from missed to captured).
Honest cost: 8 new CBF introduced (section boundaries moved a previously-captured span
out of window), 3 of which were OVR/CORRECT->lost. Net on changed queries: +9 gained,
-3 lost = +6, plus +1 non-CBF margin shift = +7 overall. The subset ledger reconciles
to the headline delta query-by-query.

**Scope of claim:** Confirmed on CUAD only — the richest-structure, high-CBF corpus
(friendliest case). NOT yet generalization-tested. MAUD is the real test (CBF 15.5% +
20.6% PARTIAL gap). ContractNLI is a negative control (DRM-bound, not CBF-bound —
expect little benefit; a null there CONFIRMS the lever is CBF-specific, not a generic
boundary-nudge artifact). PrivacyQA is the graceful-degradation check (prose -> falls
through to fixed-stride -> should ~= baseline and must not hurt).

**Precludes:** Claiming section-aware chunking generalizes before MAUD/ContractNLI/
PrivacyQA A/Bs run. Quoting the +9 gross win without the -3 regression cost.

---

### 2026-05-29 — CUAD CBF baseline reconciled: 17.5% routing-ON vs 21.6% routing-OFF (NOT a doc error)

**Decision:** The CUAD CBF baseline is ~17.5% routing-ON and ~21.6% routing-OFF. Both
are real eval outputs measuring the same corpus under different routing states.
Routing-ON (17.5%) is the shipped-config anchor (routing is always-ON, Finding 24).
CLAUDE.md updated to state both numbers so the bare "17.5%" can't be misread again.

**Why this was investigated:** Run A of the section A/B reported CBF 22.2% against a
documented baseline cited as 17.5% — a ~4.7pp gap on the CONTROL. Recon confirmed Run
A's config is byte-for-byte identical to the four-corpus sweep config. The "gap"
resolved to two benign causes: (1) 17.5% was the routing-ON figure, 21.6%/22.2% the
routing-OFF range; (2) run-to-run nondeterminism (±3 queries, bidirectional across
CBF<->OVR<->ICR — the signature of noise, not a systematic shift).

**Secondary observation (reproducibility leak, flagged not fixed):** on-disk CUAD eval
files were overwritten by a later re-run and no longer match the documented
FourCorpus.md numbers. Eval outputs need run-traceable filenames (config hash / date)
so numbers can always be traced to the run that produced them. Deferred.

**Precludes:** Treating a control that doesn't reproduce the documented baseline as
automatically a bug — verify config identity and routing state first. Citing CUAD CBF
as a bare single number without the routing state.

---

### 2026-05-29 — Finding 26: Faithfulness added as a Layer-2 metric (validated); it is orthogonal to correctness

**Decision:** Faithfulness (RAGAS definition: fraction of answer claims entailed by
the retrieved context) is added as a second Layer-2 LLM-judged metric, alongside
correctness. It is judged against the FULL retrieved context (all top-8 chunks the
model saw), not the model's own cited_text — because judging a claim against the
snippet the model chose is partly blind to the OK+INCORRECT failure this metric exists
to instrument. It NEVER touches the deterministic span taxonomy (Layer 1). Judge model
pinned (deepseek-v4-flash, temp 0, thinking disabled), held constant across the
campaign, same discipline as the correctness judge.

**Build:** Harness change saves context_chunks (Phase 7 output) per record. New judge
(scripts/judge_faithfulness.py) feeds Phase 8's pre-extracted claims (claim +
cited_chunk_id + cited_text) to the entailment judge — Phase 8's structured output
means RAGAS's claim-extraction step is skipped entirely.

**Validation (gated before scaling):**
- Variance: 5/40 queries scored sub-1.0 (0.50-0.80), mean 0.961 — not rubber-stamping.
- NOT_ENTAILED reasons inspected: all 5 substantively correct, no false negatives.
- Negative control: a fabricated claim ("signed in 1823 by Napoleon...") correctly
  marked NOT_ENTAILED, score dropped 1.0->0.67.
- Diagnostic payoff: contractnli-0411 (OK retrieval + INCORRECT + faithfulness 0.50) —
  the model had right evidence, contradicted it, the judge caught it. Splits
  OK+INCORRECT into "unfaithful (ignored evidence)" vs "faithful but wrong reasoning."

**Two findings about what it measures:**
1. Faithfulness is ORTHOGONAL to correctness. DRM queries score faithfulness 1.0 (the
   model faithfully reports what the WRONG document says). Correctness catches
   DRM/wrong-conclusion; faithfulness catches hallucination/contradiction. Two metrics,
   two failure modes — by design.
2. As built, the judge is partly a CONTRADICTION detector, not pure RAGAS faithfulness
   — most NOT_ENTAILED verdicts flagged claims the context REFUTES, not just claims it
   is silent on. Broader than stock RAGAS (which centers on "unsupported"). Describe it
   as such; do not conflate with textbook RAGAS faithfulness.

**Expected behavior on legal:** runs high (extractive domain — model mostly quotes,
little room to hallucinate). Flat-high on legal is a TRUE finding (low hallucination),
not a dud metric. Faithfulness's real value is the BEIR transfer, where inferential
domains make hallucination/ungrounding the live risk.

**Cross-tab required going forward:** correctness × faithfulness 2x2 per corpus. The
CORRECT+UNFAITHFUL cell measures correctness borrowed against parametric knowledge
rather than retrieval — a transfer-leak signal only visible once faithfulness exists.

**Precludes:** Reading faithfulness as a correctness proxy (they are orthogonal).
Judging faithfulness against cited_text instead of full context. Putting faithfulness
verdicts into the Layer-1 span taxonomy.

---

### 2026-05-29 — RAGAS evaluated as a framework, declined; our judge kept; few-shot lift deferred to pre-BEIR

**Decision:** Do NOT adopt the RAGAS library for the Layer-2 faithfulness/correctness
metrics. Keep our own judge. Lift exactly two things from RAGAS's (Apache-2.0) source
as a deferred PRE-BEIR prompt upgrade: their two entailment few-shot examples,
especially the "unrelated-but-true" calibration case (a claim true in the world but
not supported by the context). Nothing else.

**Why (evaluated their actual source, not from memory):** RAGAS is Apache-2.0, vendorable.
Compared their faithfulness implementation to ours head-to-head:
- Claim decomposition: ours is better — Phase 8 gives atomic claims WITH citations for
  free; RAGAS spends an extra LLM call to re-decompose the raw answer post-hoc into
  citation-less statements.
- Entailment prompt: roughly equivalent. Ours is explicit on paraphrase and
  contradiction; theirs is zero-explicit but carries 2 few-shot examples. Their
  "unrelated-but-true" example is the one genuine edge ours lacks.
- Scoring: identical ratio (entailed/total); ours handles the empty edge better
  (0.0 vs their silently-propagating NaN).
- Determinism: ours pins model + temp 0; RAGAS pins NEITHER by default — stock RAGAS
  has the judge-drift problem we explicitly guard against.
- Dependencies: pip-installing RAGAS pulls langchain-core, datasets, their prompt
  framework — for a metric we've already built and validated.

**The general lesson:** "use the standard framework, don't handroll" was the right
default but wrong in this specific case — borrowing RAGAS's DEFINITION while owning the
implementation is better-fit, pinned, dependency-free, and validated. Adopting the
library would have been a downgrade.

**Sequencing:** few-shot lift is deferred so it does not split the campaign across two
judge versions. Finish four corpora on the validated judge, THEN add the few-shot
examples as a deliberate step, re-run the variance + negative-control gates (plus an
explicit "true-but-unrelated" test), re-judge legal once if numbers move materially,
then run BEIR on the final calibrated judge.

**Precludes:** pip-installing ragas. Changing the faithfulness judge prompt or model
mid-campaign. Running BEIR before the few-shot calibration step.

---

### 2026-05-29 — Finding 27: Section-aware chunking is NOT corpus-general — negative result, do not ship

**Decision:** Section-aware / conditional-clause chunking is rejected as a general
lever. It is corpus-dependent: marginally helpful on CUAD, neutral on ContractNLI,
and HARMFUL on MAUD and PrivacyQA. Do not ship it as a default; do not re-attempt it
as a corpus-general strategy. The right structural answer to the problem it exposed is
hierarchical chunking (next lever).

**Full five-corpus A/B (baseline fixed-stride vs section-aware, both arms fresh+paired,
both Layer-2 metrics, single variable = chunk boundaries):**
  Corpus        R@8 delta    Correctness delta    Verdict
  CUAD          +1.4pp       +5.2pp               marginal help (partly parametric)
  ContractNLI   -1.2pp        0.0pp               neutral (DRM-bound, as predicted)
  MAUD          -22.8pp      -12.8pp              HARMFUL (fragmentation, see Finding 28)
  PrivacyQA     -3.3pp       -7.2pp               HARMFUL (prose, no real sections)

**The mechanism (meta-level):** A chunk serves two opposed jobs — the unit you RETRIEVE
(wants small + pure for sharp embeddings) and the unit you READ (wants large + complete
for sufficient context). Section chunking optimizes the retrieval job at the expense of
the reading job. It wins only where the answer fits inside one section (CUAD's
single-clause commercial-contract answers). It loses where answers span multiple
sections (MAUD merger agreements — the 20.6% PARTIAL gap IS multi-span answers) because
it cuts along the exact seams that separate the pieces of a single answer.

**Controls behaved as predicted (validates the method, not just the result):**
ContractNLI flat (DRM-bound — section chunking can't help a document-discrimination
problem). PrivacyQA harmful (flowing prose, section detector forces bad splits, DRM
8->28). The lever is not a generic boundary-nudge artifact — it does nothing where it
structurally cannot help and damage where structure is absent.

**Precludes:** Shipping section-aware chunking as a default. Re-attempting it as a
corpus-general strategy. Treating "more semantic boundaries" as universally good —
boundary-awareness that splits multi-span evidence is net harmful.

---

### 2026-05-29 — Finding 28: MAUD section-chunking recall collapse is real fragmentation (mandates hierarchical chunking)

**Decision:** The MAUD R@8 -22.8pp collapse under section chunking is a genuine
fragmentation effect, confirmed by chunk-distribution analysis — not a parquet bug.
This is the direct, evidence-backed motivation for hierarchical chunking.

**Evidence (MAUD section parquet vs baseline parquet):**
  - Chunk count: 45,324 -> 145,601 (3.21x more chunks)
  - Median chunk size: 1,800 -> 485 chars; baseline 89.5% over the 512 cap (2048-char
    fixed-stride blocks); section 8.9% under the 200 floor (short section stubs)
  - Mean chunks/doc: 302 -> 971 (3.21x); 100% of MAUD docs hit the section detector,
    zero fixed-stride fallback (merger agreements are heavily structured)

**Mechanism:** Baseline's large 2048-char blocks held whole multi-span merger clauses
intact in one retrievable unit. Section chunking shattered each into 3-4 ~485-char
section-boundary chunks plus sub-floor stubs that embed poorly. The multi-span evidence
still exists but no longer arrives TOGETHER in top-8 — the retriever can't surface all
the fragments of one answer at once. Recall collapses.

**The deceptive faithfulness signal:** MAUD section faithfulness ROSE (49.5%->85.1%)
while correctness FELL (-12.8pp). This is NOT a grounding fix. Shorter chunks give the
model less context to be ungrounded about — it faithfully answers from tight fragments
that lack the full answer. High faithfulness + low recall + low correctness = "model
faithfully reports evidence that doesn't contain the answer." Faithfulness and recall
moved as the same event, opposite directions.

**Implication — hierarchical chunking is the indicated next lever:** retrieve on small
pure children (section chunking's precise retrieval) but feed the PARENT section to the
model (the complete context section chunking destroyed). Separates the retrieve-unit
from the read-unit so neither job is sacrificed. NOTE: hierarchy reintroduces a Tier B
measurement-scope question (child-span vs parent-span = "the retrieved chunk"?) that
MUST be locked before any build — same class as Finding 7's top-8/top-64 scope trap.

**Precludes:** Tuning section-boundary parameters as the MAUD fix (the problem is
fragmentation of multi-span evidence, not boundary placement). Reading a faithfulness
RISE alongside a recall FALL as an improvement.

---

### 2026-05-29 — Finding 29 CORRECTED (superseded twice): MAUD faithfulness collapse was a 6000-char truncation bug, NOT an NLI failure

**History:** Originally logged as "judge ~35% false-negative on legal cross-references."
Corrected after recon to "6000-char truncation bug." Then corrected AGAIN after
re-judging revealed the strict cited-chunk fix was itself a metric-definition change
(see Finding 30). The truncation bug diagnosis stands; the cited-chunk fix was the
wrong remedy. Final fix: holistic full-context judge with truncation limit raised
from 6000 to 20000.

**The real cause:** judge_faithfulness.py truncated context to the first 6000
characters (context_str[:6000], line 101). MAUD's 2048-char chunks produced ~13K
contexts, 100% truncated to ~3.2 of 8 chunks. 87.7% of NOT_ENTAILED verdicts cited
chunks invisible to the judge. Other corpora (~450-char chunks, ~4K context) were
unaffected.

**Fix applied:** truncation limit raised from 6000 to 20000 (DeepSeek-v4-flash
handles 64K+). Holistic full-context mode retained as default per Finding 30.

**Precludes:** Trusting any faithfulness number produced under the 6000-char limit.
Evaluating any large-chunk strategy with the old truncating judge.

---

### 2026-05-29 — Finding 30: Faithfulness metric redesigned — holistic groundedness is the default; strict cited-chunk is a separate citation-precision diagnostic

**Decision:** The standard faithfulness metric going forward is DOMAIN-AGNOSTIC HOLISTIC
GROUNDEDNESS: is the answer/claim supported by the retrieved context AS A WHOLE (full
context, truncation bug fixed). It is the headline grounding number and it means the same
thing on every corpus (legal and, later, inferential/BEIR). The STRICT cited-chunk check
(claim judged against its own cited_chunk_id) is RETAINED but DEMOTED to a separate,
clearly-labeled CITATION-PRECISION diagnostic — valid only where citations are meaningful
(extractive/legal), never the faithfulness headline, never compared across regimes.

**Why holistic is the default (the future-proofing reason):** the strict cited-chunk
judge has a hidden dependency — it assumes a claim maps cleanly to ONE source chunk. That
holds on extractive legal text and collapses on dense/inferential corpora (e.g. "is this
company healthy?" synthesizes across many passages, maps to no single chunk). The strict
judge would read near-zero faithfulness there and a future reader could mistake
domain-mismatch for failure. Holistic groundedness has no such dependency and is the
correct metric where ground-truth citation structure is thin. This is also the
RAGAS/RAG-triad definition — adopting the industry framing is correct HERE, at the
inferential regime, because that metric was designed for exactly these conditions.

**Why citation precision is kept as a diagnostic (the core insight):** a wrong-citation
answer is NOT an incorrect answer. A claim can correctly answer the query (correctness
PASS) while citing the wrong/incomplete chunk (citation-precision FAIL). The true test is
correctness against the query; citation precision is a quality layer below it. Conflating
the two is what made the MAUD analysis misread "model cited loosely" as "model is wrong."
On legal, where a user must be able to follow a citation to the right clause, citation
precision is a REAL and valuable signal — just not a verdict on answer correctness.

**The three-rung regime ladder (metric matches the ground-truth structure the domain
provides):**
  1. Extractive + meaningful citations (legal): holistic groundedness (headline) +
     cited-chunk citation-precision (diagnostic).
  2. Extractive, weak citations: holistic groundedness only.
  3. Inferential / dense (BEIR, finance): holistic groundedness only (RAGAS-style).
Regime is a CONFIGURED property of the eval, set when pointing at a corpus — NOT
auto-detected at runtime (a wrong runtime guess would silently swap the metric). Same
spirit as always-on routing: one fixed choice per deployment.

**What the strict-judge cited-chunk failures actually are (the citation-precision
signal):** three patterns, none of them hallucination — (a) wrong/absent section label
(substance right, "Section 6.02" not in the cited chunk), (b) incomplete paraphrase
(maud-0922: "$2.10 cash" vs chunk's "$2.10 cash AND 0.0228 shares"), (c) cross-chunk
synthesis (claim spans chunks A/B/C, cites only A). On extractive legal text the live
faithfulness risk is MIS-CITATION, not fabrication.

**Action required:** re-judge all four corpora under HOLISTIC groundedness (truncation
fix retained) to produce the canonical faithfulness headline. The strict numbers from the
prior re-judge become the citation-precision diagnostic. (Cheap — re-judge existing files,
no regeneration.)

**Hierarchy prediction (two independent, no-longer-contaminating signals):** hierarchy's
feed-the-parent step should (1) improve citation precision (whole section in one unit ->
fewer cross-chunk citation failures) and (2) hold/improve holistic groundedness. Measured
by two metrics that no longer contaminate each other.

**Precludes:** Using the strict cited-chunk judge as the faithfulness headline. Comparing
strict-judge and holistic numbers in one row. Reading a citation-precision failure as an
answer-correctness failure. Auto-detecting domain regime at runtime. Carrying the
strict-only judge into inferential/BEIR transfer.

---

### 2026-05-29 — Finding 28 & 29 corrections consolidated: MAUD had NO parametric leak; the "deceptive faithfulness rise" was a truncation artifact

**Corrects Findings 28 and 29.** After the faithfulness truncation fix (6000-char cutoff
removed; see prior corrected Finding 29) and re-judging:

**There is NO parametric leak on MAUD.** The entire thread — "44% parametric," then
"~20-25%," then "maud-0019 is the one real fabrication" — was the truncation bug at
progressively higher zoom. maud-0019's "fabricated $5M/$25M thresholds" were REAL TEXT in
truncated chunks. Under the fixed judge, MAUD baseline CORRECT+UNFAITHFUL dropped 58 -> 20,
and the 20 survivors are CITATION-PRECISION failures (wrong section label / incomplete
paraphrase / cross-chunk synthesis), NOT fabrication. No fabricated-from-nothing leak
survives on any corpus. The transfer-risk thesis ("legal correctness borrowed against
parametric memory, will collapse on BEIR") has NO evidence behind it and is withdrawn.

**The MAUD "+35.6pp faithfulness improvement" (Finding 28) is withdrawn.** It measured
truncation-presence (baseline, huge chunks, clipped) vs truncation-absence (section,
small chunks, not clipped). Under the fixed holistic-comparable judge, section faithfulness
is slightly LOWER than baseline, not higher — consistent with section chunking's
fragmentation causing cross-chunk citation failures. Faithfulness and recall both fall
under section chunking; these are the SAME underlying cause (small chunks fragment
multi-span evidence), showing up in two metrics, not two independent findings.

**Section-aware chunking conclusion (Finding 27) STANDS and is reinforced:** corpus-
dependent non-win. The fragmentation that collapsed MAUD recall (-22.8pp) also degrades
citation precision. Not shippable as a general lever. Hierarchical chunking is the
indicated structural answer.

**Meta-lesson (third artifact this session):** a too-dramatic number (MAUD 49.5% faithful)
was theorized as an NLI failure, logged, then found to be a mechanical truncation bug on
recon — and the "confirmed real leak" exemplar was itself the artifact. Verify the
instrument's actual INPUT before theorizing about its reasoning. Three artifacts caught
this session by refusing to trust a clean-looking number (CBF 17.5% reconciliation,
MAUD NLI misdiagnosis, MAUD parametric-leak that wasn't).

**Precludes:** Citing any MAUD parametric-leak rate. Quoting the +35.6pp section
faithfulness delta. Treating section-chunking faithfulness and recall effects as
independent findings.

---

### 2026-05-30 — Finding 33: Cross-reference graph is extractable but addresses the WRONG bottleneck; MAUD synthesis failures are comprehension, not access

**Decision:** Do NOT build the cross-reference graph (or DTGG) as a fix for MAUD's synthesis
failures. The graph is highly extractable but targets information ACCESS, while MAUD's
dominant failure is information COMPREHENSION — the model has the evidence and reasons over
it wrong. Wrong bottleneck.

**The graph IS viable (extraction was never the question):**
- 101,478 cross-reference matches, 677/doc mean. 95.6% of "Section X" refs resolve to a
  section header in the same document. 155 defined terms/doc. A regex extractor would
  produce a rich, production-grade graph.

**Why it would NOT help (the decisive check):** Of 5 sampled MAUD INCORRECT+FAITHFUL
(grounded-but-wrong) queries, 4/5 are pure reasoning errors on evidence ALREADY PRESENT and
ALREADY cross-referenced in context:
- maud-0018: denied a tail provision that was in 28 cross-referenced context chunks.
- maud-0126: misread a specific threshold number.
- maud-0012: failed to extract a detail present in 21 cross-referenced chunks.
- maud-0531: entity confusion (Parent vs Company), not a cross-reference issue.
- Only maud-0130 (1/5) involved a missing cross-reference (a retrieval gap).
The model HAS the cross-references (7-28 per query) and misreads them. Declaring more edges
to a model that already has the cross-referenced text and still answers wrong addresses
access, which is not the failure.

**The chain this completes (what does NOT fix MAUD multi-span/synthesis):**
- Retrieval-unit changes — section chunking (F28) and hierarchy (F31): proven harmful twice.
- Cross-reference graph (this finding): wrong bottleneck — access, not comprehension.
- The agentic loop: re-retrieves, useless when evidence is already present and misread.
What's LEFT and dead-center on the failure: Phase 8 SYNTHESIS REASONING QUALITY — same
evidence in, better reasoning out. Levers: chain-of-thought / structured reasoning prompt,
or a stronger synthesis model, or both.

**Meta (fourth verify-before-build save this session):** the graph was my (advisor's)
predicted multi-span fix. Checked against 5 labeled failures, refuted — the failures are
comprehension, not access. Prior saves: truncation bug, parametric-leak-that-wasn't,
section/hierarchy retrieval-unit dead ends. The pattern holds: confident hypothesis, checked
on labeled cases, redirected before the build.

**Precludes:** Building the cross-reference graph or DTGG to fix synthesis quality. Treating
MAUD's grounded-but-wrong cluster as an information-access problem. Using the loop (re-
retrieval) to fix sufficient-evidence-misread failures.

---

### 2026-05-30 — Finding 34: The "synthesis gap" decomposes into FOUR distinct failure types; rewrite-off is the cheap real win; stronger model is a null

**Decision:** A five-arm sweep (Pro-on-Phase-8, rewrite-off, windowed-parent, SAC-visible,
+ prior CoT) on the 18 labeled MAUD INCORRECT+FAITHFUL failures proved there is NO single
"synthesis gap." It is at least four distinct failures, each touched by a different lever,
and the most-expected lever (stronger model) did the least. Bank rewrite-off; do NOT ship
windowed; log Pro as a null.

**The sweep (18 queries, by-name reading, faithfulness on every arm):**
  Arm              CORRECT  Faith   Note
  Baseline         2        0.940   —
  CoT (prior)      1        0.944   dead lever, no diagnostic flips
  A: Pro synthesis 3        0.952   flipped 2 NON-diagnostic borderlines; flipped ZERO of
                                    the 4 hard diagnostics. Capability is NOT the ceiling.
  B: Rewrite OFF   3        0.950   flipped maud-0012 AND maud-0126 (TWO hard diagnostics);
                                    16/18 queries got different chunks — verified mechanism,
                                    not noise. Faithful.
  C: Windowed parent 4      0.679   highest CORRECT, flipped maud-0018 (marquee case) BUT
                                    faithfulness COLLAPSED 0.94->0.68, CORRECT+UNFAITHFUL=4.
                                    Correct-but-ungrounded — disqualified as built.
  D: SAC visible   2        0.972   maud-0531 (entity confusion) -> PARTIAL only. Best
                                    faithfulness; framing helps grounding, not comprehension.

**The four failure types (the real deliverable — the gap was never one problem):**
1. RETRIEVAL-QUALITY (rewrite degrading chunks): maud-0012, maud-0126. Fixed by rewrite-OFF.
   The single rewrite normalized queries toward generic legal vocabulary and pulled blander,
   harder-to-read chunks. Raw query retrieved chunks the model could read correctly. These
   were retrieval failures MASQUERADING as comprehension.
2. ACCESS-LIMITED (answer needs wider context): maud-0018, maud-0130. Windowed parent flips
   them — but at unacceptable faithfulness cost as built (model infers from wide context
   rather than grounding).
3. ENTITY-CONFUSION (which entity a term refers to): maud-0531. Partially helped by SAC
   document-framing; not cleanly fixed by anything.
4. GENUINE COMPREHENSION RESIDUAL: maud-0684, 0788, 1114, 1452, 1453 — INCORRECT across ALL
   five arms. Not retrieval, not capability, not context-width, not framing. The real hard
   core, now isolated and ~5 cases (down from 18).

**Two findings that update prior decisions:**
- Pro is a NULL on the hard cases — stronger model does not fix misread-present-evidence.
  This KILLS the "frontier model on Phase 8" direction before it was paid for. Capability is
  not the ceiling for these failures.
- Rewrite was "exonerated" as a DRM lever (Finding 12) but is net-NEGATIVE on
  synthesis/retrieval-quality. Finding 12 tested DRM, never synthesis. Rewrite-off is the
  cheap win — turn off a component, fixes 2 hard cases, faithful.

**Windowed parent is the Arm-C trap:** highest correctness, lowest faithfulness. The
faithfulness metric caught exactly what it was built for — correct-but-ungrounded answers
from a model inferring over wide context. Without the metric this reads "4 CORRECT, ship it";
with it, "trades comprehension for hallucination, disqualified." Metric earned its keep again.

**Precludes:** Treating the synthesis gap as one problem. Pursuing a frontier model on Phase 8
to fix the hard comprehension cases (proven null). Shipping windowed parent as-built (faith
collapse). Reading a correctness gain without checking the faithfulness cost.

**Next:** validate rewrite-off at scale (touches shipped config). Test two query-time
successors on the 18 — multi-query retrieval (the principled successor to rewrite-off) and
decomposition (the one query/reasoning-time lever aimed at the comprehension residual, with
faithfulness as the guardrail per the Arm-C lesson). If neither beats plain rewrite-off, ship
rewrite-off and the comprehension residual becomes the final-phase target.
