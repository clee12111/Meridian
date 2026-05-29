# Meridian — Four-Corpus Retrieval Sweep Summary
**Date:** 2026-05-28
**Config:** SAC + NoRerank + CC fusion + single-shot, all corpora on voyage-4
**Method:** chunk-alpha tuned on 50-query slices, validated on full 194 queries each

---

## HEADLINE RESULT

Meridian's optimized stack beats the published LegalBench-RAG (ZeroEntropy /
Pipitone & Alami) baseline on every corpus with a published baseline, by large
margins. SAC + CC fusion transfers across all four corpora — it is NOT a
ContractNLI-specific result.

```
Corpus        Metric   ZeroEntropy baseline   Meridian   Delta
ContractNLI   P@1      8.84%                  38.08%     +29.2pp
ContractNLI   R@8      50.29%                 80.74%     +30.4pp
MAUD          P@1      2.65%                  22.63%     +20.0pp
MAUD          R@8      6.18%                  69.86%     +63.7pp
PrivacyQA     P@1      n/a                    30.83%     —
PrivacyQA     R@8      n/a                    57.95%     —
CUAD          P@1      n/a                    29.43%     —
CUAD          R@8      n/a                    66.89%     —
```

MAUD is the benchmark's hardest dataset (authors flagged it as most
challenging). The +63.7pp R@8 gain there is the strongest single result.

Framing: this is "our optimized stack (voyage-4 + SAC + CC) vs their baseline
stack (text-embedding-3-large + RCTS, no SAC)." A system-vs-system comparison,
not a controlled single-component ablation.

---

## STAGE 1 — Chunk-alpha sweep (routing OFF, 50-query slices)

```
Corpus        a=0.1      a=0.2      a=0.3      a=0.4      a=0.5     Best
ContractNLI   R@8=.774   R@8=.798*  R@8=.754   R@8=.718   R@8=.682   0.2
PrivacyQA     R@8=.517*  R@8=.487   R@8=.494   R@8=.457   R@8=.445   0.1
CUAD          R@8=.734*  R@8=.701   R@8=.700   R@8=.631   R@8=.654   0.1
MAUD          R@8=.663   R@8=.674*  R@8=.648   R@8=.614   R@8=.539   0.2
```

All corpora prefer dense-heavy alpha (0.1-0.2). Per-dataset alpha confirmed
warranted but the range is narrow. CC fusion (score-preserving) over RRF
holds across all four.

---

## STAGE 3 — Full validation (194 queries each, winning alphas)

```
Metric          ContractNLI   PrivacyQA   CUAD     MAUD
P@1             38.08%        30.83%      29.43%   22.63%
R@8             80.74%        57.95%      66.89%   69.86%
R@1             42.98%        18.35%      30.65%   36.60%
DRM             20.6%         2.1%        0.5%     1.0%
OK              11.3%         24.7%       7.7%     11.9%
Routing recall  80.9%         n/a*        n/a*     n/a*
```
*Routing recall invalid for non-ContractNLI — see KNOWN BUG below.

### Failure taxonomy (full 194 each)
```
           ContractNLI    PrivacyQA    CUAD         MAUD
OK         22 (11.3%)     48 (24.7%)   15 ( 7.7%)   23 (11.9%)
DRM        40 (20.6%)      4 ( 2.1%)    1 ( 0.5%)    2 ( 1.0%)
CBF         2 ( 1.0%)     23 (11.9%)   42 (21.6%)   31 (16.0%)
SGP        11 ( 5.7%)     61 (31.4%)   11 ( 5.7%)   42 (21.6%)
ICR         6 ( 3.1%)     18 ( 9.3%)   15 ( 7.7%)   11 ( 5.7%)
OVR       113 (58.2%)     40 (20.6%)  110 (56.7%)   85 (43.8%)
```

---

## KEY FINDINGS

### 1. SAC + CC transfers across all four corpora
Every corpus: R@8 > 0.57, P@1 > 0.22 on voyage-4. The approach is general,
not ContractNLI-specific. Only the chunk-alpha is corpus-specific (0.1-0.2).

### 2. DRM is ContractNLI-specific
CUAD 0.5%, MAUD 1.0%, PrivacyQA 2.1% vs ContractNLI 20.6%. Document
discrimination is a topically-homogeneous-corpus problem (95 near-identical
NDAs), not a pipeline problem. CUAD (462 diverse contracts), MAUD (150 distinct
mergers), PrivacyQA (7 docs) don't have it.

### 3. CBF is the next real lever on long-document corpora
CUAD 21.6%, MAUD 16.0% CBF (chunk boundary failures) — the answer straddles a
chunk split. Section-aware / conditional-clause-boundary chunking directly
attacks this. Corpus-general, helps long documents most. This is the next
deliberate re-index.

### 4. R@8 = R@16 = R@64 everywhere
Top-8 captures everything the retriever finds; no signal at deeper k. The
50->8 CC fusion cut is not losing recall.

---

## CAVEATS — read before trusting any number here

### A. KNOWN BUG: routing invalid for 3 of 4 corpora
DocumentRouter.__init__ had a frozen default-argument bug (Python evaluates
`index_path = _get_routing_index_path()` once at import, freezing it to
ContractNLI's index). PrivacyQA/CUAD/MAUD routing loaded ContractNLI's 95 NDAs,
producing 0% routing recall — a BUG, not a corpus property. Direct-call test
confirmed routing works when the correct index loads (MAUD golden doc ranked
#1). Routing results for non-ContractNLI corpora are INVALID pending fix +
re-run of Stage 2/3 routing-ON for those three. ContractNLI routing is valid
(it happened to want the frozen index): +5.6pp R@8, routing recall 80.9%.

### B. OVR is mostly measurement artifact, NOT degradation
The high OVR (58% ContractNLI, 57% CUAD, 44% MAUD) looks alarming but ~83% of
OVR is a chunk-boundary measurement artifact (Finding 19) — chunks are coarser
than the LLM's actual cited evidence. OVR queries convert to CORRECT answers at
~70% (Finding 22). The chunk-span taxonomy over-reports OVR. Do not read high
OVR as a quality problem.

### C. These are RETRIEVAL metrics, not answer correctness
The ground truth is the end-to-end judged answer pass rate, which retrieval
metrics overpredict. Answer-correctness judging has been run only on
ContractNLI (45.4% -> 56.7% with the full best config). CUAD/MAUD/PrivacyQA
have NOT been judged for answer correctness. The +64pp R@8 on MAUD is genuine
retrieval improvement but is not yet confirmed to translate to correct answers.
Per-corpus answer judges are the missing next pass (each corpus's ground-truth
answer structure differs and the judge needs per-corpus adaptation).

### D. Stack comparison, not embedding ablation
voyage-4 + SAC + CC vs ZeroEntropy's text-embedding-3-large + RCTS. The gains
come from the whole pipeline. Frame as system-vs-system.

### E. PrivacyQA / CUAD published baselines not yet sourced
ZeroEntropy per-dataset baselines for PrivacyQA and CUAD were not available in
this run. Published table (Pipitone & Alami, naive method) has:
PrivacyQA P@1 7.86 / R@8 32.38; CUAD P@1 9.27 / R@8 40.70 — but confirm against
the RCTS rows (higher than naive) before using as the comparison baseline.

---

## NEXT STEPS (in priority order)

1. Fix the frozen-default routing bug (routing.py:136, change to
   `index_path: Path | None = None`, resolve in body). Audit confirmed this is
   the ONLY instance of the bug in the codebase.
2. Re-run Stage 2/3 routing-ON for CUAD/MAUD/PrivacyQA only (ContractNLI
   stands). Verify per-corpus with a single-query check before the full re-run.
   Expected: routing helps document-naming corpora (CUAD/MAUD quote parties),
   may not help PrivacyQA. Given near-zero DRM on these, routing's marginal
   value is the open question.
3. Per-corpus answer-correctness judge (the actual ground truth) on all four.
   This converts retrieval metrics into end-to-end pass rates.
4. Source the RCTS PrivacyQA/CUAD baselines for the full four-corpus comparison.
5. Next retrieval lever: section-aware / conditional-clause chunking to attack
   CBF on CUAD/MAUD (re-index).
6. Reserved: voyage-4-large for the final headline run once chunking + config
   are locked (do not spend it on a config still in flux).

---

## BUDGET STATUS
voyage-4: ~46.8M used of 200M, ~153M remaining.
voyage-4-large: ~74M reserved for final headline run.
All four corpora indexed on voyage-4 (consistency for clean comparison).
Token estimates corrected this session: real cost ~142 tok/chunk (CUAD),
~391 tok/chunk (MAUD) — earlier ~612 estimate was 4x too high.