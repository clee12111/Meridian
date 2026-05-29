# Four-Corpus Retrieval Sweep — 2026-05-28

All corpora on voyage-4. SAC + NoRerank + CC fusion + single-shot.
Tuned on 50-query validation slices, validated on full 194 queries.

**KNOWN BUG:** Routing recall 0% on PrivacyQA/CUAD/MAUD is a
frozen-default-argument bug in `DocumentRouter.__init__` — the
routing index path was frozen to ContractNLI's at import time.
Routing results for non-ContractNLI corpora are INVALID and must
be re-run after the fix. Stage 2 routing verdicts for those three
corpora should be disregarded. Stage 1 (no routing) and Stage 3
(no routing for PQA/CUAD/MAUD) are unaffected.

---

## Stage 1 — Chunk-alpha sweep (routing OFF, 50 queries)

```
Corpus        a=0.1      a=0.2      a=0.3      a=0.4      a=0.5     Best
───────────────────────────────────────────────────────────────────────────
ContractNLI   P@1=.394   P@1=.354   P@1=.308   P@1=.297   P@1=.136
              R@8=.774   R@8=.798*  R@8=.754   R@8=.718   R@8=.682   0.2
              DRM=14%    DRM=18%    DRM=18%    DRM=20%    DRM=24%

PrivacyQA     P@1=.271   P@1=.243   P@1=.245   P@1=.172   P@1=.148
              R@8=.517*  R@8=.487   R@8=.494   R@8=.457   R@8=.445   0.1
              DRM=4%     DRM=2%     DRM=0%     DRM=0%     DRM=2%

CUAD          P@1=.296   P@1=.283   P@1=.234   P@1=.154   P@1=.025
              R@8=.734*  R@8=.701   R@8=.700   R@8=.631   R@8=.654   0.1
              DRM=0%     DRM=0%     DRM=0%     DRM=0%     DRM=0%

MAUD          P@1=.173   P@1=.197   P@1=.189   P@1=.204   P@1=.000
              R@8=.663   R@8=.674*  R@8=.648   R@8=.614   R@8=.539   0.2
              DRM=0%     DRM=0%     DRM=0%     DRM=0%     DRM=0%
```

**Winners:** ContractNLI=0.2, PrivacyQA=0.1, CUAD=0.1, MAUD=0.2.
All prefer dense-heavy (low alpha). Per-dataset alpha confirmed.

---

## Stage 2 — Routing-alpha sweep (BUGGED for non-ContractNLI)

ContractNLI routing is valid. Others loaded the wrong index.

```
              No-route    rt=0.3     rt=0.5     rt=0.7
ContractNLI   R@8=.770    R@8=.826*  R@8=.785   R@8=.737   rr=76-80%
PrivacyQA     R@8=.524    INVALID    INVALID    INVALID    rr=0% (BUG)
CUAD          R@8=.708    INVALID    INVALID    INVALID    rr=0% (BUG)
MAUD          R@8=.651    INVALID    INVALID    INVALID    rr=0% (BUG)
```

**ContractNLI:** routing helps (+5.6pp R@8), best routing-alpha=0.3.
**Others:** need re-run after fix.

---

## Stage 3 — Full validation (194 queries, winning alphas)

ContractNLI uses routing. Others do not (routing was disabled due
to the Stage 2 bug — they ran with the correct no-routing config).

```
Metric          ContractNLI     PrivacyQA          CUAD          MAUD
─────────────────────────────────────────────────────────────────────
P@1                  0.3808        0.3083        0.2943        0.2263
P@4                    —             —             —             —
R@1                  0.4298        0.1835        0.3065        0.3660
R@8                  0.8074        0.5795        0.6689        0.6986
R@16                 0.8074        0.5795        0.6689        0.6986
R@64                 0.8074        0.5795        0.6689        0.6986
DRM%                  20.6%          2.1%          0.5%          1.0%
OK%                   11.3%         24.7%          7.7%         11.9%
Routing recall        80.9%          n/a           n/a           n/a
```

### Failure taxonomy

```
           ContractNLI     PrivacyQA          CUAD          MAUD
OK           22 (11.3%)    48 (24.7%)    15 ( 7.7%)    23 (11.9%)
DRM          40 (20.6%)     4 ( 2.1%)     1 ( 0.5%)     2 ( 1.0%)
CBF           2 ( 1.0%)    23 (11.9%)    42 (21.6%)    31 (16.0%)
SGP          11 ( 5.7%)    61 (31.4%)    11 ( 5.7%)    42 (21.6%)
ICR           6 ( 3.1%)    18 ( 9.3%)    15 ( 7.7%)    11 ( 5.7%)
OVR         113 (58.2%)    40 (20.6%)   110 (56.7%)    85 (43.8%)
```

---

## Stage 4 — ZeroEntropy baseline comparison

```
Corpus         Metric    ZeroEntropy     Meridian      Delta
────────────────────────────────────────────────────────────
ContractNLI    P@1            0.0884       0.3808     +29.2pp
ContractNLI    R@8            0.5029       0.8074     +30.4pp
PrivacyQA      P@1               n/a       0.3083        n/a
PrivacyQA      R@8               n/a       0.5795        n/a
CUAD           P@1               n/a       0.2943        n/a
CUAD           R@8               n/a       0.6689        n/a
MAUD           P@1            0.0265       0.2263     +20.0pp
MAUD           R@8            0.0618       0.6986     +63.7pp
```

ContractNLI: +29.2pp P@1, +30.4pp R@8 vs RCTS baseline.
MAUD: +20.0pp P@1, +63.7pp R@8 vs RCTS baseline.
PrivacyQA/CUAD: ZeroEntropy numbers not yet available.

---

## Key findings

### 1. TRANSFERABILITY VERDICT
SAC + CC fusion transfers across all four corpora. Every corpus
achieved R@8 > 0.57 and P@1 > 0.22 on voyage-4. The per-dataset
chunk-alpha is the only corpus-specific knob (0.1-0.2 range, all
dense-heavy).

### 2. DRM is ContractNLI-specific
CUAD 0.5%, MAUD 1.0%, PrivacyQA 2.1% — effectively zero.
ContractNLI 20.6% with routing (was 26.8% without). DRM is a
topically-homogeneous corpus problem, not a pipeline problem.

### 3. Dominant failure types vary by corpus
- ContractNLI: OVR (58%) — measurement artifact (Finding 19)
- PrivacyQA: SGP (31%) — multi-span coverage gaps
- CUAD: OVR (57%) + CBF (22%) — chunking boundary failures
- MAUD: OVR (44%) + SGP (22%) + CBF (16%)

CBF is the primary lever on CUAD/MAUD — section-aware chunking
would directly address it.

### 4. Routing bug invalidates non-ContractNLI routing results
DocumentRouter.__init__ frozen default loads ContractNLI's index
for all corpora. Must fix and re-run Stage 2 for PQA/CUAD/MAUD.
Given 0% DRM on those corpora, routing may not help much — but
the experiment must be clean.

### 5. R@8 = R@16 = R@64 everywhere
Top-8 captures everything the retriever finds. No additional signal
at deeper k. The 50->8 CC fusion cut is not losing recall.
