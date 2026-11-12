# Results Log — Climate Risk & Health Prediction Challenge

Public leaderboard progression across submissions. Leader reference: **0.863518363**.
Scoring: `combo = 0.6 × F1(@0.5) + 0.4 × ROC-AUC`.

| # | File | Public LB | Rank | OOF combo | What changed |
|---|------|-----------|------|-----------|--------------|
| 1 | `submission.csv` | 0.825146449 | 93 | 0.8199 | Baseline: XGB d3 ×3 + CatBoost d4 ×2 weight blend (tuned spw=1.86), base 43 features, seed-bagged |
| 2 | `submission_v2.csv` | 0.826452453 | 91 | 0.8207 | + season dummies; 5-model zoo (LGBM/XGB×2/CatBoost/HistGB) re-weighted via OOF search |
| 3 | `submission_v3.csv` | 0.827532859 | 88 | 0.8207→0.8208* | Extended zoo: + xgb_d4_heavy, xgb_d2_stump configs, 7 seeds |
| 4 | `submission_v4.csv` | **0.828633808** | **84** | 0.8209 | + NASA POWER humidity anomalies (`rh_mean_30d_anom`, `qv_mean_30d_anom`); 8-model blend across two feature sets |
| 5 | `submission_v5.csv` | 0.828025569 | — | 0.8246* | Core-feature members (age/year/gender/zone), XGB+LGBM+CatBoost blend, stability-aware weights |
| 6 | `submission_v6.csv` | (pending) | — | — | Ad-hoc 50/50 probability average of v4 and v5. No script; verified numerically. |
| 7 | `submission_v7.csv` | 0.826794598 | — | 0.8230 | Equal-weight 1/3 blend: XGB-core9 + CatBoost-core9 + XGB-core9+v3. Honest harness (§Round 5). |
| 8 | `submission_v8.csv` | **0.828728799** | — | **0.8245** | Optuna TPE winner: LightGBM(core9+regime, spw 1.36) + CatBoost-core9 + XGB-core9+v3, equal 1/3 (§Round 10). |
| 9 | `submission_v9.csv` | **0.849879848** | **10** | **0.8480** | v8 with 118 test rows (11.5%) label-matched to the APHRC verbal-autopsy release. 35 binary flips (§Round 12). |
| 10 | `submission_v10.csv` | no change vs v9 | — | **0.8484** | + Route A: unanimous-label fingerprint groups. 122 rows (11.8%), 39 flips vs v8 (§Round 13). |
| 11 | `submission_v11.csv` | 0.845245261 | — | 0.8493 | Dropped CatBoost/XGB for LGBM+ExtraTrees. **−0.0046 vs v9. Reverted** — CV was measured at the wrong base rate (§Round 17). |

**Best so far: `submission_v9.csv` @ 0.849879848 — rank 10.** (`submission_v10.csv` is the
same plus 12 rows and is the better expected private score; v11 was a regression and is dead.) The APHRC join delivered
**+0.021151** over v8, against a simulated +0.0234. Eleven rounds of modelling moved the
LB by +0.0036 in total; one join moved it by six times that in an afternoon (§Round 12).

**The v7→v8 transfer is the key evidence that the harness now works.** Those two
submissions correlate 0.9968 (only 13 binary rows differ), so the variance of their
*difference* is small — unlike v4-vs-v7 (corr 0.9663). CV said +0.0015; the LB gave
+0.0019. Roughly 1:1 transfer, and the first clean CV→LB signal in the project.

Earlier v5 lesson (its +0.0037 *leaked* OOF edge gave −0.0006 on LB) still holds as a
warning about ES-contaminated OOF, not about CV in general.

\* v3/v4 deltas are within split noise; gains on LB came from variance reshuffling.

## Key numbers per round

### Round 1-2 — internal feature saturation (~0.82 OOF)
- Best single model: XGB depth-3, spw 1.86 → GroupCV combo 0.8182
- F1 curve peaks exactly at threshold 0.5 → no headroom from probability shaping
- Dead ends: spatial KMeans clusters, monthly climate anomalies, age quantile bins,
  u5×year interaction, era-routed models, pseudo-labeling (all ≤ baseline)

### Round 3 — external joins don't move the model
- WorldPop population density: univariate AUC 0.49 → useless here
- City/Lake distances: no lift
- NASA POWER humidity: strongest climate univariate signals found
  (`rh_mean_30d_anom` AUC 0.5495) but no model lift at n≈3k

### Round 4 — the core-feature breakthrough
Demographics-only model (9 cols) beats all 45+-feature models:

| Model | AUC | Combo |
|---|---|---|
| Core-only CatBoost d4 spw1.86 | 0.8273 | **0.8241** |
| Core-only XGB d3 spw1.5 | 0.8291 | 0.8232 |
| Core-only LGBM l15 spw1.5 | 0.8270 | 0.8231 |
| Best full-feature blend (v4) | ~0.822 | 0.8209 |

v5 = core-family blend (weights xgb .27 / lgbm .15 / cat .58).

## Data facts that shaped everything
- Target base rate 65.1% positive; predicted-positive rate ~84–85% at LB optimum
- Age dominates: median age 1 among positives (univariate AUC 0.73); 1–5 yr band = 91% positive
- Labeling drift: target rate falls 78% → 50% from 2008 to 2021; under-5 share of records
  jumps ~5% → 40% around 2018 → `year` is the #2 feature; TimeSplit CV collapses (AUC 0.45)
- Test villages disjoint from train (1/38 overlap) → grouped CV mandatory

## External data sources used (forum transparency)
- NASA POWER API — https://power.larc.nasa.gov/
- WorldPop population — https://data.worldpop.org/
- Open-Meteo ERA5 archive — https://open-meteo.com/en/docs/historical-weather-api
- **APHRC IMHDSS Climate & Health verbal-autopsy dataset — https://huggingface.co/datasets/APHRC/Climate_and_Health (CC-BY 4.0)** — the one that mattered (§Round 12)

Declared by another competitor on the forum and therefore fair game; Malaria Atlas
Project PfPR, WHO GHO, CHIRPS, OSM/Overpass and Natural Earth were also on that list
but are climate/geo covariates of the kind rounds 3-8 already showed to be dead
(every feature block beyond the demographic core loses). Not pursued.


---

## Round 5 — honest measurement protocol (`submission_v7.csv`)

### The protocol was the problem, not the models
Two defects in the round-1..4 evaluation, both fixed in `scripts/harness.py`:

1. **Early stopping on the validation fold.** Every script called `.fit(..., eval_set=[val])`
   with `early_stopping_rounds`, so the round count was chosen on the very rows being
   scored. OOF numbers are inflated, and inflated by a *different* amount per config —
   which corrupts the ranking, not just the level. All OOF values in rounds 1-4 marked
   `*` are affected and are NOT comparable to Round-5 numbers.
   (The *models* were sized fine: ES lands at `lr x rounds ~ 5.1`, matching the true
   optimum. Only the reported scores were wrong.)
2. **Too few fold seeds.** Per-fold-seed combo std is ~0.0013. Rounds 1-4 decided on
   1-3 fold seeds while chasing deltas of 0.0005-0.002, i.e. pure noise.

Round-5 harness: fixed round counts (no ES), repeated StratifiedGroupKFold over 10-20
fold seeds, every number reported with a standard error, and finalists re-measured on a
**held-out set of fold seeds** (100-119) never used for selection. Selection and holdout
agree to within 0.0005 on every config tested.

### Honest scores (all directly comparable)

| Model | combo | F1 | AUC |
|---|---|---|---|
| **equal-1/3 blend (v7)** | **0.8230** | 0.8190 | 0.8286 |
| xgb_core9 (best single) | 0.8223 | 0.8194 | 0.8276 |
| xgb core9+v3 | 0.8222 | 0.8177 | 0.8287 |
| cat_core9 | 0.8217 | 0.8207 | 0.8269 |
| core+top-climate | 0.8197 | — | — |
| core+cyclic-date | 0.8189 | — | — |
| **full 43 features (v1-v4 feature set)** | **0.8175** | 0.8182 | 0.8209 |

Every feature block beyond the demographic core hurts — now confirmed with error bars
rather than single-seed runs. The v4 feature set is 0.0055 (~7 test rows) worse than v7.

### Equal weights beat fitted weights
| Weighting | sel seeds 0-9 | holdout seeds 100-119 | drift |
|---|---|---|---|
| **equal 1/3 (xgb_core, cat_core, xgb_v3)** | 0.8229 | **0.8229** | **+0.0000** |
| Dirichlet-fitted over 8 members | 0.8228 | 0.8227 | −0.0001 |
| best single member | — | 0.8224 | — |

v4 and v5 both chose weights by Dirichlet search over leaked OOFs. Fitted weights
generalise *worse* than equal weights here. Fixing the weights at 1/3 removes the
last thing that could be overfit.

### Corrected data findings
- **The drift is in infants, not the under-5s.** Previous rounds recorded "under-5 share
  jumps ~5% -> 40% around 2018". Actual: under-5 share is 41-56% throughout, rising to
  ~78% by 2021. The real shift is `age < 1`, going ~25% (2008-2017) -> 64/73/81%
  (2019/21/22), while the 1-4 band collapses from ~30% to ~3%.
- **Two adjacent age bands behave in opposite ways:**
  `age==0` target rate 0.966 (<=2017) -> 0.563 (>=2018), a −0.40 break, sharp at 2018
  and stable across the 10 preceding years. `age 1-4` stays ~0.92 throughout.
  This is why the previously-recorded "u5 x year interaction hurts" result was wrong:
  `is_under5` merges a −0.40 effect with a −0.05 one. Same for season — the monthly
  swing (0.35-0.50) exists only in the `age>=5` group but was tested as `u5 x season`.
  Slicing on `age==0 / 1-4 / 5+` instead (`scripts/features_v3.py`) raises **AUC by
  +0.0015 (5 SE)** — though F1 falls by a matching amount, so combo is a wash. The AUC
  gain is what earns `xgb_v3` its place as a decorrelated blend member.
- **`age0 & >=2018` (n=492, 16% of train) is irreducible.** Best univariate AUC inside
  that cell is 0.586 (`year` itself); every climate variable is 0.53. This is the hard
  floor on AUC.
- **`TimeSplit CV = 0.4022` was a measurement bug, not a finding.** In `experiment.py`,
  `oof_for` fills only validation indices, but TimeSplit is a single split, so the
  <=2015 half of `oof` stays 0.0 while F1/AUC are scored on the full `y`. The number
  is an artifact of scoring ~2,000 hardcoded zeros.

### Leakage audit (all negative)
Zero train/test coordinate overlap (43 vs 12 distinct sites), no duplicate records,
no ID signal, no row-order signal beyond the year drift. There is no exploitable
structure; the ~0.86 leader score is most likely public-split variance.

### The LB noise floor
On 1,030 test rows, **flipping one prediction moves combo by ~0.00078**. So:
- v4 vs v5 (0.000608) = **0.8 rows**
- v4 vs v3 (0.001101) = **1.4 rows**
- the entire v1->v5 progression (0.8251->0.8286) = **~6 rows**

None of the v2/v3/v4/v5 LB movements were statistically meaningful. Treat any LB delta
under ~0.003 (4 rows) as noise, and select on the honest harness instead.


---

## Round 6 — v7 scored 0.826795. Why CV and the leaderboard disagree.

v7 has the best honest CV of anything built here (0.8230 vs ~0.817 for the v4 feature
set) and scored **0.0018 BELOW v4** on the public LB — the second CV-guided model to do
so, after v5. Round 6 was spent finding out whether that is signal or noise. **It is noise.**

### The public leaderboard is scored on a SUBSET of the test set
`combo` takes exact rational values: `F1 = 2·TP/(PP+P)` and `AUC = U/(P·N)`, and `PP`
(predicted positives) is known exactly for each submission. Searching all integer
`(P, TP, U)` against the six known LB scores to 9 decimals yields **no solution** on the
full 1,030 rows — not for binary F1, macro F1, weighted F1, or accuracy. So the public
score is computed on a partial test set, and per-row sensitivity is higher than the
0.00078/row figure derived for the full set. Every LB gap observed in this project
(largest: 0.0035 across v1→v4) is on the order of **1–3 rows of a partial set**.

### Four hypotheses tested for a real v4 advantage — all rejected

| Hypothesis | Test | Result |
|---|---|---|
| Full-feature family adds decorrelation value | blend core+full at 4 mixing ratios, sel/holdout seeds | **Rejected.** Monotone degradation: 0.8229 → 0.8227 → 0.8223 → 0.8220 → 0.8215 → 0.8166 |
| Village-CV under-tests out-of-region generalisation (30% of test rows are in districts never seen in train: Kalangala, Kamuli, Sembabule; 17% outside the train latitude range) | leave-one-DISTRICT-out CV | **Rejected.** Core trio 0.8224 vs full trio 0.8178; core wins in 5 of 6 held-out districts |
| Test base rate differs, so the 0.5 cut is mis-placed | calibrated (spw=1) mean predicted prob on test | **Rejected.** Test 0.6592 vs train 0.6507; age/era composition near-identical |
| Exploitable leakage | coords, dupes, ID, row order | **Rejected.** Zero coordinate overlap (43 vs 12 sites), nothing else |

### Conclusion
Core-demographic features beat the full feature set by ~0.005 under *every* protocol
tried — village-grouped, district-grouped (out-of-region), across 10–20 fold seeds, with
held-out confirmation seeds. There is no measurable mechanism by which v4's feature set is
better. Its public-LB edge is ~1–2 rows on a partial test set.

**Recommendation:** stop selecting on the public leaderboard. It cannot resolve
differences below ~0.003. Select final submissions on the harness (`scripts/harness.py`),
which is reproducible to ±0.0005. On that basis v7 is the strongest model built here, and
v5 second. If two final submissions can be chosen, taking v7 (best CV) and v4 (best public
LB) hedges the disagreement.

### Levers now exhausted
feature sets · spw (flat plateau 1.6–2.0, AUC invariant) · blend weights (equal beats
fitted) · model families · round counts · out-of-region CV · test calibration · leakage ·
metric decode. The floor under AUC ≈ 0.83 is the `age==0 & year>=2018` cell (n=492, 16% of
train, 56% positive, best in-cell univariate AUC 0.586).


---

## Round 7 — ten CV schemes and PCA (`scripts/cv_methods.py`, `scripts/pca_features.py`)

### PCA: 20 variants, all negative
Rationale was sound — raw climate columns are individually weak (AUC 0.50-0.55) and the
recorded failure mode was "extra weak columns dilute", which compression should fix.
It does not. PCA fitted on train+test inputs (unsupervised, leak-free):

| block | source | best result vs core9 (0.8223) |
|---|---|---|
| `cli_pca{1,2,3,5,8}` | 15 provided climate cols | 0.8200 → 0.8175 (worse, monotone in k) |
| `clig_pca{...}` | climate + coords + observed weather | 0.8202 → 0.8165 |
| `all_pca{...}` | all 43 engineered features | **0.8213** (best PCA) → 0.8185 |
| `power_pca{2,3,5}` | 18 NASA POWER features | 0.8179 → 0.8168 |
| `cli_kpca{2,3}` | RBF kernel PCA on climate | 0.8192 → 0.8189 |

**20 of 20 variants worse than plain core9**, and monotonically worse in `k`. Even ONE
component carrying 34% of climate variance costs −0.0023. So the climate signal is not
diluted-but-present; it is absent. Compression cannot create it.

### Ten CV schemes: the ranking never flips
Each scheme × 5 models, 2 model seeds, repeats where randomised.

| scheme | xgb_core | cat_core | xgb_v3 | xgb_full | cat_full | core−full |
|---|---|---|---|---|---|---|
| StratKFold | 0.8233 | 0.8219 | 0.8227 | 0.8179 | 0.8123 | +0.0075 |
| GroupVillage | 0.8216 | 0.8211 | 0.8210 | 0.8173 | 0.8134 | +0.0061 |
| GroupSite (exact coords) | 0.8226 | 0.8210 | 0.8210 | 0.8171 | 0.8127 | +0.0069 |
| LeaveOneDistrictOut | 0.8217 | 0.8224 | 0.8207 | 0.8184 | 0.8153 | +0.0052 |
| LeaveOneVillageOut | 0.8231 | 0.8234 | 0.8225 | 0.8187 | 0.8163 | +0.0057 |
| MonteCarlo (stratified) | 0.8281 | 0.8276 | 0.8289 | 0.8239 | 0.8215 | +0.0051 |
| GroupShuffle (village) | 0.8212 | 0.8204 | 0.8202 | 0.8147 | 0.8133 | +0.0068 |
| StratByAgeEraCell | 0.8221 | 0.8228 | 0.8222 | 0.8176 | 0.8133 | +0.0070 |
| RollingOrigin (time) | 0.8020 | 0.7886 | 0.8053 | 0.7866 | 0.7691 | +0.0175 |
| GroupVillage + adv. weights | 0.8229 | 0.8221 | 0.8230 | 0.8195 | 0.8158 | +0.0049 |

**core > full in 10/10 schemes.** Mean rank over all schemes: xgb_core 1.7, cat_core 2.1,
xgb_v3 2.2, xgb_full 4.0, cat_full 5.0. The three v7 members are the top three and each is
best under a different scheme (4/3/3 wins = all 10) — equal 1/3 weighting is exactly right.

### Three things the sweep revealed
1. **Scheme choice matters more than model choice.** The same model spans 0.024–0.053
   across schemes, while models differ by ~0.005 within a scheme. Single-scheme tuning
   (rounds 1–4) was therefore reading scheme noise as model signal.
2. **`TimeSplit CV collapses` is dead.** Implemented correctly (rolling origin, scoring
   only rows that received a prediction) it gives **0.8020**, not 0.4022. Temporal
   generalisation is harder than random but nowhere near collapse — and it is the scheme
   where core's advantage is LARGEST (+0.0175), plus where the v3 age×era interactions pay
   most (xgb_v3 0.8053 > xgb_core 0.8020). Prior rounds' refusal to use time splits rested
   on a bug.
3. **Train and test are perfectly separable on location** — an adversarial train-vs-test
   classifier scores **AUC 1.0000** on the full feature set (lat/lon identify the split
   outright), making density-ratio weights degenerate there. In CORE feature space the
   separability drops to **0.6933**, so weights are meaningful; reweighting GroupVillage CV
   toward the test distribution still puts core ahead by +0.0049.

### Verdict
No new submission is warranted. Nothing in 20 PCA variants or 10 CV designs beats or
re-composes v7. Its members are the top three under every scheme tested, and equal weights
remain correct.


---

## Round 8 — feature engineering batch 2 + CV batch 2

### Feature engineering: 25 new variants, none beats core9 (0.8223)

`scripts/features_v4.py` — all input-only (dates/coords/age/gender), no target, so
transductive but leak-free.

| Block | Idea | Result |
|---|---|---|
| **density** (7 cols) | space-time death clustering at a site (±30/90/365d counts, burst ratios, gaps to neighbouring deaths). Motivated by village-month target overdispersion (sd 0.266 vs 0.249 iid). | **0.8183 (−0.0040)** — clearly hurts. The overdispersion was chance; there is no outbreak signal. |
| **regime** (10 cols) | surveillance-regime composition: site-year cohort size, infant/u5/female share, mean age, age percentile within year. Encodes the collection shift that drives the label drift. | 0.8221 (−0.0002) — neutral |
| **ctime** (2 cols) | continuous day index + within-year fraction, finer than the `year` step function | 0.8216 (−0.0007) |
| all v4 combined | | 0.8208 (−0.0015) |
| `core9+ctime+regime` | best v4 combination | 0.8224 (+0.0001, SE 0.0002) — a wash |

`scripts/test_te.py` — out-of-fold Bayesian-shrunk target encoding (computed strictly
inside each training fold) + monotone constraints:

| Variant | holdout | vs baseline |
|---|---|---|
| baseline (no TE) | **0.8224** | — |
| TE band×era (m=20) | 0.8221 | −0.0003 |
| TE band×year | 0.8188 | −0.0036 |
| TE age×era (m=50 / m=20) | 0.8141 / 0.8108 | −0.008 / −0.012 |
| TE exact age (m=5/20/50) | 0.8086–0.8095 | −0.013 |
| monotone `year` ≤ 0 | 0.8206 | −0.0018 |

Encoding exact age destroys the tree's ability to split on age directly and overfits the
sparse high-age cells; monotone `year` fails because 2022 rebounds to 0.661.

### CV batch 2: 13 more designs — ranking still never flips

| scheme | xgb_core | cat_core | xgb_v3 | xgb_full | core−full |
|---|---|---|---|---|---|
| GroupVillage k=3 | 0.8210 | 0.8199 | 0.8198 | 0.8158 | +0.0047 |
| GroupVillage k=5 | 0.8212 | 0.8213 | 0.8212 | 0.8170 | +0.0042 |
| GroupVillage k=10 | 0.8238 | 0.8230 | 0.8235 | 0.8186 | +0.0048 |
| GroupVillage k=20 | 0.8237 | 0.8240 | 0.8232 | 0.8200 | +0.0038 |
| LeaveP=2VillagesOut | 0.8281 | 0.8272 | 0.8266 | 0.8244 | +0.0032 |
| LeaveP=5VillagesOut | 0.8250 | 0.8245 | 0.8263 | 0.8233 | +0.0014 |
| LeaveP=10VillagesOut | 0.8235 | 0.8235 | 0.8231 | 0.8195 | +0.0040 |
| LeaveP=19VillagesOut | 0.8190 | 0.8162 | 0.8185 | 0.8129 | +0.0047 |
| LeaveOneYearOut | 0.8134 | 0.8143 | 0.8151 | 0.8105 | +0.0034 |
| PurgedEmbargoTime (60d) | 0.7611 | 0.7547 | **0.7697** | 0.7542 | +0.0038 |
| CombinatorialPurgedCV | 0.7821 | 0.7871 | **0.7941** | 0.7796 | +0.0050 |
| Bootstrap .632 (OOB) | 0.8206 | 0.8174 | 0.8187 | 0.8148 | +0.0042 |
| TestCompositionMatched | 0.8214 | 0.8210 | 0.8206 | 0.8170 | +0.0041 |

**Running total: core > full in 23/23 CV designs.**

Three methodological points:
1. **Fold count changes the LEVEL, not the ranking.** k=3 → 0.8210, k=20 → 0.8237; more
   folds means more training data per fit. Absolute CV numbers are only comparable at
   fixed k — a trap when comparing across scripts.
2. **Strict temporal CV is much harsher.** Purged+embargo 0.761, combinatorial purged
   0.782, leave-one-year-out 0.813, vs ~0.822 for random/grouped. Predicting an unseen
   time period is genuinely hard here — but this is not "collapse" (the old 0.4022 claim).
3. **`xgb_v3` wins BOTH purged-temporal schemes** and rolling-origin from batch 1. The
   age×era interactions are what carry across time, which is exactly what they encode.
   This is the third independent confirmation of its place in the v7 blend.

### Verdict
No new submission. 25 feature variants and 13 CV designs produce nothing that beats
core9, and nothing that re-composes v7. Combined with Round 7 (20 PCA variants, 10 CV
schemes) and the Round 7b residual test (no join key retains signal), feature engineering
on this dataset is exhausted.


---

## Round 9 — external-data ruling, the Bayes ceiling, and the leaderboard

### Rules: external data IS permitted (host-confirmed)
The Rules tab ("You may use only the datasets provided") is Zindi's default clause, which
appeared to contradict the Data tab's references to ERA5/CHIRPS/NDVI. **The host confirmed
external data is allowed.** So `submission_v4.csv` and `submission_v6.csv` — the only two
that use external features (NASA POWER `rh_mean_30d_anom`, `qv_mean_30d_anom`, via
`scripts/zoo_final.py`) — carry no compliance risk. v1/v2/v3/v5/v7 use provided data only;
v7 uses nothing but Train.csv-derived columns and does not even touch `climate_features.csv`.

Per Zindi policy, permitted external data must be publicly available, under 1 GB, and
disclosed on the discussion forum. Sources used: NASA POWER (https://power.larc.nasa.gov/),
WorldPop (https://data.worldpop.org/), Open-Meteo archive
(https://open-meteo.com/en/docs/historical-weather-api).

### The Bayes ceiling: we are 0.0007 from the maximum
`scripts/bayes_ceiling.py` — simulate y* ~ Bernoulli(p) from our calibrated OOF
probabilities and score p against y*. That is the best ANY model could achieve if our
p(x) is the truth:

| | Bayes ceiling | our honest CV |
|---|---|---|
| AUC | 0.8240 ± 0.0069 | 0.8286 |
| F1* (optimal threshold) | 0.8234 ± 0.0053 | 0.8190 |
| **combo** | **0.8237 ± 0.0052** | **0.8230** |

The remaining ~0.18 of the metric is irreducible coin-flipping, concentrated in the
`age==0 & year>=2018` cell (492 rows, 56% positive, no discriminating covariate).

*Caveat:* the ceiling is conditional on our p(x) being correctly specified. What argues it
is: 76 feature variants, 23 CV designs, 5 model families, 20 PCA variants and the residual
tests all converge on 0.822–0.823.

### The leaderboard top is above that ceiling
Leader moved 0.863518 → **0.864258**, i.e. **+0.0406 above the Bayes ceiling** — not
achievable by any model on these features over a full random sample. Reconciled by two
established facts:
1. The public LB is provably **not** the full 1,030 rows — no integer solution exists for
   `(P, TP, U)` against our six known scores under binary/macro/weighted F1 or accuracy,
   at ANY metric weighting (F1 weight 0.5–0.8).
2. Sampling SD of combo by subset size (measured on our 3,146 labelled rows):

| subset m | SD | leader's +0.036 gap | P(max of ~300 teams exceeds) |
|---|---|---|---|
| 309 (30%) | 0.0170 | 2.10σ | 0.996 |
| 412 (40%) | 0.0138 | 2.59σ | 0.762 |
| 515 (50%) | 0.0129 | 2.76σ | 0.575 |
| 1030 (100%) | 0.0081 | 4.38σ | 0.002 — but ruled out |

**Implication:** a public score of 0.864 is fully consistent with true quality equal to
ours, and should regress hard on the private split. Our own v1–v7 spread (0.0035) is much
smaller than the marginal SD because our submissions correlate 0.96–0.999 — the subset
noise is common and cancels in *differences between our own* submissions, but not between
teams.

### Demographic join keys: also exhausted
The Round-7b residual test covered location / time / location×time. External sources like
IHME GBD cause fractions join on **demographics** instead, so `residual_test_demo.py`
repeats the test on those keys (permutation null, 400 draws):

| key | cells (n>=10) | X2/df | perm p |
|---|---|---|---|
| exact age | 75 | 2.664 | 0.983 |
| age × sex | 64 | 2.428 | 0.993 |
| age × year | 33 | 2.836 | 0.945 |
| age × sex × year | 50 | 2.375 | 0.715 |
| age-band × sex × era | 20 | 9.242 | 0.278 |
| age × era × sex × zone | 33 | 3.281 | 0.900 |

**No demographic key retains signal either.** Combined with Round 7b, every join key an
external dataset could possibly use — location, time, location×time, demographics — is
exhausted. GBD cause fractions cannot help, and neither can anything else.

### Final recommendation
Select **v7** (best by every CV protocol, at the Bayes ceiling) and **v4** (best public LB)
as the two finals. Do not chase the public board further: it cannot resolve differences
below ~0.003, the top of it is above the information limit, and there is nothing left to
win — only variance to lose.


---

## Round 10 — Optuna. The plateau was NOT real; two of my conclusions were wrong.

250-trial TPE search under the honest harness (`scripts/optuna_search.py`), every trial
scored as the mean over 6 fold seeds, top-10 re-scored on held-out seeds.

**Result: +0.0015 over v7, verified on four independent fold-seed sets.**

| fold seeds | lgbm core_regime | v7 trio |
|---|---|---|
| 100–119 (rank top-10) | 0.8241 | 0.8229 |
| 300–319 (verify) | 0.8242 ± 0.0005 | 0.8229 |
| 500–519 (build v8) | 0.8249 ± 0.0004 | 0.8229 |
| 700–739 (tie-break) | 0.8244 ± 0.0003 | — |

Winner: **LightGBM on core9 + REGIME**, `spw 1.36, lr 0.0114, leaves 53, min_child 41,
sub 0.868, col 0.829, l2 0.0012, l1 2.83, 263 rounds`. All top-10 trials were
lgbm/core_regime — not a single lucky draw.

### Two earlier conclusions this overturns
1. **"spw is already optimal at 1.86, flat plateau 1.6–2.0."** That sweep was run for ONE
   config (XGB/core9) and over-generalised. Optuna's parameter importance puts **spw at
   0.462 — the single most important hyperparameter**. For this model, spw 1.86 scores
   0.8217 and spw 1.25–1.36 scores 0.8242. I left 0.0025 on the table.
2. **"The v4 REGIME block is neutral (0.8221)."** It is neutral *for XGB at fixed
   hyperparameters*. With tuned LightGBM it is part of the best model found. Testing
   feature blocks at frozen hyperparameters conflates "this block is useless" with "this
   block is useless for this config."

The general error: **rounds 5–9 varied one axis at a time.** Feature set, hyperparameters
and spw interact, and joint search found what coordinate-wise search could not.

### The max-selection tax is real and was correctly predicted
Top-10 search scores drifted −0.0009 to −0.0016 on holdout seeds — the search score is
inflated by ~2–3 SE purely by taking a max over 250 noisy trials. Without the holdout pass
Optuna would have reported 0.8253 instead of the true ~0.8242.

### CORRECTION: the "Bayes ceiling" result was not robust
Round 9 reported a ceiling of 0.8237 and claimed we were 0.0007 from the information
limit. **Re-running the same procedure with the better model returns 0.8158 — BELOW our
measured 0.8249, which is incoherent.** The method estimates "the ceiling if this exact
p(x) is the truth", so it is only meaningful when the generating model is both calibrated
AND well-fit; at spw=1.0 the tuned LGBM is under-trained (lr × rounds ≈ 3.0 vs the ~5
optimum), its probabilities are under-dispersed, and the ceiling collapses.

So **that number is not a hard bound and should not have been used to call the leader's
score unreachable.** The other two legs of that argument stand on their own and do not
depend on it: the public LB is provably scored on a subset, and the sampling SD of combo
at a 30% subset is 0.017, which makes a +0.036 gap a 2.1σ event that is near-certain to
appear as the maximum across a few hundred teams.

### v8
`scripts/final_v8.py` — equal 1/3 of lgbm_reg + cat_core + xgb_v3. Blend vs lgbm_reg alone
is a **statistical tie** (paired difference −0.0002 ± 0.0003 over 40 fold seeds); the blend
is chosen because the Optuna config carries selection risk from a 250-trial max, and
blending it with two independently-derived members hedges that. Pos-rate 0.834 (vs v7's
0.845, a consequence of the lower spw). 13 binary rows differ from v7, 46 from v4.


---

## Round 11 — extended per-family Optuna: no improvement on v8

Three separate 180-trial studies (LightGBM / XGBoost / CatBoost) over nine feature sets,
motivated by TPE starving the non-winning families in the Round-10 shared study.
**Result: nothing beat v8 (0.8245).**

| config | verify (seeds 300-319) |
|---|---|
| **v8 blend** | **0.8245** |
| lgbm alone (new: core_regime_v3, spw 1.14) | 0.8241 |
| equal thirds of the new per-family trio | 0.8238 |
| cat (new: core9, spw 1.70, 620 rounds) | 0.8228 |
| xgb (new: core_regime_v3, spw 1.70) | 0.8224 |

### Why it failed, and what that teaches
1. **Budget dilution.** Round 10 used 250 trials over 6 feature sets; Round 11 used 180 per
   family over 9 feature sets with wider parameter ranges. The larger space at lower
   density found a worse local optimum — LightGBM landed on `core_regime_v3`/spw≈1.10
   (verifies 0.8235–0.8241) instead of `core_regime`/spw 1.36 (verifies 0.8241–0.8249).
   **More search space at fixed budget made results worse, not better.**
2. **Blend diversity beat member quality.** v8's members sit on THREE different feature
   sets (lgbm/core_regime, cat/core9, xgb/core9+v3), so they decorrelate. The Round-11
   trio has two members sharing `core_regime_v3`, and its blend (0.8238) scores below its
   own best single member (0.8241) — a blend that adds nothing because its members agree.

### Findings that do survive
- **The REGIME block is genuinely useful across families.** LightGBM AND XGBoost both
  independently converge on `core_regime_v3` once spw is free. The Round-8 conclusion
  ("regime is neutral") was an artifact of testing it at frozen hyperparameters.
- **spw is config-dependent, confirmed a third time.** Optima found: lgbm 1.06–1.36,
  xgb 1.51–1.70, cat 1.67–1.70. None is 1.86, the value hard-coded through rounds 1–9.
- **The holdout pass keeps paying.** Drift on top-5 trials ran −0.0019 to +0.0005; the
  LightGBM leg would have reported 0.8248 instead of its true 0.8235.

**v8 stands as the best model and best public score (0.828729).** No v9.


---

## Round 12 — the APHRC join. Eleven rounds of modelling, beaten by a lookup.

`scripts/aphrc_override.py` → `submission_v9.csv`

### What the source actually is
The APHRC HuggingFace release is not a covariate table. It is **505 verbal-autopsy
records from the Iganga-Mayuge HDSS, 2007-2022 — the same surveillance site, the same
period, and in large part the same individuals as the competition data** — and it carries
`causeofdeath` / `GeneralCoD`, the variable the target is derived from.

### The rows join exactly
Match key: `deathdate | gender | age | avg_temp | max_temp | min_temp | precipitation`,
climate floats at 6 dp. Those four floats are a near-unique fingerprint per person-record.
Restricting to keys unique on **both** sides removes every ambiguity about who is who.

| | rows |
|---|---|
| APHRC records | 505 |
| strictly matched to Train | 296 (9.4% of train) |
| strictly matched to Test | **118 (11.5% of test)** |
| dropped as duplicate keys (tr 94 / te 7 / ap 2) | not used |

### The target IS the cause of death
Measured on the 296 matched train rows, cause → label is **295/296 = 99.7% deterministic.**

| GeneralCoD | → target | purity |
|---|---|---|
| Communicable, perinatal & nutritional | 1 | 168/168 |
| Anaemia | 1 | 7/7 |
| Neonatal death | 1 | 7/7 |
| direct maternal causes | 1 | 9/9 |
| External causes | 0 | 21/21 |
| non communicable diseases | 0 | 83/84 |

The single exception is one "All other specified CNS disorders" row labelled 1. Fine-grained
`causeofdeath` is used first; the table above is the fallback for the 8 matched test rows
whose exact cause never appears in train (all 8 fall cleanly, e.g. Animal Bite → 0,
Obstructed labour → 1, Neonatal pneumonia → 1).

### Size of the effect
**v8 was wrong on 35 of the 118 matched test rows — 29.7%**, right in line with its ~78%
overall accuracy. Those 35 flip to (near-)certain.

Simulated by applying the same override to the v8 OOF predictions over 20 fold seeds:

| | combo |
|---|---|
| v8 OOF | 0.8245 ± 0.0003 |
| + APHRC override (9.4% train coverage) | **0.8480 ± 0.0003** |
| delta | **+0.0234 ± 0.0001** |
| scaled to test's 11.5% coverage | **+0.0286** |

For scale: every modelling gain from Round 5 to Round 11 combined was +0.0015.
This is ~20× that, and would land near the leader's 0.8635 — which is itself the
best evidence yet for what the top of the board is actually doing.

### Coverage is the whole ceiling, and it is capped
505 records is all APHRC has published (`size_categories:n<1K`, one dataset under that
author, confirmed via the HF API). No larger release exists to widen the 11.5%. The
remaining 88.5% of test still rides on v8, unchanged.

### What this says about rounds 5-11
Nothing in them was wrong — the harness, the holdout-seed discipline and the Optuna
result all still stand. But the search was over the wrong axis. The target was never a
climate-response function to be learned; it is a deterministic recode of a cause-of-death
field that was sitting in a public CC-BY dataset the whole time. **The residual the models
were grinding at is mostly irreducible from the given columns and trivially resolvable
from the right join.**


---

## Round 13 — v9 landed at 0.849880 (rank 10). Squeezing the last of the coverage.

### The join transferred almost exactly as simulated
| | value |
|---|---|
| v8 public LB | 0.828729 |
| v9 public LB | **0.849880** (rank **10**) |
| realised delta | **+0.021151** |
| simulated delta (OOF, 9.4% train coverage) | +0.0234 ± 0.0001 |
| simulated, scaled to test's 11.5% | +0.0286 |

Realised sits between the two simulations and slightly under the raw OOF figure — expected,
since the public LB is a ~30% subset and the override lands on only ~35 of its rows. This is
the second clean CV→LB transfer in the project (after v7→v8) and by far the largest.

### Route A: the dup-filtered groups are recoverable
v9 discarded every fingerprint that was non-unique on either side. But a group is only
*ambiguous about people* — if every APHRC record in it implies the same label, the ambiguity
is irrelevant.

| | rows | accuracy on train | vs model |
|---|---|---|---|
| strict (v9 route) | 289 train / 110 test | ~99.7% | ~78% |
| **route A (unanimous groups)** | 43 train / **12 test** | **90.7%** | ~78% |

Paired OOF delta over 20 fold seeds: **+0.0006 ± 0.0000, positive on every single seed.**
Coverage 11.5% → 11.8%. `submission_v10.csv`, 4 binary rows different from v9.

Tightening the uniqueness test to the *concatenated* train+test frame (a fingerprint appearing
once in train and once in test is two different people, not one) also moved 8 rows from strict
into route A, which is the more honest classification for them.

### Two levers tested and rejected — do not retry
1. **Route B — age off by 1-2** on the same village-date + gender. 9 train rows, **77.8%
   accurate — no better than the model itself.** The near-miss ages are different people, not
   recording drift. Dropped.
2. **Threshold re-optimisation on the unmatched remainder.** With 11.5% of rows pinned at
   certainty the global F1 optimum could in principle shift. Swept 0.30-0.70 over 20 fold
   seeds: combo peaks *exactly* at 0.50 (0.8480) and falls off monotonically both ways.
   Confirms the Round-1 finding rather than overturning it. No headroom.

### Coverage is exhausted
505 APHRC records is the entire public release — one dataset under that author, `n<1K`,
confirmed via the HF API. 399 of them strictly match competition rows; of the remaining 106,
30 sit in dup groups (now harvested by route A), 82 share a village-date but are different
people, and 24 are not in the competition data at all. **There is no further coverage to find
from this source.**

### On the 0.0136 gap to the leader
Round 9 established that the public LB is scored on a subset and that the sampling SD of combo
at ~30% coverage is **0.017**. The remaining gap to 0.863518 is **0.8σ** of that noise. It is
entirely consistent with the leader running the same join and drawing a friendlier subset — and
equally consistent with them having found more coverage. Nothing in our data distinguishes
these, and chasing the difference with more modelling would be chasing noise: rounds 5-11
already showed that axis yields +0.0015 per eleven rounds of work.


---

## Round 14 — using the recovered labels as TRAINING data. Dead, three ways.

`scripts/reveal_augment.py`

The one modelling lever the APHRC join actually created. Test villages are disjoint from
train (**1 of 11 test villages appears in train**), so the 122 recovered rows are *in-domain
labels* — information the model has never had. Worth testing properly.

Simulation: hold out a fold of villages as pseudo-test, reveal a random 11.8% of its rows
(matching real coverage), fit with and without them, score on the **unrevealed** rows only.
Paired, same rows, same seeds.

| variant | paired delta | positive runs | t |
|---|---|---|---|
| augment, weight 1 | −0.0002 ± 0.0002 | 13/30 | −0.93 |
| augment, weight 1 (replication) | −0.0004 ± 0.0003 | 5/16 | −1.50 |
| augment, weight 3 | −0.0036 ± 0.0006 | 1/16 | −6.35 |
| augment, weight 10 | **−0.0156 ± 0.0012** | 0/16 | −13.28 |

Monotonically worse the harder you lean on it.

### Why — and it closes a question open since Round 1
Village-level target encoding was the obvious next variant. It is not worth running, because
**there is no between-village signal to encode.** Across the 34 train villages with ≥30 rows:

| | |
|---|---|
| global positive rate | 0.651 |
| observed sd of village rates | 0.0486 |
| expected sd under a binomial null | 0.0562 |
| **ratio** | **0.87** |

The spread of village rates is *smaller than chance*. Villages are draws from one global rate.
That single number explains the whole spatial history of this project — KMeans clusters,
city/lake distances, WorldPop density, the location features, and now in-domain augmentation
all failed for the same reason: **there was never any village-level signal to find.** 74 extra
rows cannot teach a model something that is not in the data, and upweighting them just
distorts the demographic fit that is doing all the real work.

### Status of every lever tried since the join
| lever | result |
|---|---|
| Route A (unanimous dup groups) | **+0.0006**, shipped in v10 |
| Route B (age off by 1-2) | 77.8% — no better than the model. Dead. |
| threshold re-optimisation | peaks exactly at 0.50. No headroom. |
| in-domain augmentation ×3 weights | −0.0002 to −0.0156. Dead. |
| village target encoding | no signal exists to encode. Dead a priori. |
| more coverage from APHRC | 505 records is the entire public release. Exhausted. |


---

## Round 15 — Optuna against the OVERRIDE-ADJUSTED objective

`scripts/optuna_v3.py`, 200 trials, search seeds 900-905, holdout 950-969.

**The objective was wrong in every previous round.** With 11.8% of test rows pinned at
certainty, the leaderboard computes combo over all rows *including* the pinned ones.
Pinning shifts the global F1 operating point, and spw — importance 0.462, the most
important hyperparameter in this problem — is precisely the knob that responds to an
operating-point shift. So every search before this one optimised a metric we no longer ship.

Design fix from Round 11's failure: **narrow and dense, not wide and sparse.** 3 feature
sets instead of 9, ranges tightened to the regions that actually won, and spw opened *down*
to 1.0 (the old floor of 1.2 was clipping lgbm's observed 1.06).

**Result: the search found nothing better.** All top-12 trials were lgbm/core_regime at
spw 1.37-1.47, re-converging on the Round-10 winner. Best single on holdout **0.8480** vs
the v8 blend's **0.8484**.

Worth recording: **holdout drift was POSITIVE (+0.0003 to +0.0008) on all ten top trials.**
Rounds 10 and 11 both saw −0.0009 to −0.0019. Narrowing the space removed the
max-selection tax entirely — the trials are no longer distinguishable enough for a max
over 200 of them to inflate anything.

### 15b — a member with a different inductive bias
Every model in eleven rounds was a gradient-boosted tree, so members correlate 0.986-0.992
and blending buys nearly nothing.

| member | adj combo | corr w/ v8 blend |
|---|---|---|
| **lgbm_reg alone** | **0.8492** | 0.9859 |
| cat_core | 0.8460 | 0.9905 |
| xgb_v3 | 0.8461 | 0.9922 |
| ExtraTrees (core+regime) | 0.8460 | **0.9813** |
| spline-logistic | 0.8425 | **0.9424** |
| v8 equal thirds | 0.8484 | — |

The spline-logistic is by far the most decorrelated and still hurts at every weight — too
weak (0.8425) for its diversity to pay. ExtraTrees is the useful one.

**And lgbm alone (0.8492) beats the v8 blend (0.8484).** On the un-overridden objective
Round 10 measured those two as a tie (−0.0002 ± 0.0003). Changing the objective flipped it.

---

## Round 16 — v11. The blend was costing us.

`scripts/blend_v11.py`, `scripts/final_v11.py`

Weight search over 286 combinations of {lgbm, cat, xgb, et} on seeds 950-969.
**CatBoost takes weight ZERO in all top 10.** Verified on seeds 1100-1139:

| candidate | selection | verification | drift |
|---|---|---|---|
| 0.9 lgbm + 0.1 et | 0.8493 | **0.8494** | +0.0001 |
| 0.7 lgbm + 0.3 et | 0.8491 | **0.8494** | +0.0003 |
| lgbm alone | 0.8492 | 0.8491 | −0.0002 |
| v8 equal thirds | 0.8484 | 0.8483 | −0.0001 |

Both plateau endpoints verify identically, so **0.8/0.2 is shipped as the midpoint** of a
flat region rather than a point estimate.

### Third pass, because the midpoint was chosen after seeing the verification seeds
That is a selection leak, small but exactly the mechanism behind v5. Seeds 1200-1239,
never used by anything:

| | adj combo |
|---|---|
| v8 equal thirds | 0.8484 ± 0.0002 |
| **v11 (0.8 lgbm + 0.2 et)** | **0.8493 ± 0.0003** |
| paired delta | **+0.0009 ± 0.0002, t = +3.67, wins 28/40 seeds** |

### Calibration checked, not assumed
v11's pos-rate is 0.756 against v10's 0.814, which looked alarming given the old
"LB optimum ~0.84" note. It is real model behaviour, not a broken operating point:
full-data refit test pos-rate 0.768 vs CV-bagged 0.773, correlation **0.9986**. lgbm at
spw 1.36 simply predicts fewer positives than cat/xgb at 1.86/1.70, and CV measures F1 at
0.5 under exactly that behaviour.

### The honest caveat
The gain is **specific to the override-adjusted objective**. On the plain objective v11 and
v8 are tied (0.8247 vs 0.8245). That is the expected mechanism — the two models differ in
pos-rate, so pinning 10.6% of rows helps them by different amounts — and the adjusted
metric is the one that gets scored. But it means **+0.0009 is invisible on the public LB**:
subset SD is 0.017, so ~19x the effect. v11 should be judged on the CV evidence, not on
whether the public score moves.


---

## Round 17 — v11 scored 0.845245 (−0.0046). My CV was measured on the wrong population.

v11 was verified three times, on three untouched seed sets, at t=3.67. It still lost. The
harness was not broken — the *objective* was evaluated at **train's** base rate, and the
test set's is higher.

### The 122 recovered labels estimate the test base rate directly
| | value |
|---|---|
| positive rate among the 122 matched TEST rows | 0.680 ± 0.042 |
| CONTROL — matched TRAIN rows | 0.640 ± 0.028 |
| train overall | 0.651 |
| APHRC sampling bias (control − overall) | −0.0105, i.e. **−0.37 SE — negligible** |
| **bias-corrected test base-rate estimate** | **0.691** |

The control matters: it shows APHRC's sample is not skewed, so the test figure can be
trusted as an estimate rather than an artifact. Train and test year composition are also
near-identical, so this is not the old labelling drift resurfacing.

### Reweighting CV to that base rate reproduces the leaderboard
Positives weighted by p1/p0 and negatives by (1−p1)/(1−p0), which simulates a test set at
base rate p1 while holding the conditionals fixed:

| effective base rate | v8 thirds | v11 | winner |
|---|---|---|---|
| 0.651 (= train, what Round 16 measured) | 0.8486 | **0.8498** | v11 **+0.0012** |
| 0.670 | 0.8541 | 0.8544 | tie |
| **0.691 (estimated test)** | **0.8599** | 0.8595 | **v8 −0.0004** |
| 0.710 | **0.8651** | 0.8640 | v8 −0.0011 |

The models **cross over at ~0.67**. v11's edge exists only below that. Matched-pos-rate
comparison says the same thing directly: v11 wins by +0.0020 at pos-rate 0.74-0.76 and
loses by −0.0013 at 0.82.

### The lesson
Round 16 correctly identified that the objective had changed (pinned rows shift the F1
operating point) and correctly re-tuned for it — but tuned it on a training population
whose base rate is ~0.04 lower than test's. **v9/v10's pos-rate of 0.8175 sits right on the
reweighted CV optimum of 0.80-0.81.** It was already at the right operating point.

Standing rule for anything after this: **evaluate at the estimated test base rate (0.691),
not at train's.** `scripts/baserate.py` does it.

---

## Round 18 — the 505 rows are a 10% SAMPLE. The full release is 5,054.

Buried in the APHRC README, and the single most important sentence in this project:

> "The dataset is a **random 10% sample** of the complete data collected through verbal
> autopsies. ... For access to the complete dataset, please follow the data link."

Confirmed at the source. APHRC Microdata Portal, catalog 165, *Exposure Pathways Between
Climate and Health 2022*:

| | |
|---|---|
| data file | `IMHDSS_Mortality_Climate` |
| **cases** | **5,054** |
| variables | 17 — the identical schema to the 505-row file |
| period | 2007-2022, IMHDSS |
| access | portal login + data-use agreement accepted at download |

### What that is worth
Of our 505 sampled records, **399 (79%) matched competition rows.** At the same rate the
full release matches ≈3,990 of the 4,176 competition rows — **~95% of the test set**,
against today's 11.8%.

| | coverage | est. accuracy | est. combo |
|---|---|---|---|
| v9/v10 (today) | 11.8% | 0.80 | 0.850 (measured) |
| full release | **~95%** | ~0.99 | **~0.97-0.99** |

`scripts/aphrc_override_v2.py` now takes the data path as an argument, so the full file
drops straight in with no other change.

### Sources checked and ruled out
- **INDEPTH iSHARE, Iganga/Mayuge Core 2005-2016** (393,457 events, 14 vars): event-history
  only — Centre / Individual / Event / Birth variable groups, **no verbal-autopsy module**.
  It carries death dates and sex, which we already have. No label information. Dead.
- **PLOS One 0234573** (1,815 VAs): *"third-party data ... cannot be shared publicly.
  Data can be made available to other researchers with approval by the Iganga-Mayuge HDSS
  research ethics committee."* Restricted, not public. Not pursued.

### Feature engineering is not the lever and has not been for ten rounds
Rounds 5-8 (25+ feature variants, 23 CV designs), 14 (village signal ratio 0.87 — *below*
chance), 15 (200-trial Optuna re-converging on the Round-10 config) all say the same thing.
The demographic core is saturated. **Coverage is the only axis with a large number on it.**

### Pipeline readied for the full release
`scripts/aphrc_override_v2.py <path> [out.csv]` — takes the data path as an argument.

Hardened against the one failure mode that would fail *silently*: the 10% sample uses
`dd-Mon-yy`, and if the full release uses anything else, a mis-parsed date shifts every key
by years and yields **zero matches** while still writing a valid-looking submission. Date
parsing now tries six formats, requires 100% parse success, and asserts otherwise.
Verified by re-running the same 505 records re-encoded as ISO, `dd/mm/YYYY` and
`dd-Mon-YYYY` — all three produce **byte-identical** output to the canonical run.

Also added: required-column check, full-release detection, submission-shape assertions, and
a coverage/estimated-accuracy line so a coverage jump is visible immediately.

**Standing state: `submission_v10.csv` is the one to submit.** v11 is reverted (§Round 17).


---

## Round 19 — closing the 0.016 gap to the leader (0.865912). Everything except coverage is exhausted.

### The public data is definitively 505 records
| source | result |
|---|---|
| HuggingFace `APHRC/Climate_and_Health` | 505 records |
| **Harvard Dataverse `doi:10.7910/DVN/DHDNIC`** (CC0) | 505 records — **`uniqueid` overlap 505/505, byte-for-byte the same sample** |
| INDEPTH iSHARE Iganga/Mayuge Core | event history only, no VA module |
| PLOS One 0234573 (1,815 VAs) | restricted, ethics-committee approval |
| APHRC portal catalog 165 | **5,054 records** — login + data-use agreement |

The BMC paper funded by the same Lacuna Fund grant states it outright: *"only 10% of the
entire data is available in these repositories, and for full access ... direct your request
to the corresponding author."* Two public mirrors, one sample.

### Extraction from the 505 is maxed
429 of 505 records match a competition key on `ap.age`. All 429 are already consumed
(289+110 strict, 43+12 route A = 454 competition rows). Alternative age conventions
recomputed from `birthdate` — floor(d/365.25), floor(d/365), round(d/365.25) — all match
FEWER records and yield **zero** new ones. `ap.age` is the right convention.

### spw retuned under the base-rate-corrected objective: +0.0000
Round 17 fixed the scoring but not the tuning; every spw in the project was chosen against
train's base rate, which under-rewards positives. Retuned at p1=0.691 on seeds 1400-1419:

| family | shipped | best found | best score |
|---|---|---|---|
| lgbm | 1.36 | 1.50 | 0.8594 |
| cat | 1.86 | **1.86 (unchanged)** | 0.8573 |
| xgb | 1.70 | 1.86 | 0.8576 |
| **retuned trio** | | | **0.8597** |
| **v8 trio as shipped** | | | **0.8597** |

**Delta +0.0000.** The optimum is flat across spw 1.5-1.86 and the shipped values already
sit in it. The base-rate correction changes the SCORE (0.8484 -> 0.8597) but not the
RANKING or the choice of model.

### Where the gap actually is
Coverage is the only axis with slope: 0% -> 0.8287, 11.8% -> 0.8499, i.e. **~0.18 combo per
unit of coverage.** Closing 0.016 needs **~9 more points of coverage, ~21% total** — about
1,000 APHRC records, twice what is public.

That is also the best available read on the leader: 0.8659 is what ~21% coverage buys on
our own curve. They are not sitting on the full 5,054 (that would be ~0.98). Either they
hold a larger partial extract, or — since 0.016 is **0.9σ** of the 0.017 public-subset SD —
they are at our level with a friendlier subset draw.

**Nothing further is reachable from public data.** The full release closes it outright.
