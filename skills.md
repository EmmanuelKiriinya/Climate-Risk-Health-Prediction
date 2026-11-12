# skills.md — Climate Risk & Health Prediction Challenge (Zindi)

Playbook for an AI agent to reproduce / improve the current best solution.
Project root: `/Users/luxdevhq/Desktop/Zindi/climate-risk-health-prediction`

---

## 1. Task & scoring (read first — this shapes every decision)

Binary classification: predict whether a death record in Uganda is **climate-sensitive**
(`is_climate_sensitive` ∈ {0,1}).

Leaderboard score = **0.6 × F1 + 0.4 × ROC-AUC** ("combo").

Hard rules:
- Binary prediction MUST come from the default threshold: `TargetF1 = (TargetRAUC >= 0.5)`.
  Custom thresholds are forbidden.
- You may still influence where 0.5 falls *through training* (e.g. `scale_pos_weight`,
  class weights). That is legitimate model design, not a threshold hack.

Submission format (exact columns, exact order):

```
ID,TargetF1,TargetRAUC
ID_E760D84B,1,0.5328145296161371
```

Validation asserts before submitting:
```python
list(sub.columns) == ["ID", "TargetF1", "TargetRAUC"]
sub.ID.equals(test.ID)                      # same order, no missing IDs
sub.TargetF1.isin([0, 1]).all()
(((sub.TargetRAUC >= .5).astype(int)) == sub.TargetF1).all()   # binary == prob>=0.5
no NaNs, probs within [0, 1]
```

## 2. Data map (`Data/`)

| File | Rows | Notes |
|---|---|---|
| `Train.csv` | 3,146 | incl. target `is_climate_sensitive`; 65.1% positive |
| `Test.csv` | 1,030 | same features minus target |
| `climate_features.csv` | 4,176 | pre-engineered ERA5/CHIRPS/MODIS/SRTM features |
| `data_dictionary.csv`, `downloaded_climate_features_data_dictionary.csv` | — | column docs |

- Join climate file on `ID` (100% coverage of train AND test). Drop its duplicate
  `deathdate` column before merging.
- No missing values anywhere. `hot_days_30d` is constant (all zeros) — dead feature.
- Both splits span 2007–2022 with proportional year counts → random-ish temporal split.
- Columns: demographics (`age`, `gender`, `zone`), date, observed weather
  (avg/max/min temp, precip), coords (`latitude`,`longitude`), free-text `location`
  ("village, district, Uganda").

## 3. Critical findings (verified empirically)

1. **Age is the dominant signal.** Univariate AUC 0.733. Climate-sensitive deaths skew
   very young: median age 1 (mean 14) vs mean 43 for negatives.
   Target rate by band: ≤1 yr 82%, 1–5 yrs **91%**, 5–18 56%, 18–40 54%, 40–60 47%, 60+ 30%.
2. **Test villages are disjoint from train.** Only 1 of 38 train villages appears in test;
   zero rounded-coordinate overlap. → Never one-hot / target-encode village or district;
   they will not generalize. Test-only districts: Kamuli, Kalangala, Sembabule.
3. **Climate features are individually weak (AUC 0.50–0.54) but additive** in GBMs
   (top gains after age/year: tavg_obs_minus_rean, rain_sum_90d, max_daily_rain_30d,
   ndvi_90d, tx_tavg_gap…).
4. **`year` is the #2 feature by gain** (~half of age's importance) — labeling drift over
   time is real; keep it. The drift is specifically in INFANTS: `age==0` target rate is
   0.966 up to 2017 then 0.563 from 2018 (sharp break, stable across the 10 prior years),
   while `age 1-4` holds ~0.92 throughout. Slice on `age==0 / 1-4 / 5+` — NEVER on
   `is_under5`, which merges a −0.40 effect with a −0.05 one and cancels the signal.
5. Mild seasonality (Jan/Nov ~69% vs Mar/Sep/Oct ~60%). Encode months cyclically.
6. Gender: Female 67% vs Male 63% positive. Zone (Rural vs Peri_urban): negligible.
7. GroupCV ≈ StratifiedCV scores → little spatial leakage inflation, but still prefer
   group-aware CV for honesty (test IS village-disjoint).

## 4. Cross-validation protocol (non-negotiable)

Use `scripts/harness.py`. Two rules, both learned the hard way (see results.md Round 5):

1. **Never early-stop on the validation fold.** `.fit(..., eval_set=[val], early_stopping_rounds=N)`
   picks the round count on the rows being scored. It inflates OOF by a different amount
   per config, so it corrupts the RANKING. Treat `n_estimators` as a hyperparameter and
   fit with a fixed count. (`lr × rounds ≈ 5` is the optimum for depth-3 XGB here.)
2. **Repeat over ≥10 fold seeds and report a standard error.** Per-fold-seed combo std is
   ~0.0013. A single-seed run cannot resolve the ~0.001 deltas this problem produces.
   Confirm finalists on a held-out set of fold seeds (e.g. 100–119) never used for
   selection — this catches noise-fitting before the leaderboard does.

```python
from harness import evaluate            # fixed rounds, repeated StratifiedGroupKFold(5)
groups = pd.factorize(location.split(",")[:2].join(","))   # village+district
mean, se, scores, oofs, test_pred = evaluate(fam, params, spw, rounds, X, y, groups,
                                             fold_seeds=range(10))
score = 0.6 * f1_score(y, p >= 0.5) + 0.4 * roc_auc_score(y, p)   # optimize THIS
```

## 5. Feature engineering (implemented in `scripts/features.py`)

Keep (43 features total):
- Raw: `age`, observed temps, precipitation, lat/lon, all 15 informative climate cols
- Categorical → int: `gender_bin` (Female=1), `zone_bin` (Rural=1)
- Date: `year`, sin/cos of month and day-of-year
- Age transforms: `age_log1p`, flags `is_infant`(≤1), `is_under5`, `is_child`(5–17),
  `is_elderly`(≥60)
- Climate deltas/ratios: `t_anom_7d_30d`, `t_anom_30d_90d`, `tx_tavg_gap`,
  `rain_share_7d_30d`, `rain_share_30d_90d`, `dryness_idx` (rain30/tavg30),
  `ndvi_delta`, plus obs-vs-reanalysis agreement (`tavg_obs_minus_rean`,
  `precip_obs_vs_rain30`)
Drop: `location`, district, village strings, `hot_days_30d`.

## 6. Winning recipe (as submitted — `scripts/predict.py`)

Weighted blend of 5 configs, each seed-bagged (seeds `[42, 7, 2026, 99, 5]`),
tree counts = median GroupCV early-stop round × 1.15:

| Family | Params | Weight |
|---|---|---|
| XGBoost | lr 0.03, depth 3, mcw 1 | 0.29 |
| XGBoost | lr 0.05, depth 3, mcw 5 | 0.29 |
| XGBoost | lr 0.05, depth 3, mcw 1 | 0.21 |
| CatBoost | lr 0.03, depth 4 | 0.14 |
| CatBoost | lr 0.05, depth 4 | 0.07 |

Common: `scale_pos_weight=1.86`, subsample 0.8, colsample 0.8, reg_lambda 1.0.

Results ladder (GroupCV combo):
- LogisticRegression (balanced): 0.779
- LightGBM best (leaves 15, lr 0.03, spw 1.86, mcs 40): 0.8155
- Single XGB depth-3 spw 1.86: **0.8182**
- 5-config OOF-weighted blend: **0.8199**

Depth-3 XGB beats deeper trees and LGBM here — dataset is tiny (3k rows) and the
signal is mostly monotone-in-age + additive climate terms. Shallow trees generalize best.

## 7. Pitfalls checklist

- ❌ Custom probability thresholds → instant rule violation.
- ❌ Village/district one-hots or target encoding → disjoint across splits.
- ❌ Plain KFold without groups → optimistic, but here it also happens to mislead less;
  still use GroupCV for tuning decisions.
- ❌ Post-hoc probability transforms to shift the 0.5 cut (isotonic toward F1-optimal
  cutoff etc.) → considered threshold circumvention. Tune `scale_pos_weight` instead.
- ❌ Trusting `hot_days_30d` / constant cols.
- ✅ Seed bagging + small blend ≈ +0.002 combo, essentially free.
- ✅ Always assert submission integrity (section 1).

## 8. Reproduce

```bash
cd "/Users/luxdevhq/Desktop/Zindi/climate-risk-health-prediction"
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python pandas numpy scikit-learn lightgbm xgboost catboost matplotlib

.venv/bin/python scripts/eda.py       # exploratory stats
.venv/bin/python scripts/train.py     # baseline comparison, GroupCV vs StratCV
.venv/bin/python scripts/tune.py      # hyperparam sweep, blend-weight search, importances
.venv/bin/python scripts/predict.py   # writes final submission.csv
```

Artifacts: `artifacts/tune_results.csv`, `artifacts/oofs.npz`, `artifacts/best_blend.json`.

## 9. Round-2 experiments (post first submission: LB 0.8251, OOF blend 0.8199)

Public LB 0.8251 vs OOF 0.8199 → CV slightly *under*-estimates; no overfitting.

### Non-stationarity (the defining property of this dataset)
- Target rate falls from ~78% (2008–10) to ~50% (2018–21). The composition shift is in
  INFANTS: `age<1` share goes ~25% (2008–17) → 64/73/81% (2019/21/22) while the 1–4 band
  collapses ~30% → 3%. (An earlier note here said "under-5 share jumps ~5% → 40%" — that
  was wrong; under-5 share is 41–56% throughout.) `year` is therefore the #2 feature.
  The old "**TimeSplit CV collapses** (AUC 0.45)" claim was a BUG, not a finding: in
  `experiment.py`, `oof_for` fills only validation indices, but TimeSplit is a single
  split, so the ≤2015 half of `oof` stays 0.0 while F1/AUC are scored on the full `y`.
- Test spans the same years proportionally → random/grouped CV is the right protocol;
  never tune on time splits here.
- Era-routed models (separate ≤2017 / ≥2018 models) scored WORSE (0.8169 vs 0.8191).

### Feature blocks that do NOT help (all tested, GroupVillage combo)
- Spatial KMeans(10) clusters + distances + coord polys: 0.8168–0.8185
- Monthly climate anomalies (per cluster×month normals, train+test inputs only): 0.8164
- Season dummies (Uganda long/short rains): +0.001 at best → keep in final
- Age quantile bins, u5×year interaction: hurt (0.8173 / 0.8165)
- Dropping zone/gender: hurts (keep them)
Internal features are saturated at ~0.82 OOF; the LB gap to top teams (~0.863)
almost certainly requires external public data (see §10).

### CV scheme comparison (XGB d3, base+season features)
| Scheme | Combo |
|---|---|
| StratifiedKFold(5) | 0.8194 |
| StratifiedGroupKFold by village | 0.8185–0.8191 |
| StratifiedGroupKFold by spatial cluster | 0.8153 |
| TimeSplit | 0.4022 (do not use for tuning) |

### Zoo + stacking (`scripts/zoo.py`, artifacts/zoo.npz)
Seed-bagged (3 seeds) GroupVillage OOFs: lgbm_l15 0.8194, cat_d4 0.8188,
xgb_d3_lr05 0.8183, xgb_d3_lr03 0.8175, histgb 0.7931.
- Weight search over zoo OOFs: **combo 0.8208**
  weights ≈ {xgb_lr05 .34, xgb_lr03 .10, lgbm .23, cat .31, histgb .03}
- Rank-average blending: 0.7822 — destroys spw-shaped calibration, never use with F1@0.5
- LR meta-stack: 0.8184 — no gain over weight search
→ Final: `submission_v2.csv` from weight-blended test probs (pos-rate 0.844).

### Threshold headroom check
F1 across thresholds on OOF blend peaks exactly AT 0.5 (spw=1.86 already places the
operating point optimally). Further gains must come from ranking quality (AUC), not
probability shaping.

### Round-2b: external data + pseudo-labels (LB v2 = 0.8265, rank 91)
- **WorldPop 2020 1km population** sampled at coords (`scripts/external.py`):
  pop-only AUC 0.49; adding to features: no lift (0.8188 vs 0.8191).
- City distances (Kampala/Jinja/Iganga/Mbale/Tororo/Entebbe/Mukono), Lake Victoria
  distance, log_pop×near-Kampala: no lift.
- **Pseudo-labeling** confident test rows back into training: HURTS (0.8166).
- Refined 24-config XGB random search around winner: flat surface, best 0.8179 < zoo's 0.8183.
- Extended 7-model zoo weight search: OOF combo 0.8207 ≈ previous 0.8208 → hard plateau.
- `submission_v3.csv` differs from v2 in only 8 binary rows; expected LB delta ≈ noise.

**Conclusion:** internal + simple-join external features saturate at ~0.82 OOF / ~0.826 LB.
The remaining gap to ~0.86 likely needs richer climate time-series downloads (raw CHIRPS /
ERA5 with different variable sets, LST, humidity), or exploits structure we haven't found.

### Round-3 external APIs (LB v3 = 0.8275, rank 88)
- **NASA POWER daily API** (free, no auth): RH2M, QV2M, T2M, WS2M(+MAX),
  ALLSKY_SFC_SW_DWN, PRECTOTCORR, T2M_RANGE for all 55 sites × 2006-12..2023-01
  (`scripts/fetch_power.py` → `data_external/power_cache`, features in
  `scripts/features_power.py`). Rolling windows incl. **dry-spell length**, VPD,
  wet days, site-month anomalies.
- Strongest climate univariate signals found anywhere:
  `rh_mean_30d_anom` AUC **0.5495**, `rh_mean_90d` 0.5413, `qv_mean_30d_anom` 0.5409,
  `vpd_mean_30d` 0.5352 — humidity anomalies matter more than any provided CHIRPS/ERA5 col.
- BUT no model lift even with curated subsets (combo 0.8192 vs 0.8191 ref). Depth-3 +
  n≈3k saturates; extra weak columns dilute.
- Open-Meteo archive API also works (hourly quota! ~80s per full-range request;
  chunk per year + cache) — see `scripts/fetch_openmeteo.py`.
- Final 8-model zoo blend (old + new feature sets): OOF combo **0.8209**.
  `submission_v4.csv` differs from v3 in 8 rows → expect ±0.001 LB movement.

**Plateau is real and robust**: every family of attempts lands at 0.817–0.821 OOF /
~0.826–0.828 LB. Data sources used (for competition forum transparency):
NASA POWER API https://power.larc.nasa.gov/, WorldPop https://data.worldpop.org/,
Open-Meteo historical weather https://open-meteo.com/en/docs/historical-weather-api

### Round-4: THE core-feature discovery (LB v4 = 0.8286, rank 84)
A model using ONLY `age, age_log1p, is_infant/is_under5/is_child/is_elderly,
gender_bin, zone_bin, year` beats every 45+-feature model ever built here:

| Model | AUC | combo |
|---|---|---|
| **core-only XGB d3 spw1.5** | **0.8291** | 0.8232 |
| **core-only CatBoost d4 spw1.86** | **0.8273** | **0.8241** |
| core-only LGBM l15 spw1.5 | 0.8270 | 0.8231 |
| best full-feature blend (v4) | ~0.822 | 0.8209 |

Lessons:
1. The climate/location columns add training-fold signal but NO generalizable ranking
   power at n=3k. Simplicity wins. Always benchmark a demographics-only baseline.
2. Adding season/anomalies to core: hurts (0.8210-0.8212).
3. spw sweep on core: 1.5 > 1.86 > 2.2 > 2.6.
4. Blending the three core families (xgb .27 / lgbm .15 / cat .58) with weights chosen
   to maximize MEAN combo over 3 fold seeds: **OOF 0.8246** → `submission_v5.csv`
   (`scripts/v5_final.py`, 53 rows differ from v4).
5. Gotcha: full-data refits must use CV-median early-stop rounds × 1.15 — running max
   rounds without ES pushed ALL test probs above 0.5 (pos-rate 1.0 bug).

## 9b. Round 5 — the honest-measurement rebuild (`submission_v7.csv`)

Re-measuring everything with the harness above overturned several round-1..4 conclusions
and produced the current best-founded model.

**Recipe (`scripts/final_v7.py`) — equal weights, nothing fitted:**

| Member | Features | Model | spw | rounds |
|---|---|---|---|---|
| xgb_core | core9 | XGB depth 3, lr .05, sub .85, col .9, λ1 | 1.86 | 100 |
| cat_core | core9 | CatBoost depth 4, lr .05, l2 3.0 | 1.86 | 250 |
| xgb_v3 | core9 + v3 interactions | XGB depth 3, lr .05 | 1.70 | 100 |

Weights fixed at **1/3 each**. Test probs bagged over 20 fold seeds × 5 folds × 3 model
seeds. Honest combo **0.8230**, reproduced on three independent fold-seed sets.

**Why equal weights:** Dirichlet-fitted weights scored 0.8228 on the seeds they were fit
on but 0.8227 on held-out seeds; equal thirds scored 0.8229 on BOTH. v4 and v5 both fitted
weights on leaked OOFs — that is the mechanism behind v5's non-transferring +0.0037.

**Honest ranking of feature sets** (all comparable, ±0.0003):
`core9` / `min5` **0.8223** > `core+topclimate` 0.8197 > `core+cyclic` 0.8189 >
**`full 43` 0.8175**. Everything beyond the demographic core hurts.

**spw is already optimal.** Sweeping 1.0→3.0: flat plateau 1.6–2.0, peak 1.86. AUC is
invariant to spw (0.8277 ± 0.0001) exactly as theory predicts, so spw only ever buys F1.
The empirical F1-optimal cut on calibrated probs is t*=0.385 → implied spw 1.60. No headroom.

**Leakage audit: all negative.** Zero train/test coordinate overlap (43 vs 12 sites), no
duplicate records, no ID or row-order signal. Do not spend more time here.

**The LB noise floor — read this before chasing any delta.** On 1,030 test rows, flipping
ONE prediction moves combo by ~0.00078. The entire v1→v5 LB progression (0.8251→0.8286)
is ~6 rows. v4 vs v5 (0.000608) is **0.8 rows**. Do not treat sub-0.003 LB movements as
signal; select on the harness.

**The irreducible core:** `age==0 & year>=2018` (n=492, 16% of train) sits at 56% positive
with best in-cell univariate AUC 0.586 (`year` itself) and every climate variable at 0.53.
That cell is the floor under AUC ≈ 0.83.

## 9c. STOP CONDITION (read before starting another round)

External data **is permitted** (host-confirmed) — but it cannot help:
- Every join key an external dataset could use is exhausted. Permutation residual tests
  find no signal beyond chance at location, time, location×time (Round 7b) or demographic
  keys incl. age×sex×year (Round 9). That covers GBD, MAP, DHS, WorldPop — everything.
- The **Bayes ceiling is combo 0.8237**; v7 scores **0.8230**. We are 0.0007 from the
  theoretical maximum for these features (`scripts/bayes_ceiling.py`).
- The public leaderboard is provably scored on a SUBSET of the 1,030 test rows, so its
  top score (0.864258) sits +0.041 ABOVE that ceiling and is a noise artifact.

Tally of what has been ruled out with error bars: 76 feature variants (incl. 20 PCA, v3
interactions, v4 density/regime/continuous-time, out-of-fold target encoding, monotone
constraints), 23 CV designs, 5 model families, spw 1.0–3.0, blend weightings, and four
external APIs. Everything lands at 0.817–0.823.

**SUPERSEDED by Round 10.** A 250-trial Optuna TPE search found +0.0015 over v7, verified
on four independent fold-seed sets. The plateau was an artifact of coordinate-wise search:
rounds 5–9 varied feature set, hyperparameters and spw ONE AT A TIME, and those axes
interact. Specifically, `spw` is the most important hyperparameter (importance 0.462) and
its optimum is config-dependent — the earlier "spw=1.86 is optimal" finding came from a
single-config sweep and cost 0.0025.

Lessons that survive: use the honest harness, always re-score a search winner on held-out
fold seeds (the 250-trial max was inflated by ~0.0011), and joint-search rather than
one-axis-at-a-time. The residual-signal tests (no join key retains structure) still stand,
so EXTERNAL data remains unpromising — but internal joint search was not exhausted.

## 10. Ideas not yet tried (ordered by expected value)

1. External public data per challenge rules (link sources on forum): ERA5 land-surface
   temperature, malaria-incidence rasters, population density, flood extent — join via
   lat/lon or district name. This is the most plausible path to close the ~0.04 gap.
2. District-level socio-demographic covariates (Uganda census/DHS): poverty rate,
   health-facility density matched by district string.
3. Stacking: LR/logistic meta-model on OOF probs of 5+ diverse base models
   (add HistGB, sklearn ExtraTrees) instead of fixed weights.
4. More seeds (10–20) + per-config weight re-search on stacked OOF matrix.
5. Monotone-constrained GBMs on age within era subsets.

