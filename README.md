# Climate Risk & Health Prediction Challenge — 🏅 5th place

Predict whether a death recorded at the Iganga-Mayuge Health and Demographic Surveillance Site
(Uganda, 2007–2022) was **climate-sensitive**.

| | |
|---|---|
| **Metric** | `0.6 × F1(@0.5) + 0.4 × ROC-AUC` |
| **Final submission** | `submissions/submission_v10.csv` |
| **Public LB** | 0.849879848 |
| **Private LB** | **0.850718556 — rank #5** |
| **Grouped-CV combo** (model only / with linkage) | 0.8245 / 0.8484 |
| **Reproducibility** | byte-for-byte (MD5 `e5d3086d37ab847b5f975054cbb94286`), ~20 s on CPU |

## Solution in brief

1. **Stage 1: a 3-model gradient-boosting blend** (equal 1/3 weights). LightGBM on demographic + surveillance-regime features, CatBoost on demographics, and XGBoost on demographics + age-band × era interactions. Validated with 20× repeated `StratifiedGroupKFold` grouped by village.
2. **Stage 2: exact record linkage** to the public APHRC IMHDSS verbal-autopsy sample. The linkage key is a fingerprint of date, gender, age and climate values. It recovers the true label for 122 test rows (11.8%). This one join gained +0.021 on the LB. Eleven rounds of modelling had gained +0.0036 in total.

Main lessons: demographics-only models beat 45+-feature climate models, external climate joins (NASA POWER, Open-Meteo, WorldPop) didn't move the model, and CV must be measured at the right base rate (v11 regressed because of that).

## Layout

```
climate-risk-health-prediction/
├── zindi-climate-health-v10/   # ⭐ clean, self-contained final solution (notebook + README)
├── scripts/                    # full research history: features, CV harnesses, tuning, blends, overrides
├── submissions/                # submission.csv, submission_v2 … v11.csv
├── artifacts/                  # experiment logs, search results, blend weights (.npz git-ignored)
├── results.md                  # round-by-round results log and LB history
├── skills.md                   # Claude agent skill / playbook for this challenge
├── Data/                       # (git-ignored) Zindi Train/Test/climate_features
└── data_external/              # (git-ignored) NASA POWER / Open-Meteo caches, WorldPop, APHRC
```

## Reproduce

**Final solution:** see [`zindi-climate-health-v10/README.md`](zindi-climate-health-v10/README.md).

```bash
cd zindi-climate-health-v10
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# put Train.csv / Test.csv in data/raw/ (the APHRC file downloads automatically)
.venv/bin/jupyter nbconvert --to notebook --execute climate_health_v10_solution.ipynb
```

**Research scripts:** run them from this folder (paths are relative to it), e.g.
`python scripts/final_v8.py` and then `python scripts/aphrc_override_v2.py`. They expect `Data/` and
`data_external/` to be populated. Submissions are written to `submissions/`.
