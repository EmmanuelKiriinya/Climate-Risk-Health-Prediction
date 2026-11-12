"""Final model: weighted blend of tuned XGB/CatBoost configs, seed-bagged, trained on full data.

Records best iterations via GroupCV, then refits on all data and writes submission.csv.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer, make_submission

SEEDS = [42, 7, 2026, 99, 5]
N_SPLITS = 5

CONFIGS = [
    # (family, params, weight)
    ("xgb", dict(learning_rate=0.03, max_depth=3, min_child_weight=1), 0.29),
    ("xgb", dict(learning_rate=0.05, max_depth=3, min_child_weight=5), 0.29),
    ("xgb", dict(learning_rate=0.05, max_depth=3, min_child_weight=1), 0.21),
    ("cat", dict(learning_rate=0.03, depth=4), 0.14),
    ("cat", dict(learning_rate=0.05, depth=4), 0.07),
]


def make_model(family, params, seed, rounds):
    if family == "xgb":
        return xgb.XGBClassifier(
            n_estimators=rounds, scale_pos_weight=1.86, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
            eval_metric="logloss", verbosity=0, **params)
    return CatBoostClassifier(
        iterations=rounds, scale_pos_weight=1.86, l2_leaf_reg=3.0,
        random_seed=seed, verbose=False, allow_writing_files=False, **params)


def main():
    tr, te = load_data()
    X, Xte = engineer(tr), engineer(te)
    y = tr.is_climate_sensitive.reset_index(drop=True)

    # pass 1: find typical early-stop round counts per config
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    splits = list(cv.split(X, y, groups))
    rounds_per_cfg = []
    for family, params, _ in CONFIGS:
        stops = []
        for tr_i, va_i in splits:
            kw = dict(eval_set=[(X.iloc[va_i], y.iloc[va_i])], verbose=False) if family == "xgb" \
                else dict(eval_set=(X.iloc[va_i], y.iloc[va_i]))
            m = xgb.XGBClassifier(n_estimators=3000, eval_metric="logloss",
                                  early_stopping_rounds=100, verbosity=0,
                                  scale_pos_weight=1.86, subsample=0.8, colsample_bytree=0.8,
                                  reg_lambda=1.0, random_state=42, **params) \
                if family == "xgb" else \
                CatBoostClassifier(iterations=3000, early_stopping_rounds=100, verbose=False,
                                   allow_writing_files=False, scale_pos_weight=1.86,
                                   l2_leaf_reg=3.0, random_seed=42, **params)
            m.fit(X.iloc[tr_i], y.iloc[tr_i], **kw)
            best = m.get_booster().best_iteration + 1 if family == "xgb" else m.get_best_iteration() + 1
            stops.append(best)
        rounds_per_cfg.append(int(np.median(stops) * 1.15))
        print(f"{family} {params}: median best round {int(np.median(stops))} -> using {rounds_per_cfg[-1]}")

    # pass 2: seed-bagged fit on full data, blend with weights
    proba = np.zeros(len(Xte))
    for (family, params, w), rounds in zip(CONFIGS, rounds_per_cfg):
        cfg_p = np.zeros(len(Xte))
        for seed in SEEDS:
            m = make_model(family, params, seed, rounds)
            m.fit(X, y)
            cfg_p += m.predict_proba(Xte)[:, 1] / len(SEEDS)
        proba += w * cfg_p
        print(f"blended {family} {params} (w={w})")

    sub = make_submission(te.ID.values, proba, path="submissions/submission.csv")
    print("\nsubmission.csv:", sub.shape, "| positives:", int(sub.TargetF1.sum()),
          f"({sub.TargetF1.mean():.3f})", "| prob range:",
          round(proba.min(), 4), "-", round(proba.max(), 4))
    print(sub.head())


if __name__ == "__main__":
    main()
