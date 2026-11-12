"""Model zoo: diverse configs -> GroupVillage OOF probs + seed-bagged full-fit test probs.

Saves artifacts/zoo.npz with y, oof_<key>, test_<key> for stacking analysis.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data
from features_v2 import build_v2

SEEDS = [42, 7, 2026]
SPW = 1.86


def models():
    return {
        "xgb_d3_lr05": lambda s: xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=3,
            min_child_weight=1, scale_pos_weight=SPW, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=s, eval_metric="logloss", early_stopping_rounds=100, verbosity=0),
        "xgb_d3_lr03": lambda s: xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=3,
            min_child_weight=1, scale_pos_weight=SPW, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=s, eval_metric="logloss", early_stopping_rounds=100, verbosity=0),
        "lgbm_l15": lambda s: lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=15,
            scale_pos_weight=SPW, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=s, verbose=-1),
        "cat_d4": lambda s: CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=4,
            scale_pos_weight=SPW, l2_leaf_reg=3.0, random_seed=s, verbose=False,
            allow_writing_files=False, early_stopping_rounds=100),
        "histgb": lambda s: HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=15, max_iter=800, min_samples_leaf=40,
            l2_regularization=1.0, class_weight={0: 1.0, 1: SPW}, random_state=s),
    }


def main():
    tr_raw, te_raw = load_data()
    y = tr_raw.is_climate_sensitive.reset_index(drop=True)
    Xtr_all, Xte_all, _ = build_v2(tr_raw, te_raw)
    use = list(Xtr_all.columns[: len(engineer_cols := 43)] if False else Xtr_all.columns)
    # base + season only (best blockset from cv_scheme_results)
    season_cols = [c for c in Xtr_all.columns if c.startswith("season_")]
    X, Xte = Xtr_all[list(Xtr_all.columns[:43]) + season_cols], Xte_all[list(Xte_all.columns[:43]) + season_cols]

    village = pd.factorize(tr_raw.location.str.split(",").str[:2].str.join(","))[0]
    sgk = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    splits = list(sgk.split(X, y, village))

    store = {"y": y.values}
    for key, ctor in models().items():
        oof = np.zeros(len(y)); prob_te = np.zeros(len(Xte))
        for seed in SEEDS:
            oof_s = np.zeros(len(y)); te_s = np.zeros(len(Xte))
            for tr_i, va_i in splits:
                m = ctor(seed)
                fit_kw = {}
                if key.startswith("lgbm"):
                    fit_kw = dict(eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                                  callbacks=[lgb.early_stopping(100, verbose=False)])
                elif key == "histgb":
                    pass  # no ES support; fixed iters
                else:
                    fit_kw = dict(eval_set=[(X.iloc[va_i], y.iloc[va_i])], verbose=False)
                m.fit(X.iloc[tr_i], y.iloc[tr_i], **fit_kw)
                oof_s[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
                te_s += m.predict_proba(Xte)[:, 1] / len(splits)
            oof += oof_s / len(SEEDS); prob_te += te_s / len(SEEDS)
        f1 = f1_score(y, (oof >= 0.5).astype(int)); auc = roc_auc_score(y, oof)
        print(f"{key:12s} F1={f1:.4f} AUC={auc:.4f} combo={0.6*f1+0.4*auc:.4f}")
        store[f"oof_{key}"] = oof; store[f"test_{key}"] = prob_te

    np.savez_compressed("artifacts/zoo.npz", **store)


if __name__ == "__main__":
    main()
