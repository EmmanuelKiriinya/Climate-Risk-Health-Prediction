"""Final ensemble: zoo on base+season+rh/qv anomalies, blended with prior zoo OOFs."""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
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
        "n_xgb_d3_lr05": lambda s: xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=3,
            min_child_weight=1, scale_pos_weight=SPW, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=s, eval_metric="logloss", early_stopping_rounds=100, verbosity=0),
        "n_xgb_d3_lr03": lambda s: xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=3,
            min_child_weight=1, scale_pos_weight=SPW, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=s, eval_metric="logloss", early_stopping_rounds=100, verbosity=0),
        "n_lgbm_l15": lambda s: lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=15,
            scale_pos_weight=SPW, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=s, verbose=-1),
        "n_cat_d4": lambda s: CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=4,
            scale_pos_weight=SPW, l2_leaf_reg=3.0, random_seed=s, verbose=False,
            allow_writing_files=False, early_stopping_rounds=100),
    }


def main():
    tr_raw, te_raw = load_data()
    y = tr_raw.is_climate_sensitive.reset_index(drop=True)
    Xall, Xte_all, _ = build_v2(tr_raw, te_raw)
    cols = list(Xall.columns[:43]) + [c for c in Xall.columns if c.startswith("season_")]
    X, Xte = Xall[cols], Xte_all[cols]
    P_tr = pd.read_csv("artifacts/power_features_train.csv")[["rh_mean_30d_anom", "qv_mean_30d_anom"]]
    P_te = pd.read_csv("artifacts/power_features_test.csv")[["rh_mean_30d_anom", "qv_mean_30d_anom"]]
    X = pd.concat([X.reset_index(drop=True), P_tr], axis=1)
    Xte = pd.concat([Xte.reset_index(drop=True), P_te], axis=1)

    village = pd.factorize(tr_raw.location.str.split(",").str[:2].str.join(","))[0]
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, village))

    store = {"y": y.values}
    z = np.load("artifacts/zoo.npz")
    for k in z.files:
        store["old_" + k] = z[k]

    for key, ctor in models().items():
        oof = np.zeros(len(y)); prob_te = np.zeros(len(Xte))
        for seed in SEEDS:
            oof_s = np.zeros(len(y)); te_s = np.zeros(len(Xte))
            for tr_i, va_i in splits:
                m = ctor(seed)
                if key.startswith("n_lgbm"):
                    m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
                elif key.startswith("n_cat"):
                    m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=(X.iloc[va_i], y.iloc[va_i]))
                else:
                    m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])], verbose=False)
                oof_s[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
                te_s += m.predict_proba(Xte)[:, 1] / len(splits)
            oof += oof_s / len(SEEDS); prob_te += te_s / len(SEEDS)
        f1 = f1_score(y, (oof >= .5).astype(int)); auc = roc_auc_score(y, oof)
        print(f"{key}: F1={f1:.4f} AUC={auc:.4f} combo={0.6*f1+.4*auc:.4f}")
        store[f"oof_{key}"] = oof; store[f"test_{key}"] = prob_te

    np.savez_compressed("artifacts/zoo_final.npz", **store)

    # weight search over old + new zoos
    keys = ["xgb_d3_lr05", "xgb_d3_lr03", "lgbm_l15", "cat_d4",
            "n_xgb_d3_lr05", "n_xgb_d3_lr03", "n_lgbm_l15", "n_cat_d4"]
    def oof_key(k):
        return f"oof_{k}" if k.startswith("n_") else f"old_oof_{k}"
    def test_key(k):
        return f"test_{k}" if k.startswith("n_") else f"old_test_{k}"
    O = np.vstack([store[oof_key(k)] for k in keys])
    T = np.vstack([store[test_key(k)] for k in keys])

    def score(p):
        return 0.6 * f1_score(y, (p >= .5).astype(int)) + 0.4 * roc_auc_score(y, p)

    rng = np.random.default_rng(42)
    best = (-1, None)
    for _ in range(40000):
        w = rng.dirichlet(np.ones(len(keys)) * 2)
        s = score(w @ O)
        if s > best[0]:
            best = (s, w)
    print("\nBEST combo=%.4f" % best[0])
    print(dict(zip(keys, best[1].round(3))))
    np.savez_compressed("artifacts/v4_blend.npz", weights=best[1], keys=np.array(keys),
                        test_probs=best[1] @ T)


if __name__ == "__main__":
    main()
