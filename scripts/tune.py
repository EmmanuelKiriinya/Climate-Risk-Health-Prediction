"""Hyperparameter sweep under GroupCV + feature importance. Saves OOF probs for the best configs."""
import itertools
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer

SEED = 42
N_SPLITS = 5


def combo(y, p):
    return 0.6 * f1_score(y, (p >= 0.5).astype(int)) + 0.4 * roc_auc_score(y, p)


def run_cv(fit_predict, X, y, groups):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in cv.split(X, y, groups):
        oof[va_idx] = fit_predict(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx])
    return oof


def lgb_fp(p):
    def fit_predict(Xtr, ytr, Xva, yva):
        m = lgb.LGBMClassifier(random_state=SEED, verbose=-1, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m.predict_proba(Xva)[:, 1]
    return fit_predict


def xgb_fp(p):
    def fit_predict(Xtr, ytr, Xva, yva):
        m = xgb.XGBClassifier(random_state=SEED, eval_metric="logloss",
                              early_stopping_rounds=100, verbosity=0, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m.predict_proba(Xva)[:, 1]
    return fit_predict


def cat_fp(p):
    def fit_predict(Xtr, ytr, Xva, yva):
        m = CatBoostClassifier(random_seed=SEED, verbose=False, allow_writing_files=False,
                               early_stopping_rounds=100, **p)
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
        return m.predict_proba(Xva)[:, 1]
    return fit_predict


def main():
    tr, te = load_data()
    X = engineer(tr)
    y = tr.is_climate_sensitive.reset_index(drop=True)
    village = tr.location.str.split(",").str[:2].str.join(",")
    groups = pd.factorize(village)[0]

    grid = []
    # LGBM variants
    for leaves, lr, spw, mcs in itertools.product([15, 31], [0.03, 0.05], [1.86, 2.4], [20, 40]):
        grid.append(("lgbm", lgb_fp(dict(n_estimators=3000, learning_rate=lr, num_leaves=leaves,
                                         scale_pos_weight=spw, min_child_samples=mcs,
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                         reg_lambda=1.0)), p := dict(n_estimators=3000, learning_rate=lr,
                                         num_leaves=leaves, scale_pos_weight=spw, min_child_samples=mcs)))
    # XGB variants
    for depth, lr, spw, mcw in itertools.product([3, 5], [0.03, 0.05], [1.86, 2.4], [1, 5]):
        grid.append(("xgb", xgb_fp(dict(n_estimators=3000, learning_rate=lr, max_depth=depth,
                                        scale_pos_weight=spw, min_child_weight=mcw,
                                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)),
                     dict(depth=depth, lr=lr, spw=spw, mcw=mcw)))
    # CatBoost variants
    for depth, lr, spw in itertools.product([4, 6], [0.03, 0.05], [1.86]):
        grid.append(("cat", cat_fp(dict(iterations=3000, learning_rate=lr, depth=depth,
                                        scale_pos_weight=spw, l2_leaf_reg=3.0)),
                     dict(depth=depth, lr=lr, spw=spw)))

    rows, oofs = [], {}
    for name, fp, params in grid:
        oof = run_cv(fp, X, y, groups)
        f1 = f1_score(y, (oof >= 0.5).astype(int))
        auc = roc_auc_score(y, oof)
        key = f"{name}:{json.dumps(params, default=str)}"[:90]
        oofs[key] = oof
        rows.append((key, f1, auc, 0.6 * f1 + 0.4 * auc))
        print(f"{key:90s} F1={f1:.4f} AUC={auc:.4f} combo={0.6*f1+0.4*auc:.4f}")

    res = pd.DataFrame(rows, columns=["config", "f1", "auc", "combo"]).sort_values("combo", ascending=False)
    res.to_csv("artifacts/tune_results.csv", index=False)

    # blend search over top-6 OOFs
    top = res.head(6).index
    mats = np.vstack([oofs[res.loc[i, "config"]] for i in top])
    best = (-1, None)
    for w in itertools.product([0, .25, .5, .75, 1], repeat=len(top)):
        w = np.array(w, float)
        if w.sum() == 0:
            continue
        b = (w[:, None] * mats).sum(0) / w.sum()
        c = combo(y, b)
        if c > best[0]:
            best = (c, w / w.sum())
    print("\nBest blend:", dict(zip(res.loc[top, 'config'], best[1].round(2))), f"combo={best[0]:.4f}")

    # feature importance from a full-data LGBM
    m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=31,
                           scale_pos_weight=2.4, min_child_samples=40, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=SEED, verbose=-1)
    m.fit(X, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("\nTop-20 gain importances:")
    print(imp.head(20).round(0).to_string())

    import os
    os.makedirs("artifacts", exist_ok=True)
    np.savez_compressed("artifacts/oofs.npz",
                        y=y.values, **{k: v for k, v in oofs.items() if k in set(res.head(6).config)})
    with open("artifacts/best_blend.json", "w") as f:
        json.dump({"configs": list(res.head(6).config), "weights": best[1].tolist()}, f, indent=2)


if __name__ == "__main__":
    main()
