"""Extend zoo with 2 diverse tuned configs (7 seeds total), re-run weight search."""
import sys
import warnings

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data
from features_v2 import build_v2

SEEDS = [42, 7, 2026, 99, 5, 13, 88]


def main():
    tr_raw, te_raw = load_data()
    y = tr_raw.is_climate_sensitive.reset_index(drop=True)
    Xall, Xte_all, _ = build_v2(tr_raw, te_raw)
    cols = list(Xall.columns[:43]) + [c for c in Xall.columns if c.startswith("season_")]
    X, Xte = Xall[cols], Xte_all[cols]

    village = pd.factorize(tr_raw.location.str.split(",").str[:2].str.join(","))[0]
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, village))

    z = np.load("artifacts/zoo.npz")
    store = {k: z[k] for k in z.files}

    cfgs = {
        "xgb_d4_heavy": dict(learning_rate=0.0369, max_depth=4, min_child_weight=2, gamma=0.99,
                             reg_alpha=1.11, reg_lambda=2.42, subsample=0.85, colsample_bytree=0.71),
        "xgb_d2_stump": dict(learning_rate=0.0262, max_depth=2, min_child_weight=1, gamma=0.26,
                             reg_alpha=1.85, reg_lambda=1.87, subsample=0.75, colsample_bytree=0.79),
    }
    for key, p in cfgs.items():
        oof = np.zeros(len(y)); prob_te = np.zeros(len(Xte))
        for seed in SEEDS:
            oof_s = np.zeros(len(y)); te_s = np.zeros(len(Xte))
            for tr_i, va_i in splits:
                m = xgb.XGBClassifier(n_estimators=3000, scale_pos_weight=1.86, reg_alpha=p["reg_alpha"],
                                      reg_lambda=p["reg_lambda"], gamma=p["gamma"],
                                      learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                                      min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                                      colsample_bytree=p["colsample_bytree"], random_state=seed,
                                      eval_metric="logloss", early_stopping_rounds=100, verbosity=0)
                m.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[va_i], y.iloc[va_i])], verbose=False)
                oof_s[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
                te_s += m.predict_proba(Xte)[:, 1] / len(splits)
            oof += oof_s / len(SEEDS); prob_te += te_s / len(SEEDS)
        f1 = f1_score(y, (oof >= .5).astype(int)); auc = roc_auc_score(y, oof)
        print(f"{key}: F1={f1:.4f} AUC={auc:.4f} combo={0.6*f1+.4*auc:.4f}")
        store[f"oof_{key}"] = oof; store[f"test_{key}"] = prob_te

    np.savez_compressed("artifacts/zoo.npz", **store)

    # weight search over everything except histgb-heavy noise
    keys = ["xgb_d3_lr05", "xgb_d3_lr03", "lgbm_l15", "cat_d4", "histgb",
            "xgb_d4_heavy", "xgb_d2_stump"]
    O = np.vstack([store["oof_" + k] for k in keys])
    T = np.vstack([store["test_" + k] for k in keys])

    def score(p):
        return 0.6 * f1_score(y, (p >= .5).astype(int)) + 0.4 * roc_auc_score(y, p)

    rng = np.random.default_rng(42)
    best = (-1, None)
    for _ in range(40000):
        w = rng.dirichlet(np.ones(len(keys)) * 2)
        s = score(w @ O)
        if s > best[0]:
            best = (s, w)
    print("\nbest combo=%.4f" % best[0])
    print(dict(zip(keys, best[1].round(3))))
    np.savez_compressed("artifacts/v3_blend.npz", weights=best[1], keys=np.array(keys),
                        test_probs=best[1] @ T)


if __name__ == "__main__":
    main()
