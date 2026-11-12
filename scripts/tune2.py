"""Refined random search around winning configs (GroupVillage CV)."""
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

SEED = 42


def main():
    tr_raw, te_raw = load_data()
    y = tr_raw.is_climate_sensitive.reset_index(drop=True)
    Xall, _, _ = build_v2(tr_raw, te_raw)
    cols = list(Xall.columns[:43]) + [c for c in Xall.columns if c.startswith("season_")]
    X = Xall[cols]

    village = pd.factorize(tr_raw.location.str.split(",").str[:2].str.join(","))[0]
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, village))

    rng = np.random.default_rng(SEED)
    rows = []

    def cv(fit_predict):
        oof = np.zeros(len(y))
        for tr_i, va_i in splits:
            oof[va_i] = fit_predict(X.iloc[tr_i], y.iloc[tr_i], X.iloc[va_i], y.iloc[va_i])
        f1 = f1_score(y, (oof >= .5).astype(int)); auc = roc_auc_score(y, oof)
        return f1, auc

    # XGB search
    for _ in range(24):
        p = dict(
            learning_rate=float(rng.uniform(.02, .08)),
            max_depth=int(rng.integers(2, 5)),
            min_child_weight=int(rng.choice([1, 2, 5, 10])),
            gamma=float(rng.uniform(0, 1.5)),
            reg_alpha=float(rng.uniform(0, 2)),
            reg_lambda=float(rng.uniform(.3, 3)),
            subsample=float(rng.uniform(.65, .95)),
            colsample_bytree=float(rng.uniform(.55, .95)),
        )
        spw = float(rng.choice([1.5, 1.86, 2.2]))
        def fp(Xtr, ytr, Xva, yva, p=p, spw=spw):
            m = xgb.XGBClassifier(n_estimators=3000, scale_pos_weight=spw, random_state=SEED,
                                  eval_metric="logloss", early_stopping_rounds=100, verbosity=0, **p)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            return m.predict_proba(Xva)[:, 1]
        f1, auc = cv(fp)
        rows.append(("xgb", json.dumps({**p, "spw": spw}), f1, auc, .6 * f1 + .4 * auc))
        print(f"xgb {p['learning_rate']:.3f} d{p['max_depth']} mcw{p['min_child_weight']} "
              f"g{p['gamma']:.2f} a{p['reg_alpha']:.2f} l{p['reg_lambda']:.2f} "
              f"ss{p['subsample']:.2f} cs{p['colsample_bytree']:.2f} spw{spw} -> {rows[-1][4]:.4f}")

    res = pd.DataFrame(rows, columns=["fam", "params", "f1", "auc", "combo"])
    res.to_csv("artifacts/tune2_results.csv", index=False)
    print("\nTOP-5:")
    print(res.nlargest(5, "combo").to_string(index=False))


if __name__ == "__main__":
    import json
    main()
