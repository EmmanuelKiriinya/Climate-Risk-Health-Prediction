"""Compare CV schemes and feature blocks. Model: XGB depth3 (fast, representative).

Schemes: StratKFold | GroupVillage | GroupSpatialCluster | TimeSplit(<=2015 -> >2015)
Blocks:  v1 = base 43 | +spatial | +anomaly | +season | +interact
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data
from features_v2 import build_v2

SEED = 42


def combo(y, p):
    return 0.6 * f1_score(y, (p >= 0.5).astype(int)) + 0.4 * roc_auc_score(y, p)


def fit_xgb(Xtr, ytr, Xva, yva):
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=3, min_child_weight=1,
                          scale_pos_weight=1.86, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=1.0, random_state=SEED, eval_metric="logloss",
                          early_stopping_rounds=100, verbosity=0)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict_proba(Xva)[:, 1]


def oof_for(scheme_splits, X, y):
    oof = np.zeros(len(y))
    for tr_i, va_i in scheme_splits:
        oof[va_i] = fit_xgb(X.iloc[tr_i], y.iloc[tr_i], X.iloc[va_i], y.iloc[va_i])
    return oof


def main():
    tr_raw, te_raw = load_data()
    y = tr_raw.is_climate_sensitive.reset_index(drop=True)
    Xv1 = None
    from features import engineer
    Xv1 = engineer(tr_raw)
    Xv2, _, meta = build_v2(tr_raw, te_raw)

    village = pd.factorize(tr_raw.location.str.split(",").str[:2].str.join(","))[0]
    spatial_groups = np.zeros(len(tr_raw), int)
    # recover cluster ids used in v2 for grouping: recompute KMeans labels on train coords only
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(
        pd.concat([tr_raw, te_raw])[["latitude", "longitude"]].values)
    train_pos = np.arange(len(tr_raw))
    spatial_groups = km.labels_[:len(tr_raw)]

    years = tr_raw.deathdate.dt.year.values

    schemes = {}
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    schemes["StratKFold"] = list(skf.split(Xv2, y))
    sgk = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    schemes["GroupVillage"] = list(sgk.split(Xv2, y, village))
    sgk2 = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    schemes["GroupSpatial"] = list(sgk2.split(Xv2, y, spatial_groups))
    time_mask = years <= 2015
    schemes["TimeSplit"] = [(np.where(time_mask)[0], np.where(~time_mask)[0])]

    blocks = ["base", "spatial", "anomaly", "season", "interact"]
    cols = {b: list(Xv1.columns) if b == "base" else meta[b] for b in blocks}
    cumulative = {"base": list(Xv1.columns)}
    for b in blocks[1:]:
        cumulative[b] = cumulative[blocks[blocks.index(b) - 1]] + cols[b]

    print(f"{'blockset':28s} {'scheme':14s} {'F1':>7s} {'AUC':>7s} {'combo':>7s}")
    results = []
    for bname, bcols in cumulative.items():
        X = Xv2[bcols]
        for sname, splits in schemes.items():
            oof = oof_for(splits, X, y)
            f1 = f1_score(y, (oof >= 0.5).astype(int)); auc = roc_auc_score(y, oof); c = combo(y, oof)
            results.append((bname, sname, f1, auc, c))
            print(f"{bname:28s} {sname:14s} {f1:7.4f} {auc:7.4f} {c:7.4f}")
    res = pd.DataFrame(results, columns=["blockset", "scheme", "f1", "auc", "combo"])
    res.to_csv("artifacts/cv_scheme_results.csv", index=False)
    print("\nBest per scheme:")
    print(res.loc[res.groupby("scheme").combo.idxmax()].to_string(index=False))


if __name__ == "__main__":
    main()
