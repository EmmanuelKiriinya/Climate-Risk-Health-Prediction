"""Feature-set x family search under the honest harness.

Selection on fold seeds 0-9; finalists re-measured on held-out fold seeds 100-119
to control max-order-statistic inflation from searching many configs.
"""
import sys, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from harness import evaluate, parts

CORE = ["age", "age_log1p", "is_infant", "is_under5", "is_child", "is_elderly",
        "gender_bin", "zone_bin", "year"]
MIN  = ["age", "age_log1p", "gender_bin", "zone_bin", "year"]
CYC  = ["month_sin", "month_cos", "doy_sin", "doy_cos"]
TOPCLI = ["rain_sum_90d", "ndvi_90d", "tavg_obs_minus_rean", "max_daily_rain_30d", "tx_tavg_gap"]
SEL = range(10)

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    Xf = engineer(tr).reset_index(drop=True)
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    ALL = list(Xf.columns)

    fsets = {
        "min5":        MIN,
        "core9":       CORE,
        "core+cyc":    CORE + CYC,
        "core+topcli": CORE + TOPCLI,
        "core+cyc+topcli": CORE + CYC + TOPCLI,
        "full43":      ALL,
    }
    fams = {
        "xgb_d3":  ("xgb",  dict(learning_rate=0.05, max_depth=3, min_child_weight=1,
                                 subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0), 100),
        "xgb_d4":  ("xgb",  dict(learning_rate=0.05, max_depth=4, min_child_weight=5,
                                 subsample=0.85, colsample_bytree=0.9, reg_lambda=2.0), 80),
        "lgbm_l15":("lgbm", dict(learning_rate=0.05, num_leaves=15, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 reg_lambda=1.0), 120),
        "cat_d4":  ("cat",  dict(learning_rate=0.05, depth=4, l2_leaf_reg=3.0), 250),
        "hgb":     ("hgb",  dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                 l2_regularization=1.0), 150),
    }

    rows = []
    print(f"{'fset':18s} {'fam':9s} {'mean':>7s} {'SE':>7s} {'F1':>6s} {'AUC':>6s}")
    for fn, cols in fsets.items():
        for mn, (fam, p, rounds) in fams.items():
            m, se, sc, oofs, _ = evaluate(fam, p, 1.86, rounds, Xf[cols], y, groups, fold_seeds=SEL)
            f1, auc = parts(y, oofs.mean(0))
            rows.append((fn, mn, m, se, f1, auc))
            print(f"{fn:18s} {mn:9s} {m:7.4f} {se:7.4f} {f1:6.4f} {auc:6.4f}")
    res = pd.DataFrame(rows, columns=["fset","fam","mean","se","f1","auc"]).sort_values("mean", ascending=False)
    res.to_csv("artifacts/search.csv", index=False)
    print("\nTOP-8 (selection seeds 0-9):"); print(res.head(8).to_string(index=False))

    print("\n=== CONFIRMATION on held-out fold seeds 100-119 ===")
    conf = []
    for _, r in res.head(8).iterrows():
        fam, p, rounds = fams[r.fam]
        m, se, sc, oofs, _ = evaluate(fam, p, 1.86, rounds, Xf[fsets[r.fset]], y, groups,
                                      fold_seeds=range(100, 120))
        conf.append((r.fset, r.fam, r["mean"], m, se))
        print(f"{r.fset:18s} {r.fam:9s} sel={r['mean']:.4f}  holdout={m:.4f} +/- {se:.4f}")
    pd.DataFrame(conf, columns=["fset","fam","sel","holdout","holdout_se"]).to_csv(
        "artifacts/search_confirm.csv", index=False)

if __name__ == "__main__":
    main()
