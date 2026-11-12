"""Test the corrected age-band x era interactions + district target-encoding.

Both are evaluated under the honest harness (fixed rounds, 10 fold seeds),
with a held-out confirmation pass on fold seeds 100-119.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from harness import evaluate, parts

CORE = ["age", "age_log1p", "is_infant", "is_under5", "is_child", "is_elderly",
        "gender_bin", "zone_bin", "year"]
MIN  = ["age", "age_log1p", "gender_bin", "zone_bin", "year"]
XGB  = ("xgb", dict(learning_rate=0.05, max_depth=3, min_child_weight=1,
                    subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0), 100)
CAT  = ("cat", dict(learning_rate=0.05, depth=4, l2_leaf_reg=3.0), 250)

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    Xf = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]

    sets = {
        "min5":            MIN,
        "core9":           CORE,
        "min5+v3":         MIN + V3_COLS,
        "core9+v3":        CORE + V3_COLS,
        "core9+age0only":  CORE + ["is_age0", "post2018", "age0_x_post"],
        "min5+age0only":   MIN + ["is_age0", "post2018", "age0_x_post"],
        "core9+seasonA5p": CORE + ["a5p_month_sin", "a5p_month_cos"],
    }
    print(f"{'fset':18s} {'fam':4s} {'sel(0-9)':>9s} {'SE':>7s} {'holdout(100-119)':>17s} {'SE':>7s} {'F1':>6s} {'AUC':>6s}")
    rows = []
    for name, cols in sets.items():
        for fname, (fam, p, rounds) in [("xgb", XGB), ("cat", CAT)]:
            m1, se1, _, o1, _ = evaluate(fam, p, 1.86, rounds, Xf[cols], y, groups, fold_seeds=range(10))
            m2, se2, _, o2, _ = evaluate(fam, p, 1.86, rounds, Xf[cols], y, groups, fold_seeds=range(100, 120))
            f1, auc = parts(y, o2.mean(0))
            rows.append((name, fname, m1, se1, m2, se2, f1, auc))
            print(f"{name:18s} {fname:4s} {m1:9.4f} {se1:7.4f} {m2:17.4f} {se2:7.4f} {f1:6.4f} {auc:6.4f}")
    pd.DataFrame(rows, columns=["fset","fam","sel","sel_se","holdout","holdout_se","f1","auc"]
                 ).to_csv("artifacts/test_v3.csv", index=False)

if __name__ == "__main__":
    main()
