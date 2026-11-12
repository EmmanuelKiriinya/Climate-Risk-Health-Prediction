"""Re-tune spw for the v3 feature sets: they raise AUC (+0.0015) but shift the 0.5 cut."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from harness import evaluate, parts

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
MIN  = ["age","age_log1p","gender_bin","zone_bin","year"]
XGB  = dict(learning_rate=0.05, max_depth=3, min_child_weight=1,
            subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0)

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    Xf = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]

    sets = {"core9": CORE, "core9+v3": CORE + V3_COLS, "min5+v3": MIN + V3_COLS}
    rows = []
    print(f"{'fset':10s} {'spw':>5s} {'sel':>7s} {'SE':>7s} {'holdout':>8s} {'SE':>7s} {'F1':>6s} {'AUC':>6s} {'pos':>6s}")
    for name, cols in sets.items():
        for spw in [1.5, 1.7, 1.86, 2.0, 2.2, 2.4]:
            m1, se1, _, _, _ = evaluate("xgb", XGB, spw, 100, Xf[cols], y, groups, fold_seeds=range(10))
            m2, se2, _, o2, _ = evaluate("xgb", XGB, spw, 100, Xf[cols], y, groups, fold_seeds=range(100,120))
            f1, auc = parts(y, o2.mean(0)); pr = (o2.mean(0) >= .5).mean()
            rows.append((name, spw, m1, se1, m2, se2, f1, auc, pr))
            print(f"{name:10s} {spw:5.2f} {m1:7.4f} {se1:7.4f} {m2:8.4f} {se2:7.4f} {f1:6.4f} {auc:6.4f} {pr:6.3f}")
    res = pd.DataFrame(rows, columns=["fset","spw","sel","sel_se","holdout","hold_se","f1","auc","posrate"])
    res.to_csv("artifacts/spw_v3.csv", index=False)
    print("\nTOP-5 by holdout:"); print(res.nlargest(5, "holdout").to_string(index=False))

if __name__ == "__main__":
    main()
