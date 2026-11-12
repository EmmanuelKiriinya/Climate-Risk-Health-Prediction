"""Evaluate core9 + PCA blocks under the honest harness (sel + holdout fold seeds)."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data
from pca_features import build_pca_blocks
from harness import evaluate, parts

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP = dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP = dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    Xtr, Xte, blocks = build_pca_blocks(tr, te)
    base = Xtr[CORE]

    print(f"{'features':22s} {'k':>2s} {'expvar':>7s} {'xgb sel':>8s} {'xgb hold':>9s} {'cat hold':>9s} {'F1':>6s} {'AUC':>6s}")
    rows = []
    m1,_,_,_,_ = evaluate("xgb", XGBP, 1.86, 100, base, y, groups, fold_seeds=range(10))
    m2,_,_,o2,_ = evaluate("xgb", XGBP, 1.86, 100, base, y, groups, fold_seeds=range(100,120))
    c2,_,_,co,_ = evaluate("cat", CATP, 1.86, 250, base, y, groups, fold_seeds=range(100,120))
    f1,auc = parts(y, o2.mean(0))
    print(f"{'core9 (baseline)':22s} {'-':>2s} {'-':>7s} {m1:8.4f} {m2:9.4f} {c2:9.4f} {f1:6.4f} {auc:6.4f}")
    rows.append(("core9", 0, np.nan, m1, m2, c2, f1, auc))

    for name, (a, b, ev) in blocks.items():
        X = pd.concat([base, a], axis=1)
        m1,_,_,_,_ = evaluate("xgb", XGBP, 1.86, 100, X, y, groups, fold_seeds=range(10))
        m2,_,_,o2,_ = evaluate("xgb", XGBP, 1.86, 100, X, y, groups, fold_seeds=range(100,120))
        c2,_,_,_,_ = evaluate("cat", CATP, 1.86, 250, X, y, groups, fold_seeds=range(100,120))
        f1,auc = parts(y, o2.mean(0))
        rows.append((name, a.shape[1], ev, m1, m2, c2, f1, auc))
        print(f"{'core9+'+name:22s} {a.shape[1]:2d} {ev:7.3f} {m1:8.4f} {m2:9.4f} {c2:9.4f} {f1:6.4f} {auc:6.4f}")
    res = pd.DataFrame(rows, columns=["features","k","expvar","xgb_sel","xgb_hold","cat_hold","f1","auc"])
    res.to_csv("artifacts/pca_results.csv", index=False)
    print("\nTOP-6 by xgb holdout:"); print(res.nlargest(6,"xgb_hold").to_string(index=False))

if __name__ == "__main__":
    main()
