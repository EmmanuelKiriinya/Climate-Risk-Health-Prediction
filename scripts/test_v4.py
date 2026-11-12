"""Evaluate v4 density / regime / continuous-time blocks under the honest harness."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, DENSITY, REGIME, CTIME
from harness import evaluate, parts

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP = dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP = dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    X0 = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    V4, _ = build_v4(tr, te)
    X = pd.concat([X0, V4], axis=1)

    sets = {
      "core9 (baseline)":       CORE,
      "core9 + ctime":          CORE + CTIME,
      "core9 + density":        CORE + DENSITY,
      "core9 + regime":         CORE + REGIME,
      "core9 + v4(all)":        CORE + CTIME + DENSITY + REGIME,
      "core9 + ctime + regime": CORE + CTIME + REGIME,
      "core9 + v3":             CORE + V3_COLS,
      "core9 + v3 + ctime":     CORE + V3_COLS + CTIME,
      "core9 + v3 + regime":    CORE + V3_COLS + REGIME,
      "core9 + v3 + density":   CORE + V3_COLS + DENSITY,
    }
    print(f"{'feature set':26s} {'xgb sel':>8s} {'xgb hold':>9s} {'SE':>7s} {'cat hold':>9s} {'F1':>6s} {'AUC':>6s}")
    rows=[]
    for nm, cols in sets.items():
        m1,_,_,_,_  = evaluate("xgb", XGBP, 1.86, 100, X[cols], y, groups, fold_seeds=range(10))
        m2,s2,_,o2,_= evaluate("xgb", XGBP, 1.86, 100, X[cols], y, groups, fold_seeds=range(100,120))
        c2,_,_,_,_  = evaluate("cat", CATP, 1.86, 250, X[cols], y, groups, fold_seeds=range(100,120))
        f1,auc = parts(y, o2.mean(0))
        rows.append((nm,len(cols),m1,m2,s2,c2,f1,auc))
        print(f"{nm:26s} {m1:8.4f} {m2:9.4f} {s2:7.4f} {c2:9.4f} {f1:6.4f} {auc:6.4f}")
    res=pd.DataFrame(rows,columns=["fset","ncols","xgb_sel","xgb_hold","se","cat_hold","f1","auc"])
    res.to_csv("artifacts/test_v4.csv",index=False)
    print("\nTOP-5 by xgb holdout:"); print(res.nlargest(5,"xgb_hold").to_string(index=False))

if __name__ == "__main__":
    main()
