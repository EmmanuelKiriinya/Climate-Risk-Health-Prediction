"""Round-count x learning-rate x depth x spw sweep on core features (honest harness)."""
import sys, time, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from harness import evaluate, parts

CORE = ["age", "age_log1p", "is_infant", "is_under5", "is_child", "is_elderly",
        "gender_bin", "zone_bin", "year"]
FS = range(10)

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    Xf = engineer(tr).reset_index(drop=True)
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    X = Xf[CORE]

    rows = []
    print(f"{'lr':>5s} {'d':>2s} {'rounds':>6s} {'spw':>5s} {'mean':>7s} {'SE':>7s} {'F1':>6s} {'AUC':>6s}")
    for lr, depth in [(0.05, 3), (0.05, 2), (0.10, 3), (0.03, 3), (0.05, 4)]:
        for rounds in (30, 60, 100, 150, 250):
            m, se, sc, oofs, _ = evaluate("xgb",
                dict(learning_rate=lr, max_depth=depth, min_child_weight=1,
                     subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0),
                1.86, rounds, X, y, groups, fold_seeds=FS)
            f1, auc = parts(y, oofs.mean(0))
            rows.append(("xgb", lr, depth, rounds, 1.86, m, se, f1, auc))
            print(f"{lr:5.2f} {depth:2d} {rounds:6d} {1.86:5.2f} {m:7.4f} {se:7.4f} {f1:6.4f} {auc:6.4f}")
    res = pd.DataFrame(rows, columns=["fam","lr","depth","rounds","spw","mean","se","f1","auc"])
    res.to_csv("artifacts/sweep1.csv", index=False)
    print("\nTOP-5:"); print(res.nlargest(5, "mean").to_string(index=False))

if __name__ == "__main__":
    main()
