"""Noise-floor measurement + round-count sweep under the honest harness."""
import sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from harness import evaluate, combo, parts

CORE = ["age", "age_log1p", "is_infant", "is_under5", "is_child", "is_elderly",
        "gender_bin", "zone_bin", "year"]

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    Xf, Xfe = engineer(tr).reset_index(drop=True), engineer(te).reset_index(drop=True)
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    Xc, Xce = Xf[CORE], Xfe[CORE]
    print(f"train {Xf.shape} core {Xc.shape} pos {y.mean():.4f} groups {len(np.unique(groups))}\n")

    FS = range(10)
    print(f"{'config':46s} {'rounds':>6s} {'mean':>7s} {'SE':>7s} {'min':>7s} {'max':>7s} {'F1':>6s} {'AUC':>6s}  t")
    for tag, X in [("core", Xc), ("full", Xf)]:
        for rounds in (150, 300, 500, 800):
            t0 = time.time()
            m, se, sc, oofs, _ = evaluate("xgb",
                dict(learning_rate=0.05, max_depth=3, min_child_weight=1,
                     subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0),
                1.86, rounds, X, y, groups, fold_seeds=FS)
            f1, auc = parts(y, oofs.mean(0))
            print(f"xgb_d3_lr05 spw1.86 [{tag}]{'':<20s} {rounds:6d} {m:7.4f} {se:7.4f} "
                  f"{sc.min():7.4f} {sc.max():7.4f} {f1:6.4f} {auc:6.4f}  {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
