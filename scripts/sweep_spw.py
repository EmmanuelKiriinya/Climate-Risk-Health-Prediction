"""Fine scale_pos_weight sweep. spw is the ONLY legitimate lever on the 0.5 cut."""
import sys, warnings
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
    print(f"{'spw':>5s} {'mean':>7s} {'SE':>7s} {'F1':>6s} {'AUC':>6s} {'posrate':>7s} {'implied_t':>9s}")
    for spw in [1.0, 1.15, 1.3, 1.4, 1.5, 1.6, 1.7, 1.86, 2.0, 2.2, 2.6, 3.0]:
        m, se, sc, oofs, _ = evaluate("xgb",
            dict(learning_rate=0.05, max_depth=3, min_child_weight=1,
                 subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0),
            spw, 100, X, y, groups, fold_seeds=FS)
        o = oofs.mean(0)
        f1, auc = parts(y, o)
        pr = (o >= .5).mean()
        rows.append((spw, m, se, f1, auc, pr))
        print(f"{spw:5.2f} {m:7.4f} {se:7.4f} {f1:6.4f} {auc:6.4f} {pr:7.4f} {1/(1+spw):9.4f}")
    res = pd.DataFrame(rows, columns=["spw","mean","se","f1","auc","posrate"])
    res.to_csv("artifacts/sweep_spw.csv", index=False)
    print("\nTOP-4:"); print(res.nlargest(4, "mean").to_string(index=False))

    # what would the F1-optimal threshold be on near-calibrated (spw=1) probs?
    _, _, _, oofs1, _ = evaluate("xgb",
        dict(learning_rate=0.05, max_depth=3, min_child_weight=1,
             subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0),
        1.0, 100, X, y, groups, fold_seeds=FS)
    from sklearn.metrics import f1_score
    o1 = oofs1.mean(0)
    ts = np.arange(.25, .65, .005)
    f1s = [f1_score(y, (o1 >= t).astype(int)) for t in ts]
    tstar = ts[int(np.argmax(f1s))]
    print(f"\nspw=1 probs: F1-optimal threshold t*={tstar:.3f} (F1={max(f1s):.4f}) "
          f"-> implied spw=(1-t*)/t*={((1-tstar)/tstar):.2f}")

if __name__ == "__main__":
    main()
