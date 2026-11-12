"""v7: equal-weight blend of 3 decorrelated core-feature members.

Selected under the honest harness (fixed rounds, no ES-on-validation, repeated
grouped CV). Equal weights beat Dirichlet-fitted weights on held-out fold seeds,
so weights are FIXED at 1/3 -- no weight fitting, nothing to overfit.

  member      features    family                    spw   rounds
  xgb_core    core9       XGB  d3 lr.05             1.86  100
  cat_core    core9       Cat  d4 lr.05             1.86  250
  xgb_v3      core9+v3    XGB  d3 lr.05             1.70  100

Honest repeated-CV combo: 0.8229 (sel seeds 0-9 AND holdout seeds 100-119).
Test probs are bagged over 20 fresh fold seeds x 5 folds x 3 model seeds.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from harness import evaluate, combo, parts, build

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP = dict(learning_rate=.05, max_depth=3, min_child_weight=1,
            subsample=.85, colsample_bytree=.9, reg_lambda=1.)
CATP = dict(learning_rate=.05, depth=4, l2_leaf_reg=3.)
PRED_SEEDS = range(200, 220)      # fresh: not used for selection or confirmation
MODEL_SEEDS = (42, 7, 2026)

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    Xtr = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    Xte = add_v3(engineer(te).reset_index(drop=True), te.reset_index(drop=True))
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    V3 = CORE + V3_COLS

    members = [("xgb_core", "xgb", XGBP, 1.86, 100, CORE),
               ("cat_core", "cat", CATP, 1.86, 250, CORE),
               ("xgb_v3",   "xgb", XGBP, 1.70, 100, V3)]

    oofs, tests = {}, {}
    for name, fam, p, spw, rounds, cols in members:
        m, se, _, o, t = evaluate(fam, p, spw, rounds, Xtr[cols], y, groups,
                                  fold_seeds=PRED_SEEDS, model_seeds=MODEL_SEEDS,
                                  Xte=Xte[cols])
        oofs[name], tests[name] = o, t
        f1, auc = parts(y, o.mean(0))
        print(f"{name:9s} combo={m:.4f} +/- {se:.4f}  F1={f1:.4f} AUC={auc:.4f}")

    W = 1.0 / 3
    blend_oof = sum(W * oofs[n] for n in oofs)
    per_seed = [combo(y, blend_oof[s]) for s in range(blend_oof.shape[0])]
    f1, auc = parts(y, blend_oof.mean(0))
    print(f"\nBLEND (equal 1/3) combo={np.mean(per_seed):.4f} +/- "
          f"{np.std(per_seed, ddof=1)/np.sqrt(len(per_seed)):.4f}  F1={f1:.4f} AUC={auc:.4f}")

    proba = sum(W * tests[n] for n in tests)

    # cross-check: full-data refits at the same fixed rounds
    full = np.zeros(len(Xte))
    for name, fam, p, spw, rounds, cols in members:
        acc = np.zeros(len(Xte))
        for s in MODEL_SEEDS:
            mdl = build(fam, p, spw, rounds, s)
            mdl.fit(Xtr[cols], y)
            acc += mdl.predict_proba(Xte[cols])[:, 1] / len(MODEL_SEEDS)
        full += W * acc
    print(f"bagged-vs-fulldata test probs: corr={np.corrcoef(proba, full)[0,1]:.4f} "
          f"binary agreement={( (proba>=.5)==(full>=.5) ).mean():.4f}")

    sub = pd.DataFrame({"ID": te.ID, "TargetF1": (proba >= .5).astype(int),
                        "TargetRAUC": proba.astype(float)})
    # integrity assertions (skills.md section 1)
    assert list(sub.columns) == ["ID", "TargetF1", "TargetRAUC"]
    assert sub.ID.equals(te.ID)
    assert sub.TargetF1.isin([0, 1]).all()
    assert (((sub.TargetRAUC >= .5).astype(int)) == sub.TargetF1).all()
    assert not sub.isna().any().any()
    assert sub.TargetRAUC.between(0, 1).all()
    sub.to_csv("submissions/submission_v7.csv", index=False)

    print(f"\nsubmission_v7.csv  n={len(sub)}  pos-rate={sub.TargetF1.mean():.4f}  "
          f"prob=[{proba.min():.4f},{proba.max():.4f}]  ALL ASSERTIONS PASSED")
    for v in ["submissions/submission_v4.csv", "submissions/submission_v5.csv", "submissions/submission_v6.csv"]:
        prev = pd.read_csv(v)
        print(f"  vs {v}: {(sub.TargetF1 != prev.TargetF1).sum():3d} binary rows differ, "
              f"prob corr={np.corrcoef(sub.TargetRAUC, prev.TargetRAUC)[0,1]:.4f}")
    np.savez_compressed("artifacts/v7.npz", test_probs=proba, full_probs=full,
                        **{f"oof_{k}": v for k, v in oofs.items()})

if __name__ == "__main__":
    main()
