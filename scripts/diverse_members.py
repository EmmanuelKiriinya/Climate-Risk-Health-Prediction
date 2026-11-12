"""
Round 15b — a member with a DIFFERENT inductive bias.

Every model in the project so far is a gradient-boosted tree, so the blend members
are highly correlated and blending buys almost nothing (Round 11: a trio whose
members shared a feature set scored BELOW its own best member). Age is the dominant
predictor and its effect is smooth and monotone-ish; trees approximate that with
steps. A spline-basis logistic regression gets it in a handful of parameters, with
completely different failure modes -- which is exactly what a blend wants.

Scored on the override-adjusted objective and on held-out fold seeds, same as
Round 15.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME
from harness import evaluate, combo, build
from optuna_v3 import aphrc_mask

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly",
        "gender_bin","zone_bin","year"]
HOLD = range(950, 970)

def spline_lr(C=1.0, n_knots=8, spw=1.4):
    ct = ColumnTransformer([
        ("age_spl", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False), ["age"]),
        ("yr_spl",  SplineTransformer(n_knots=5, degree=3, include_bias=False), ["year"]),
        ("pass", "passthrough", ["gender_bin","zone_bin","is_infant","is_under5"]),
    ])
    return make_pipeline(ct, StandardScaler(with_mean=False),
                         LogisticRegression(C=C, max_iter=2000,
                                            class_weight={0:1.0, 1:spw}))

def oof_sklearn(model_fn, X, y, groups, seeds):
    """OOF for a plain sklearn estimator, same folding as harness.evaluate."""
    out = np.zeros((len(seeds), len(y)))
    for i, fs in enumerate(seeds):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fs)
        for tri, tei in skf.split(X, y, groups):
            m = model_fn()
            m.fit(X.iloc[tri], y.iloc[tri])
            out[i, tei] = m.predict_proba(X.iloc[tei])[:, 1]
    return out

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    X0 = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    V4tr, _ = build_v4(tr, te)
    X = pd.concat([X0, V4tr], axis=1)
    mask, dlab = aphrc_mask(tr, te)

    def adj(o):
        s = []
        for i in range(o.shape[0]):
            p = o[i].copy(); p[mask] = np.where(dlab[mask]==1, .999, .001)
            s.append(combo(y, p))
        return float(np.mean(s)), float(np.std(s, ddof=1)/np.sqrt(len(s)))

    LGBP=dict(learning_rate=0.011374471419103494,num_leaves=53,min_child_samples=41,
              subsample=0.868088589184808,subsample_freq=1,colsample_bytree=0.8288246650597182,
              reg_lambda=0.001222299102744292,reg_alpha=2.8346683989414245)
    XGBP=dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,
              colsample_bytree=.9,reg_lambda=1.)
    CATP=dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)

    oofs = {}
    for nm,fam,p,spw,rd,cols in [("lgbm_reg","lgbm",LGBP,1.3599198869839608,263,CORE+REGIME),
                                 ("cat_core","cat",CATP,1.86,250,CORE),
                                 ("xgb_v3","xgb",XGBP,1.70,100,CORE+V3_COLS)]:
        _,_,_,o,_ = evaluate(fam,p,spw,rd,X[cols],y,groups,fold_seeds=HOLD,model_seeds=(42,))
        oofs[nm]=o
    Xs = X[["age","year","gender_bin","zone_bin","is_infant","is_under5"]]
    for nm, fn in [("spline_lr", lambda: spline_lr(C=1.0, spw=1.4)),
                   ("spline_lr_C01", lambda: spline_lr(C=0.1, spw=1.4)),
                   ("extratrees", lambda: ExtraTreesClassifier(
                        n_estimators=500, min_samples_leaf=12, max_features=.6,
                        class_weight={0:1.,1:1.4}, random_state=42, n_jobs=-1))]:
        src = Xs if nm.startswith("spline") else X[CORE+REGIME]
        oofs[nm] = oof_sklearn(fn, src, y, groups, list(HOLD))

    print(f"{'member':16s} {'adj combo':>10s} {'se':>7s}   corr with v8 blend")
    v8 = sum(oofs[n] for n in ("lgbm_reg","cat_core","xgb_v3"))/3
    mv8, sv8 = adj(v8)
    for nm, o in oofs.items():
        m, se = adj(o)
        c = np.corrcoef(o.mean(0), v8.mean(0))[0,1]
        print(f"{nm:16s} {m:10.4f} {se:7.4f}   {c:.4f}")
    print(f"\n{'v8 blend':16s} {mv8:10.4f} {sv8:7.4f}")

    print("\nadding each member to the v8 blend:")
    best = (mv8, "none", 0.0)
    for nm in ("spline_lr","spline_lr_C01","extratrees"):
        for w in (0.10, 0.15, 0.20, 0.25, 0.33):
            m,_ = adj((1-w)*v8 + w*oofs[nm])
            flag = "  <-- beats v8" if m > mv8 else ""
            print(f"  v8*{1-w:.2f} + {nm}*{w:.2f}: {m:.4f} ({m-mv8:+.4f}){flag}")
            if m > best[0]: best = (m, nm, w)
    print(f"\nbest: {best[1]} w={best[2]} -> {best[0]:.4f} ({best[0]-mv8:+.4f} vs v8)")

if __name__ == "__main__":
    main()
