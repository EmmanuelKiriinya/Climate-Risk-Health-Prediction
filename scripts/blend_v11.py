"""
Round 16 — the blend was costing us. Weight search + fresh-seed verification.

Round 15b turned up something the plain objective had hidden: on the
override-adjusted metric, lgbm_reg ALONE (0.8492) beats the v8 equal-thirds blend
(0.8484) on held-out seeds. In Round 10 those two were a tie (-0.0002 +/- 0.0003)
on the un-overridden objective. Pinning 10.6% of rows changes which member wins.

This searches weights over 4 members on seeds 950-969, then re-verifies the
finalists on seeds 1100-1139, which nothing has touched. The second pass is the
point: picking the max over ~30 weightings on one seed set is exactly the
mechanism that produced v5.
"""
import sys, warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from sklearn.ensemble import ExtraTreesClassifier
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME
from harness import evaluate, combo, parts
from optuna_v3 import aphrc_mask
from diverse_members import oof_sklearn

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly",
        "gender_bin","zone_bin","year"]
LGBP=dict(learning_rate=0.011374471419103494,num_leaves=53,min_child_samples=41,
          subsample=0.868088589184808,subsample_freq=1,colsample_bytree=0.8288246650597182,
          reg_lambda=0.001222299102744292,reg_alpha=2.8346683989414245)
XGBP=dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,
          colsample_bytree=.9,reg_lambda=1.)
CATP=dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
SEL, VER = range(950, 970), range(1100, 1140)

def et(): return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=12,
                                      max_features=.6, class_weight={0:1.,1:1.4},
                                      random_state=42, n_jobs=-1)

def get_oofs(X, y, groups, seeds):
    o = {}
    for nm,fam,p,spw,rd,cols in [("lgbm","lgbm",LGBP,1.3599198869839608,263,CORE+REGIME),
                                 ("cat","cat",CATP,1.86,250,CORE),
                                 ("xgb","xgb",XGBP,1.70,100,CORE+V3_COLS)]:
        _,_,_,oo,_ = evaluate(fam,p,spw,rd,X[cols],y,groups,fold_seeds=seeds,model_seeds=(42,))
        o[nm]=oo
    o["et"] = oof_sklearn(et, X[CORE+REGIME], y, groups, list(seeds))
    return o

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    X0 = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    V4tr,_ = build_v4(tr, te); X = pd.concat([X0,V4tr],axis=1)
    mask, dlab = aphrc_mask(tr, te)
    def adj(o):
        s=[]
        for i in range(o.shape[0]):
            p=o[i].copy(); p[mask]=np.where(dlab[mask]==1,.999,.001); s.append(combo(y,p))
        return float(np.mean(s)), float(np.std(s,ddof=1)/np.sqrt(len(s)))

    grid = [w for w in itertools.product(np.arange(0,1.01,.1), repeat=4)
            if abs(sum(w)-1) < 1e-9]
    print(f"searching {len(grid)} weightings on seeds {SEL.start}-{SEL.stop-1}")
    o = get_oofs(X, y, groups, SEL)
    names = ["lgbm","cat","xgb","et"]
    rows = []
    for w in grid:
        b = sum(wi*o[n] for wi,n in zip(w,names))
        m,_ = adj(b); rows.append((*w, m))
    df = pd.DataFrame(rows, columns=names+["adj"]).sort_values("adj", ascending=False)
    v8_sel,_ = adj((o["lgbm"]+o["cat"]+o["xgb"])/3)
    lg_sel,_ = adj(o["lgbm"])
    print(f"\nv8 equal-thirds : {v8_sel:.4f}\nlgbm alone      : {lg_sel:.4f}")
    print("\ntop 10 weightings:"); print(df.head(10).to_string(index=False))
    df.to_csv("artifacts/blend_v11_search.csv", index=False)

    # finalists: search winner, lgbm alone, v8, plus a couple of robust picks
    cands = {"search_best": tuple(df.iloc[0][names]),
             "lgbm_only":   (1.,0.,0.,0.),
             "v8_thirds":   (1/3,1/3,1/3,0.),
             "lgbm_et_7030":(0.7,0.,0.,0.3),
             "lgbm_et_5050":(0.5,0.,0.,0.5),
             "lgbm_xgb_et": (0.5,0.,0.2,0.3)}
    print(f"\nVERIFY on untouched seeds {VER.start}-{VER.stop-1}")
    ov = get_oofs(X, y, groups, VER)
    out=[]
    for nm,w in cands.items():
        b = sum(wi*ov[n] for wi,n in zip(w,names))
        m,se = adj(b)
        bs = sum(wi*o[n] for wi,n in zip(w,names)); ms,_ = adj(bs)
        out.append(dict(cand=nm, weights=str([round(x,2) for x in w]),
                        sel=round(ms,4), verify=round(m,4), se=round(se,4),
                        drift=round(m-ms,4)))
        print(f"  {nm:14s} {str([round(x,2) for x in w]):22s} sel {ms:.4f} -> verify {m:.4f} ({m-ms:+.4f})")
    pd.DataFrame(out).to_csv("artifacts/blend_v11_verify.csv", index=False)

if __name__ == "__main__":
    main()
