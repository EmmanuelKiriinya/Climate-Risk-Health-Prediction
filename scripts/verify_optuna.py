"""Third-seed verification of the Optuna winner + focused spw re-sweep + new blend.

Fold seeds 100-119 were used to RANK the top-10, so max-over-10 on those seeds is
itself mildly inflated. Seeds 300-319 have never been touched.
"""
import sys, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME, CTIME
from harness import evaluate, combo, parts

CORE=["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP=dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP=dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
LGBP=dict(learning_rate=0.011374471419103494,num_leaves=53,min_child_samples=41,
          subsample=0.868088589184808,subsample_freq=1,colsample_bytree=0.8288246650597182,
          reg_lambda=0.001222299102744292,reg_alpha=2.8346683989414245)
LGB_ROUNDS=263; LGB_SPW=1.3599198869839608

def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True)
    g=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    X0=add_v3(engineer(tr).reset_index(drop=True),tr.reset_index(drop=True))
    V4,_=build_v4(tr,te); X=pd.concat([X0,V4],axis=1)
    CR = CORE+REGIME

    print("=== 1. third-seed verification (fold seeds 300-319, never used) ===")
    for nm,fam,p,spw,rd,cols in [
        ("optuna lgbm core_regime","lgbm",LGBP,LGB_SPW,LGB_ROUNDS,CR),
        ("xgb_core (incumbent)","xgb",XGBP,1.86,100,CORE),
        ("cat_core","cat",CATP,1.86,250,CORE),
        ("xgb_v3","xgb",XGBP,1.70,100,CORE+V3_COLS)]:
        m,se,_,o,_=evaluate(fam,p,spw,rd,X[cols],y,g,fold_seeds=range(300,320))
        f1,auc=parts(y,o.mean(0))
        print(f"  {nm:26s} {m:.4f} +/- {se:.4f}   F1={f1:.4f} AUC={auc:.4f}")

    print("\n=== 2. spw re-sweep for the Optuna config (was fixed at 1.86 in earlier sweeps) ===")
    for spw in [1.15,1.25,1.36,1.5,1.65,1.86,2.1]:
        m1,_,_,_,_=evaluate("lgbm",LGBP,spw,LGB_ROUNDS,X[CR],y,g,fold_seeds=range(10))
        m2,s2,_,o2,_=evaluate("lgbm",LGBP,spw,LGB_ROUNDS,X[CR],y,g,fold_seeds=range(300,320))
        f1,auc=parts(y,o2.mean(0))
        print(f"  spw={spw:4.2f}  sel={m1:.4f}  verify={m2:.4f} +/- {s2:.4f}  F1={f1:.4f} AUC={auc:.4f}")

    print("\n=== 3. blend search including the new member ===")
    MEM={"lgbm_reg":("lgbm",LGBP,LGB_SPW,LGB_ROUNDS,CR),
         "xgb_core":("xgb",XGBP,1.86,100,CORE),
         "cat_core":("cat",CATP,1.86,250,CORE),
         "xgb_v3":("xgb",XGBP,1.70,100,CORE+V3_COLS)}
    so,ho={},{}
    for k,(fam,p,spw,rd,cols) in MEM.items():
        _,_,_,a,_=evaluate(fam,p,spw,rd,X[cols],y,g,fold_seeds=range(10))
        _,_,_,b,_=evaluate(fam,p,spw,rd,X[cols],y,g,fold_seeds=range(300,320))
        so[k],ho[k]=a,b
    keys=list(MEM)
    def bs(w,st):
        return float(np.mean([combo(y,sum(w[i]*st[k][s] for i,k in enumerate(keys)))
                              for s in range(st[keys[0]].shape[0])]))
    def wv(d):
        w=np.array([d.get(k,0.) for k in keys]); return w/w.sum()
    cands={"v7 trio (current)":wv({"xgb_core":1,"cat_core":1,"xgb_v3":1}),
           "lgbm_reg alone":wv({"lgbm_reg":1}),
           "lgbm_reg + xgb_core":wv({"lgbm_reg":1,"xgb_core":1}),
           "lgbm_reg + v7 trio (equal 4)":wv({k:1 for k in keys}),
           "lgbm_reg x2 + v7 trio":wv({"lgbm_reg":3,"xgb_core":1,"cat_core":1,"xgb_v3":1}),
           "lgbm_reg + cat_core + xgb_v3":wv({"lgbm_reg":1,"cat_core":1,"xgb_v3":1})}
    print(f"  {'weighting':32s} {'sel(0-9)':>9s} {'verify(300-319)':>16s} {'drift':>8s}")
    rows=[]
    for nm,w in cands.items():
        a,b=bs(w,so),bs(w,ho); rows.append((nm,a,b,b-a))
        print(f"  {nm:32s} {a:9.4f} {b:16.4f} {b-a:+8.4f}")
    pd.DataFrame(rows,columns=["weighting","sel","verify","drift"]).to_csv("artifacts/blend3.csv",index=False)
    np.savez_compressed("artifacts/blend3_oofs.npz", y=y.values,
        **{f"sel_{k}":v for k,v in so.items()}, **{f"ver_{k}":v for k,v in ho.items()})

if __name__=="__main__":
    main()
