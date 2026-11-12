"""v8: Optuna-found LightGBM(core+regime) + CatBoost(core9) + XGB(core9+v3), equal 1/3.

The Optuna winner beat the v7 trio by +0.0013 on fold seeds it had never seen. Its
two departures from every earlier config: it uses the v4 REGIME block (dismissed as
"neutral" when tested with fixed XGB hyperparameters) and spw ~1.36 rather than 1.86
(the earlier spw sweep was run for ONE config and over-generalised -- at spw 1.86
this model scores 0.8217, at 1.36 it scores 0.8242).

Seeds 0-9 chose the search, 100-119 ranked the top-10, 300-319 verified and picked
the blend. 500-519 -- used here -- have never been touched, so the number printed
below is clean.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME
from harness import evaluate, combo, parts, build

CORE=["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP=dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP=dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
LGBP=dict(learning_rate=0.011374471419103494,num_leaves=53,min_child_samples=41,
          subsample=0.868088589184808,subsample_freq=1,colsample_bytree=0.8288246650597182,
          reg_lambda=0.001222299102744292,reg_alpha=2.8346683989414245)
PRED=range(500,520); MSEEDS=(42,7,2026)

def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True)
    g=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    Xtr0=add_v3(engineer(tr).reset_index(drop=True),tr.reset_index(drop=True))
    Xte0=add_v3(engineer(te).reset_index(drop=True),te.reset_index(drop=True))
    V4tr,V4te=build_v4(tr,te)
    Xtr=pd.concat([Xtr0,V4tr],axis=1); Xte=pd.concat([Xte0,V4te],axis=1)
    CR=CORE+REGIME

    members=[("lgbm_reg","lgbm",LGBP,1.3599198869839608,263,CR),
             ("cat_core","cat",CATP,1.86,250,CORE),
             ("xgb_v3","xgb",XGBP,1.70,100,CORE+V3_COLS)]
    oofs,tests={},{}
    for nm,fam,p,spw,rd,cols in members:
        m,se,_,o,t=evaluate(fam,p,spw,rd,Xtr[cols],y,g,fold_seeds=PRED,
                            model_seeds=MSEEDS,Xte=Xte[cols])
        oofs[nm],tests[nm]=o,t
        f1,auc=parts(y,o.mean(0))
        print(f"{nm:9s} combo={m:.4f} +/- {se:.4f}  F1={f1:.4f} AUC={auc:.4f}")

    W=1/3
    bo=sum(W*oofs[n] for n in oofs)
    per=[combo(y,bo[s]) for s in range(bo.shape[0])]
    f1,auc=parts(y,bo.mean(0))
    print(f"\nBLEND (equal 1/3) combo={np.mean(per):.4f} +/- "
          f"{np.std(per,ddof=1)/np.sqrt(len(per)):.4f}  F1={f1:.4f} AUC={auc:.4f}")
    print(f"  v7 trio on the same protocol: 0.8229    single best earlier: 0.8224")

    proba=sum(W*tests[n] for n in tests)
    full=np.zeros(len(Xte))
    for nm,fam,p,spw,rd,cols in members:
        acc=np.zeros(len(Xte))
        for s in MSEEDS:
            mdl=build(fam,p,spw,rd,s); mdl.fit(Xtr[cols],y)
            acc+=mdl.predict_proba(Xte[cols])[:,1]/len(MSEEDS)
        full+=W*acc
    print(f"bagged-vs-fulldata: corr={np.corrcoef(proba,full)[0,1]:.4f} "
          f"binary agreement={((proba>=.5)==(full>=.5)).mean():.4f}")

    sub=pd.DataFrame({"ID":te.ID,"TargetF1":(proba>=.5).astype(int),
                      "TargetRAUC":proba.astype(float)})
    assert list(sub.columns)==["ID","TargetF1","TargetRAUC"]
    assert sub.ID.equals(te.ID) and len(sub)==1030
    assert sub.TargetF1.isin([0,1]).all()
    assert (((sub.TargetRAUC>=.5).astype(int))==sub.TargetF1).all()
    assert not sub.isna().any().any() and sub.TargetRAUC.between(0,1).all()
    sub.to_csv("submissions/submission_v8.csv",index=False)
    print(f"\nsubmission_v8.csv  n={len(sub)}  pos-rate={sub.TargetF1.mean():.4f}  "
          f"prob=[{proba.min():.4f},{proba.max():.4f}]  ALL ASSERTIONS PASSED")
    for v in ["submissions/submission_v4.csv","submissions/submission_v7.csv"]:
        pv=pd.read_csv(v)
        print(f"  vs {v}: {(sub.TargetF1!=pv.TargetF1).sum():3d} binary rows differ, "
              f"corr={np.corrcoef(sub.TargetRAUC,pv.TargetRAUC)[0,1]:.4f}")
    np.savez_compressed("artifacts/v8.npz",test_probs=proba,full_probs=full,
                        **{f"oof_{k}":v for k,v in oofs.items()})

if __name__=="__main__":
    main()
