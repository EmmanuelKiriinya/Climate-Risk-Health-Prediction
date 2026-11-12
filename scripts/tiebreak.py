"""Tie-break lgbm_reg alone vs the v8 blend on 40 fresh fold seeds, and recompute
the Bayes ceiling with the better model (the old 0.8237 was conditional on the
weaker p(x), so it was an underestimate)."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0,"scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME
from harness import evaluate, combo, parts
from sklearn.metrics import roc_auc_score

CORE=["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP=dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP=dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
LGBP=dict(learning_rate=0.011374471419103494,num_leaves=53,min_child_samples=41,
          subsample=0.868088589184808,subsample_freq=1,colsample_bytree=0.8288246650597182,
          reg_lambda=0.001222299102744292,reg_alpha=2.8346683989414245)
FS=range(700,740)

def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True)
    g=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    X0=add_v3(engineer(tr).reset_index(drop=True),tr.reset_index(drop=True))
    V4,_=build_v4(tr,te); X=pd.concat([X0,V4],axis=1); CR=CORE+REGIME

    print("=== tie-break on 40 fresh fold seeds (700-739) ===")
    O={}
    for nm,fam,p,spw,rd,cols in [("lgbm_reg","lgbm",LGBP,1.3599198869839608,263,CR),
                                 ("cat_core","cat",CATP,1.86,250,CORE),
                                 ("xgb_v3","xgb",XGBP,1.70,100,CORE+V3_COLS)]:
        m,se,_,o,_=evaluate(fam,p,spw,rd,X[cols],y,g,fold_seeds=FS)
        O[nm]=o; print(f"  {nm:9s} {m:.4f} +/- {se:.4f}")
    bl=sum(O[k]/3 for k in O)
    per=[combo(y,bl[s]) for s in range(bl.shape[0])]
    solo=[combo(y,O["lgbm_reg"][s]) for s in range(bl.shape[0])]
    d=np.array(per)-np.array(solo)
    print(f"  blend      {np.mean(per):.4f} +/- {np.std(per,ddof=1)/np.sqrt(len(per)):.4f}")
    print(f"  difference (blend - solo) = {d.mean():+.4f} +/- "
          f"{d.std(ddof=1)/np.sqrt(len(d)):.4f}  (paired, same folds)")
    print(f"  -> {'BLEND wins' if d.mean()>2*d.std(ddof=1)/np.sqrt(len(d)) else 'SOLO wins' if -d.mean()>2*d.std(ddof=1)/np.sqrt(len(d)) else 'statistical TIE'}")

    print("\n=== recomputed Bayes ceiling with the better model ===")
    _,_,_,oc,_=evaluate("lgbm",LGBP,1.0,263,X[CR],y,g,fold_seeds=range(20))
    p=oc.mean(0)
    print(f"  calibrated mean p={p.mean():.4f} vs actual {y.mean():.4f}")
    rng=np.random.default_rng(0); order=np.argsort(-p); k=np.arange(1,len(p)+1)
    A=[];F=[];C=[]
    for _ in range(300):
        ys=(rng.random(len(p))<p).astype(int)
        if ys.sum() in (0,len(ys)): continue
        a=roc_auc_score(ys,p); tp=np.cumsum(ys[order]); P=ys.sum()
        f=float((2*tp/(k+P)).max()); A.append(a);F.append(f);C.append(.6*f+.4*a)
    print(f"  AUC {np.mean(A):.4f}  F1* {np.mean(F):.4f}  combo {np.mean(C):.4f} +/- {np.std(C):.4f}")
    print(f"  (old ceiling from the weaker model: 0.8237)")
    print(f"  our best measured: 0.8249  -> gap to new ceiling {np.mean(C)-0.8249:+.4f}")
    print(f"  leader 0.864258 vs new ceiling: {0.864258-np.mean(C):+.4f}")

if __name__=="__main__":
    main()
