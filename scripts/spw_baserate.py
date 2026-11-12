"""
Round 19 — retune spw under the BASE-RATE-CORRECTED objective.

Round 17 established the evaluation should be done at the estimated test base rate
(0.691), not train's (0.651). But it only fixed the SCORING. Every spw in the
project was chosen against train's base rate, which under-rewards positives. A
higher test base rate should push the optimal spw UP.

Scored with the APHRC override applied and positives reweighted to p1, on fold
seeds 1400-1419 (untouched).
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from sklearn.metrics import f1_score, roc_auc_score
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME
from harness import evaluate
from optuna_v3 import aphrc_mask

CORE=["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
LGBP=dict(learning_rate=0.011374471419103494,num_leaves=53,min_child_samples=41,
  subsample=0.868088589184808,subsample_freq=1,colsample_bytree=0.8288246650597182,
  reg_lambda=0.001222299102744292,reg_alpha=2.8346683989414245)
XGBP=dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP=dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
S=range(1400,1420); P1=0.691

def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True); yv=y.values
    g=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    V4tr,_=build_v4(tr,te)
    X=pd.concat([add_v3(engineer(tr).reset_index(drop=True),tr.reset_index(drop=True)),V4tr],axis=1)
    mask,dlab=aphrc_mask(tr,te)
    p0=yv.mean(); w=np.where(yv==1,P1/p0,(1-P1)/(1-p0))
    def sc(o,thr=0.5):
        r=[]
        for i in range(o.shape[0]):
            p=o[i].copy(); hard=(p>=thr).astype(int)
            p2=p.copy(); p2[mask]=np.where(dlab[mask]==1,.999,.001); hard[mask]=dlab[mask]
            r.append(0.6*f1_score(yv,hard,sample_weight=w)+0.4*roc_auc_score(yv,p2,sample_weight=w))
        return float(np.mean(r)), float(np.std(r,ddof=1)/np.sqrt(len(r)))
    fams=[("lgbm","lgbm",LGBP,263,CORE+REGIME,1.36),
          ("cat","cat",CATP,250,CORE,1.86),
          ("xgb","xgb",XGBP,100,CORE+V3_COLS,1.70)]
    best={}; store={}
    print(f"objective: override + base rate {P1}, seeds {S.start}-{S.stop-1}\n")
    for nm,fam,p,rd,cols,cur in fams:
        print(f"{nm} (current spw {cur})")
        row=[]
        for spw in (1.2,1.5,1.86,2.2,2.6,3.0,3.6):
            _,_,_,o,_=evaluate(fam,p,spw,rd,X[cols],y,g,fold_seeds=S,model_seeds=(42,))
            m,se=sc(o); pr=np.mean([( (o[i]>=.5).astype(int)).mean() for i in range(o.shape[0])])
            row.append((spw,m,se,pr)); store[(nm,spw)]=o
            print(f"   spw {spw:4.2f}: {m:.4f} +/- {se:.4f}   oof pos-rate {pr:.4f}")
        b=max(row,key=lambda r:r[1]); best[nm]=b[0]
        print(f"   -> best spw {b[0]} ({b[1]:.4f})\n")
    old=(store[("lgbm",1.86)]*0)  # placeholder
    # blends
    print("blends:")
    trio_new=sum(store[(n,best[n])] for n in best)/3
    m,se=sc(trio_new); print(f"  retuned trio (spw {[best[n] for n in best]}): {m:.4f} +/- {se:.4f}")
    np.savez("artifacts/spw_baserate.npz", **{f"{n}_{s}":store[(n,s)] for n,s in store})
    print("\n(v8 baseline uses lgbm1.36/cat1.86/xgb1.70 -- rerun below for a like-for-like)")
    o8=[]
    for nm,fam,p,rd,cols,cur in fams:
        _,_,_,o,_=evaluate(fam,p,cur,rd,X[cols],y,g,fold_seeds=S,model_seeds=(42,)); o8.append(o)
    m8,se8=sc(sum(o8)/3); print(f"  v8 trio as shipped: {m8:.4f} +/- {se8:.4f}")
    print(f"  delta: {m-m8:+.4f}")
if __name__=="__main__": main()
