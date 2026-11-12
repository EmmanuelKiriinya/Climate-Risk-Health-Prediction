"""Round 16c — clean head-to-head on a THIRD untouched seed set.

The 0.8/0.2 weighting was picked as the midpoint of a plateau observed on seeds
1100-1139, i.e. informed by the verification pass. Small leak, but this project has
been burned by exactly that before (v5). Seeds 1200-1239 have never been used by
anything, and settle v8 vs v11 on equal terms.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from sklearn.ensemble import ExtraTreesClassifier
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME
from harness import evaluate, combo, parts
from optuna_v3 import aphrc_mask
from diverse_members import oof_sklearn

CORE=["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
LGBP=dict(learning_rate=0.011374471419103494,num_leaves=53,min_child_samples=41,
  subsample=0.868088589184808,subsample_freq=1,colsample_bytree=0.8288246650597182,
  reg_lambda=0.001222299102744292,reg_alpha=2.8346683989414245)
XGBP=dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP=dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
FINAL=range(1200,1240)
def ET(s=42): return ExtraTreesClassifier(n_estimators=500,min_samples_leaf=12,max_features=.6,
                                          class_weight={0:1.,1:1.4},random_state=s,n_jobs=1)
def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True)
    g=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    V4tr,_=build_v4(tr,te)
    X=pd.concat([add_v3(engineer(tr).reset_index(drop=True),tr.reset_index(drop=True)),V4tr],axis=1)
    CR=CORE+REGIME
    mask,dlab=aphrc_mask(tr,te)
    def adj(o):
        s=[]
        for i in range(o.shape[0]):
            p=o[i].copy(); p[mask]=np.where(dlab[mask]==1,.999,.001); s.append(combo(y,p))
        return float(np.mean(s)), float(np.std(s,ddof=1)/np.sqrt(len(s))), np.array(s)
    o={}
    for nm,fam,p,spw,rd,cols in [("lgbm","lgbm",LGBP,1.3599198869839608,263,CR),
                                 ("cat","cat",CATP,1.86,250,CORE),
                                 ("xgb","xgb",XGBP,1.70,100,CORE+V3_COLS)]:
        _,_,_,oo,_=evaluate(fam,p,spw,rd,X[cols],y,g,fold_seeds=FINAL,model_seeds=(42,)); o[nm]=oo
    o["et"]=oof_sklearn(ET,X[CR],y,g,list(FINAL))
    v8=(o["lgbm"]+o["cat"]+o["xgb"])/3
    v11=0.8*o["lgbm"]+0.2*o["et"]
    m8,s8,a8=adj(v8); m11,s11,a11=adj(v11)
    d=a11-a8; se=d.std(ddof=1)/np.sqrt(len(d))
    print(f"seeds {FINAL.start}-{FINAL.stop-1}, never used before")
    print(f"  v8  equal thirds : {m8:.4f} +/- {s8:.4f}")
    print(f"  v11 0.8/0.2      : {m11:.4f} +/- {s11:.4f}")
    print(f"  paired delta     : {d.mean():+.4f} +/- {se:.4f}   t={d.mean()/se:+.2f}   "
          f"v11 wins {int((d>0).sum())}/{len(d)} seeds")
    f1,auc=parts(y,v11.mean(0)); print(f"  v11 F1={f1:.4f} AUC={auc:.4f}")
    f1,auc=parts(y,v8.mean(0));  print(f"  v8  F1={f1:.4f} AUC={auc:.4f}")
if __name__=="__main__": main()
