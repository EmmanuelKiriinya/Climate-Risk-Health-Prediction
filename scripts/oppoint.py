"""
Round 17 — v11 lost 0.0046 on the LB while CV said +0.0009 (t=3.67). Why?

v11's pos-rate is 0.756; v9's is 0.817; the best pre-override submissions all sat at
0.83-0.85. Hypothesis: v11's MODEL may be fine and its OPERATING POINT wrong -- F1 is
scored at a hard 0.5, so a model that is better-ranked but shifted in calibration can
lose despite better AUC.

The decisive test is to compare v8 and v11 at MATCHED predicted-positive rate. That
separates "worse model" from "wrong threshold".
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME
from harness import evaluate
from optuna_v3 import aphrc_mask
from diverse_members import oof_sklearn

CORE=["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
LGBP=dict(learning_rate=0.011374471419103494,num_leaves=53,min_child_samples=41,
  subsample=0.868088589184808,subsample_freq=1,colsample_bytree=0.8288246650597182,
  reg_lambda=0.001222299102744292,reg_alpha=2.8346683989414245)
XGBP=dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP=dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
S=range(1300,1330)
def ET(s=42): return ExtraTreesClassifier(n_estimators=500,min_samples_leaf=12,max_features=.6,
                                          class_weight={0:1.,1:1.4},random_state=s,n_jobs=1)
def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True); yv=y.values
    g=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    V4tr,_=build_v4(tr,te)
    X=pd.concat([add_v3(engineer(tr).reset_index(drop=True),tr.reset_index(drop=True)),V4tr],axis=1)
    CR=CORE+REGIME; mask,dlab=aphrc_mask(tr,te)
    o={}
    for nm,fam,p,spw,rd,cols in [("lgbm","lgbm",LGBP,1.3599198869839608,263,CR),
                                 ("cat","cat",CATP,1.86,250,CORE),
                                 ("xgb","xgb",XGBP,1.70,100,CORE+V3_COLS)]:
        _,_,_,oo,_=evaluate(fam,p,spw,rd,X[cols],y,g,fold_seeds=S,model_seeds=(42,)); o[nm]=oo
    o["et"]=oof_sklearn(ET,X[CR],y,g,list(S))
    blends={"v8_thirds":(o["lgbm"]+o["cat"]+o["xgb"])/3, "v11":0.8*o["lgbm"]+0.2*o["et"]}

    def adj_at(b, target_rate=None, thr=None):
        """combo with override, thresholding either at `thr` or at the quantile giving target_rate."""
        sc=[]; rates=[]
        for i in range(b.shape[0]):
            p=b[i].copy()
            t = thr if thr is not None else np.quantile(p, 1-target_rate)
            hard=(p>=t).astype(int)
            p2=p.copy(); p2[mask]=np.where(dlab[mask]==1,.999,.001)
            hard[mask]=dlab[mask]
            sc.append(0.6*f1_score(yv,hard)+0.4*roc_auc_score(yv,p2)); rates.append(hard.mean())
        return float(np.mean(sc)), float(np.mean(rates))

    print(f"seeds {S.start}-{S.stop-1}\n")
    print("--- each blend at its own threshold 0.5 (what was shipped) ---")
    for nm,b in blends.items():
        m,r=adj_at(b, thr=0.5); print(f"  {nm:10s} combo {m:.4f}  pos-rate {r:.4f}")
    print("\n--- threshold swept, per blend: where does CV say the optimum is? ---")
    for nm,b in blends.items():
        best=None
        for t in np.arange(.30,.71,.02):
            m,r=adj_at(b,thr=t)
            if best is None or m>best[0]: best=(m,t,r)
        print(f"  {nm:10s} best combo {best[0]:.4f} at thr {best[1]:.2f} (pos-rate {best[2]:.4f})")
    print("\n--- MATCHED pos-rate: is v11 a better model, or just mis-thresholded? ---")
    print(f"  {'rate':>6s}  {'v8_thirds':>10s}  {'v11':>10s}  {'delta':>8s}")
    for rate in (0.74,0.76,0.78,0.80,0.82,0.84,0.86):
        m8,_=adj_at(blends["v8_thirds"], target_rate=rate)
        m11,_=adj_at(blends["v11"], target_rate=rate)
        print(f"  {rate:6.2f}  {m8:10.4f}  {m11:10.4f}  {m11-m8:+8.4f}")
if __name__=="__main__": main()
