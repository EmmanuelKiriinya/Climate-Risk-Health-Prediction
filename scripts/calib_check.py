"""Round 16b — is the full-data refit calibrated like the CV models?

v11's full-data refit gives a test pos-rate of 0.756 while every prior submission
sat at 0.81-0.85. Either the blend genuinely predicts fewer positives, or the
threshold-0.5 operating point does not survive the move from 4/5-data fold models
to a 5/5-data refit. F1 is measured at a FIXED 0.5, so a calibration shift between
CV and the shipped model is a silent, uncorrected loss.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from features import load_data, engineer
from features_v3 import add_v3
from features_v4 import build_v4, REGIME
from harness import evaluate, build
from diverse_members import oof_sklearn

CORE=["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
LGBP=dict(learning_rate=0.011374471419103494,num_leaves=53,min_child_samples=41,
  subsample=0.868088589184808,subsample_freq=1,colsample_bytree=0.8288246650597182,
  reg_lambda=0.001222299102744292,reg_alpha=2.8346683989414245)
def ET(s=42): return ExtraTreesClassifier(n_estimators=500,min_samples_leaf=12,max_features=.6,
                                          class_weight={0:1.,1:1.4},random_state=s,n_jobs=1)
def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True)
    g=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    V4tr,V4te=build_v4(tr,te)
    Xtr=pd.concat([add_v3(engineer(tr).reset_index(drop=True),tr.reset_index(drop=True)),V4tr],axis=1)
    Xte=pd.concat([add_v3(engineer(te).reset_index(drop=True),te.reset_index(drop=True)),V4te],axis=1)
    CR=CORE+REGIME; V=range(1100,1110)
    _,_,_,o_lgb,t_lgb=evaluate("lgbm",LGBP,1.3599198869839608,263,Xtr[CR],y,g,
                               fold_seeds=V,model_seeds=(42,),Xte=Xte[CR])
    o_et=oof_sklearn(ET,Xtr[CR],y,g,list(V))
    te_et=np.zeros(len(Xte)); cnt=0
    for fs in V:
        for tri,_ in StratifiedGroupKFold(5,shuffle=True,random_state=fs).split(Xtr,y,g):
            m=ET(); m.fit(Xtr[CR].iloc[tri],y.iloc[tri]); te_et+=m.predict_proba(Xte[CR])[:,1]; cnt+=1
    te_et/=cnt
    for nm,oof,tecv in (("lgbm",o_lgb.mean(0),t_lgb),("et",o_et.mean(0),te_et)):
        print(f"{nm:5s} OOF posrate@.5 {np.mean(oof>=.5):.4f} | CV-bagged TEST posrate {np.mean(tecv>=.5):.4f}"
              f" | mean prob OOF {oof.mean():.4f} test {tecv.mean():.4f}")
    b_oof=0.8*o_lgb.mean(0)+0.2*o_et.mean(0); b_te=0.8*t_lgb+0.2*te_et
    print(f"\nBLEND .8/.2  OOF posrate {np.mean(b_oof>=.5):.4f} | CV-bagged TEST posrate {np.mean(b_te>=.5):.4f}")
    pl=np.zeros(len(Xte)); pe=np.zeros(len(Xte))
    for s in (42,7,2026):
        m=build("lgbm",LGBP,1.3599198869839608,263,s); m.fit(Xtr[CR],y); pl+=m.predict_proba(Xte[CR])[:,1]/3
        m=ET(s); m.fit(Xtr[CR],y); pe+=m.predict_proba(Xte[CR])[:,1]/3
    full=0.8*pl+0.2*pe
    print(f"FULL-DATA refit TEST posrate {np.mean(full>=.5):.4f}  corr w/ CV-bagged {np.corrcoef(full,b_te)[0,1]:.4f}")
    print(f"  lgbm: CV-bagged {np.mean(t_lgb>=.5):.4f} vs full {np.mean(pl>=.5):.4f}")
    print(f"  et  : CV-bagged {np.mean(te_et>=.5):.4f} vs full {np.mean(pe>=.5):.4f}")
    np.savez("artifacts/calib_check.npz", t_lgb=t_lgb, te_et=te_et, pl=pl, pe=pe)
if __name__=="__main__": main()
