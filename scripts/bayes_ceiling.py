"""Bayes ceiling for this feature set.

If our calibrated probabilities p(x) ARE the true P(y|x), the remaining label
variation is irreducible coin-flipping. Simulating y* ~ Bernoulli(p) and scoring
p against y* gives the BEST score any model could ever achieve on these features.
That converts "is the leader's 0.864 reachable?" into an arithmetic question.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from harness import evaluate
from sklearn.metrics import roc_auc_score, f1_score

CORE=["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP=dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)

def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True)
    X=add_v3(engineer(tr).reset_index(drop=True),tr.reset_index(drop=True))
    g=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    _,_,_,o,_=evaluate("xgb",XGBP,1.0,100,X[CORE],y,g,fold_seeds=range(20))
    p=o.mean(0)
    print(f"calibrated OOF: mean p={p.mean():.4f}  actual rate={y.mean():.4f}\n")

    rng=np.random.default_rng(0); aucs=[];f1s=[];cbs=[]
    order=np.argsort(-p); ps=p[order]           # sort once, sweep every cutoff
    k=np.arange(1,len(p)+1)
    for _ in range(300):
        ys=(rng.random(len(p))<p).astype(int)
        if ys.sum() in (0,len(ys)): continue
        a=roc_auc_score(ys,p); aucs.append(a)
        tp=np.cumsum(ys[order]); P=ys.sum()
        best=float((2*tp/(k+P)).max())          # F1 at the best cutoff, vectorised
        f1s.append(best); cbs.append(.6*best+.4*a)
    A,F,C=np.mean(aucs),np.mean(f1s),np.mean(cbs)
    print("BAYES CEILING (best achievable if our p is the truth):")
    print(f"  AUC   = {A:.4f} +/- {np.std(aucs):.4f}")
    print(f"  F1*   = {F:.4f} +/- {np.std(f1s):.4f}  (optimal threshold, unconstrained)")
    print(f"  combo = {C:.4f} +/- {np.std(cbs):.4f}")
    print(f"\nOUR honest CV:  AUC 0.8286  F1 0.8190  combo 0.8230")
    print(f"gap to ceiling: AUC {A-0.8286:+.4f}   combo {C-0.8230:+.4f}")
    print(f"\nleader 0.864258 vs ceiling {C:.4f}  ->  "
          f"{'ABOVE the ceiling for these features' if 0.864258>C else 'below ceiling'}")
    print(f"  ({0.864258-C:+.4f} relative to what perfect knowledge of p(x) would score)")

if __name__=="__main__":
    main()
