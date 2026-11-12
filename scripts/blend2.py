"""Does adding the v4-style FULL-feature family to the core blend help?

full43 alone is 0.8175 (much worse than core's 0.8223) but its test predictions
correlate only 0.966 with the core family -- decorrelation can still pay on a
non-linear metric. Tested honestly: weights fixed a priori, sel vs holdout seeds.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from harness import evaluate, combo, parts

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP = dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP = dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
LGBP = dict(learning_rate=.05,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1.)

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    Xtr = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    FULL = list(engineer(tr).columns)
    COLS = {"core":CORE, "v3":CORE+V3_COLS, "full":FULL}

    MEM = {"xgb_core":("xgb",XGBP,1.86,100,"core"), "cat_core":("cat",CATP,1.86,250,"core"),
           "xgb_v3":("xgb",XGBP,1.70,100,"v3"),     "xgb_full":("xgb",XGBP,1.86,100,"full"),
           "cat_full":("cat",CATP,1.86,250,"full"), "lgbm_full":("lgbm",LGBP,1.86,120,"full")}
    SEL, HOLD = range(10), range(100,120)
    so, ho = {}, {}
    print(f"{'member':10s} {'sel':>7s} {'hold':>7s} {'F1':>6s} {'AUC':>6s}")
    for nm,(fam,p,spw,rd,ck) in MEM.items():
        m1,_,_,o1,_ = evaluate(fam,p,spw,rd,Xtr[COLS[ck]],y,groups,fold_seeds=SEL)
        m2,_,_,o2,_ = evaluate(fam,p,spw,rd,Xtr[COLS[ck]],y,groups,fold_seeds=HOLD)
        so[nm],ho[nm]=o1,o2; f1,auc=parts(y,o2.mean(0))
        print(f"{nm:10s} {m1:7.4f} {m2:7.4f} {f1:6.4f} {auc:6.4f}")

    keys=list(MEM)
    def bs(w,store):
        ns=store[keys[0]].shape[0]
        return np.mean([combo(y,sum(w[i]*store[k][s] for i,k in enumerate(keys))) for s in range(ns)])
    def wv(d):
        w=np.array([d.get(k,0.) for k in keys]); return w/w.sum()

    cands={
      "CORE TRIO (v7, current)": wv({"xgb_core":1,"cat_core":1,"xgb_v3":1}),
      "core trio + xgb_full (3:1)": wv({"xgb_core":1,"cat_core":1,"xgb_v3":1,"xgb_full":1}),
      "core trio 75% + full trio 25%": wv({"xgb_core":.25,"cat_core":.25,"xgb_v3":.25,
                                           "xgb_full":.0833,"cat_full":.0833,"lgbm_full":.0833}),
      "core trio 67% + full trio 33%": wv({"xgb_core":.222,"cat_core":.222,"xgb_v3":.222,
                                           "xgb_full":.111,"cat_full":.111,"lgbm_full":.111}),
      "core 50% + full 50%": wv({"xgb_core":.167,"cat_core":.167,"xgb_v3":.167,
                                 "xgb_full":.167,"cat_full":.167,"lgbm_full":.167}),
      "FULL TRIO only (v4-style)": wv({"xgb_full":1,"cat_full":1,"lgbm_full":1}),
    }
    print(f"\n{'weighting':34s} {'sel':>7s} {'hold':>7s} {'drift':>8s}")
    rows=[]
    for nm,w in cands.items():
        a,b=bs(w,so),bs(w,ho); rows.append((nm,a,b,b-a))
        print(f"{nm:34s} {a:7.4f} {b:7.4f} {b-a:+8.4f}")
    pd.DataFrame(rows,columns=["weighting","sel","hold","drift"]).to_csv("artifacts/blend2.csv",index=False)
    np.savez_compressed("artifacts/blend2_oofs.npz", y=y.values,
        **{f"sel_{k}":v for k,v in so.items()}, **{f"hold_{k}":v for k,v in ho.items()})

if __name__ == "__main__":
    main()
