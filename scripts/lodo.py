"""Leave-one-DISTRICT-out CV: mimics the real test, where 30% of rows are in
districts never seen in training (Kalangala, Kamuli, Sembabule).

Village-grouped CV holds out villages inside the same region, so it never tests
out-of-region extrapolation. If full-feature models beat core models HERE but
lose under village-CV, that explains the v4-vs-v7 leaderboard inversion.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from harness import build, combo, parts
from sklearn.metrics import f1_score, roc_auc_score

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP = dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP = dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
LGBP = dict(learning_rate=.05,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1.)
SEEDS=(42,7,2026,99,5)

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    Xtr = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    FULL = list(engineer(tr).columns)
    COLS = {"core":CORE, "v3":CORE+V3_COLS, "full":FULL}
    dist = tr.location.str.split(",").str[-2].str.strip().reset_index(drop=True)
    print("districts:", dist.value_counts().to_dict())

    MEM = {"xgb_core":("xgb",XGBP,1.86,100,"core"), "cat_core":("cat",CATP,1.86,250,"core"),
           "xgb_v3":("xgb",XGBP,1.70,100,"v3"),     "xgb_full":("xgb",XGBP,1.86,100,"full"),
           "cat_full":("cat",CATP,1.86,250,"full"), "lgbm_full":("lgbm",LGBP,1.86,120,"full")}

    oof = {k: np.zeros(len(y)) for k in MEM}
    for d in dist.unique():
        va = (dist==d).values; trn = ~va
        for nm,(fam,p,spw,rd,ck) in MEM.items():
            X = Xtr[COLS[ck]]
            acc = np.zeros(va.sum())
            for s in SEEDS:
                m = build(fam,p,spw,rd,s); m.fit(X[trn], y[trn])
                acc += m.predict_proba(X[va])[:,1]/len(SEEDS)
            oof[nm][va] = acc

    print(f"\n{'member':10s} {'LODO combo':>11s} {'F1':>7s} {'AUC':>7s}   (village-CV combo for reference)")
    ref={"xgb_core":.8223,"cat_core":.8215,"xgb_v3":.8224,"xgb_full":.8178,"cat_full":.8140,"lgbm_full":.8131}
    for nm in MEM:
        f1,auc = parts(y, oof[nm]); c = .6*f1+.4*auc
        print(f"{nm:10s} {c:11.4f} {f1:7.4f} {auc:7.4f}   {ref[nm]:.4f}  ({c-ref[nm]:+.4f})")

    def bl(d):
        w=np.array([d.get(k,0.) for k in MEM]); w/=w.sum()
        return sum(w[i]*oof[k] for i,k in enumerate(MEM))
    print()
    for nm,d in [("CORE TRIO (v7)",{"xgb_core":1,"cat_core":1,"xgb_v3":1}),
                 ("core trio + xgb_full",{"xgb_core":1,"cat_core":1,"xgb_v3":1,"xgb_full":1}),
                 ("core 75% + full 25%",{"xgb_core":.25,"cat_core":.25,"xgb_v3":.25,"xgb_full":.0833,"cat_full":.0833,"lgbm_full":.0833}),
                 ("core 50% + full 50%",{"xgb_core":.167,"cat_core":.167,"xgb_v3":.167,"xgb_full":.167,"cat_full":.167,"lgbm_full":.167}),
                 ("FULL TRIO (v4-style)",{"xgb_full":1,"cat_full":1,"lgbm_full":1})]:
        p=bl(d); f1,auc=parts(y,p)
        print(f"{nm:24s} LODO combo={.6*f1+.4*auc:.4f}  F1={f1:.4f} AUC={auc:.4f}")

    print("\nper-district breakdown (core trio vs full trio):")
    ct=bl({"xgb_core":1,"cat_core":1,"xgb_v3":1}); ft=bl({"xgb_full":1,"cat_full":1,"lgbm_full":1})
    for d in dist.unique():
        m=(dist==d).values
        if y[m].nunique()<2: continue
        print(f"  {d:16s} n={m.sum():5d}  core AUC={roc_auc_score(y[m],ct[m]):.4f}  "
              f"full AUC={roc_auc_score(y[m],ft[m]):.4f}")
    np.savez_compressed("artifacts/lodo.npz", y=y.values, **{f"oof_{k}":v for k,v in oof.items()})

if __name__ == "__main__":
    main()
