"""Blend search, honestly validated.

Members are decorrelated by feature set AND family. Weights are fit on selection
fold seeds (0-9) and then RE-MEASURED on held-out fold seeds (100-119). If a
weight vector only wins on the seeds it was fit on, it is noise-fitting -- the
failure mode that produced v5.
"""
import sys, warnings, itertools, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from harness import evaluate, combo, parts

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
MIN  = ["age","age_log1p","gender_bin","zone_bin","year"]

MEMBERS = {
 "xgb_core":   ("xgb", dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.), 1.86, 100, "core"),
 "xgb_v3":     ("xgb", dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.), 1.70, 100, "v3"),
 "xgb_min":    ("xgb", dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.), 1.86, 100, "min"),
 "xgb_d4":     ("xgb", dict(learning_rate=.05,max_depth=4,min_child_weight=5,subsample=.85,colsample_bytree=.9,reg_lambda=2.), 1.86,  80, "core"),
 "cat_core":   ("cat", dict(learning_rate=.05,depth=4,l2_leaf_reg=3.), 1.86, 250, "core"),
 "cat_v3":     ("cat", dict(learning_rate=.05,depth=4,l2_leaf_reg=3.), 1.86, 250, "v3"),
 "lgbm_core":  ("lgbm",dict(learning_rate=.05,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1.), 1.86, 120, "core"),
 "hgb_core":   ("hgb", dict(learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.), 1.86, 150, "core"),
}

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    Xf  = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    COLS = {"core": CORE, "min": MIN, "v3": CORE + V3_COLS}

    SEL, HOLD = range(10), range(100, 120)
    sel_oof, hold_oof = {}, {}
    print(f"{'member':11s} {'sel':>7s} {'hold':>7s} {'F1':>6s} {'AUC':>6s}")
    for name, (fam, p, spw, rounds, ck) in MEMBERS.items():
        X = Xf[COLS[ck]]
        m1,_,_,o1,_ = evaluate(fam, p, spw, rounds, X, y, groups, fold_seeds=SEL)
        m2,_,_,o2,_ = evaluate(fam, p, spw, rounds, X, y, groups, fold_seeds=HOLD)
        sel_oof[name], hold_oof[name] = o1, o2
        f1,auc = parts(y, o2.mean(0))
        print(f"{name:11s} {m1:7.4f} {m2:7.4f} {f1:6.4f} {auc:6.4f}")
    np.savez_compressed("artifacts/blend_oofs.npz", y=y.values,
        **{f"sel_{k}":v for k,v in sel_oof.items()}, **{f"hold_{k}":v for k,v in hold_oof.items()})

    keys = list(MEMBERS)
    def blend_score(w, store):
        # score each fold seed separately, then average -> matches harness semantics
        n_seeds = store[keys[0]].shape[0]
        return np.mean([combo(y, sum(w[i]*store[k][s] for i,k in enumerate(keys)))
                        for s in range(n_seeds)])

    print("\n--- candidate weightings ---")
    cands = {}
    cands["equal_all8"] = np.ones(len(keys))/len(keys)
    for subset in [("xgb_core","cat_core"), ("xgb_core","xgb_v3"), ("xgb_core","cat_core","xgb_v3"),
                   ("xgb_core","cat_core","xgb_v3","lgbm_core"),
                   ("xgb_core","cat_core","xgb_v3","xgb_min","xgb_d4"),
                   ("xgb_core","cat_core","xgb_v3","lgbm_core","xgb_min","xgb_d4")]:
        w = np.array([1.0 if k in subset else 0.0 for k in keys]); w/=w.sum()
        cands["eq_"+"+".join(s.replace("_","") for s in subset)] = w

    rng = np.random.default_rng(0); best=(-1,None)
    for _ in range(4000):
        w = rng.dirichlet(np.ones(len(keys)))
        s = blend_score(w, sel_oof)
        if s > best[0]: best=(s,w)
    cands["dirichlet_fit"] = best[1]

    rows=[]
    for name,w in cands.items():
        s_sel = blend_score(w, sel_oof); s_hold = blend_score(w, hold_oof)
        rows.append((name, s_sel, s_hold, s_hold-s_sel))
        print(f"{name:46s} sel={s_sel:.4f}  hold={s_hold:.4f}  drift={s_hold-s_sel:+.4f}")
    res=pd.DataFrame(rows,columns=["weighting","sel","hold","drift"]).sort_values("hold",ascending=False)
    res.to_csv("artifacts/blend.csv",index=False)
    print("\nbest single member (hold): %.4f"%max(
        np.mean([combo(y,hold_oof[k][s]) for s in range(hold_oof[k].shape[0])]) for k in keys))
    print("\nBEST BY HOLDOUT:"); print(res.head(4).to_string(index=False))
    print("\ndirichlet weights:", dict(zip(keys, best[1].round(3))))

if __name__ == "__main__":
    main()
