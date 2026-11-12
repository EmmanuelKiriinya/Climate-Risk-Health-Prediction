"""Per-family Optuna, then blend the winners.

The 250-trial shared study was won by LightGBM, which means CatBoost and XGB were
barely sampled ON THE REGIME FEATURE SET -- the block that turned out to matter.
Giving each family its own budget removes that starvation, and produces three
independently-tuned members to blend rather than one tuned member plus two legacy ones.

Discipline unchanged: search on seeds 0-5, re-score top-5 per family on 100-119,
blend and verify on 300-319 (all previously used only for other purposes -- the final
build will use fresh seeds again).
"""
import sys, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME, CTIME, DENSITY
from harness import evaluate, combo, parts

CORE=["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
MIN=["age","age_log1p","gender_bin","zone_bin","year"]
SEL=range(6); HOLD=range(100,120); VER=range(300,320)
N=int(sys.argv[1]) if len(sys.argv)>1 else 180

def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True)
    g=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    X0=add_v3(engineer(tr).reset_index(drop=True),tr.reset_index(drop=True))
    V4,_=build_v4(tr,te); X=pd.concat([X0,V4],axis=1)
    FS={"core":CORE,"min":MIN,"core_v3":CORE+V3_COLS,"core_regime":CORE+REGIME,
        "core_regime_v3":CORE+REGIME+V3_COLS,"core_regime_ctime":CORE+REGIME+CTIME,
        "min_regime":MIN+REGIME,"core_regime_density":CORE+REGIME+DENSITY,
        "full":list(engineer(tr).columns)}

    def spec(fam,t):
        fs=t.suggest_categorical("fset",list(FS)); spw=t.suggest_float("spw",1.0,2.6)
        if fam=="xgb":
            p=dict(learning_rate=t.suggest_float("lr",.005,.30,log=True),
                   max_depth=t.suggest_int("depth",2,8),
                   min_child_weight=t.suggest_int("mcw",1,40),
                   subsample=t.suggest_float("sub",.5,1.),
                   colsample_bytree=t.suggest_float("col",.3,1.),
                   reg_lambda=t.suggest_float("l2",1e-3,50,log=True),
                   reg_alpha=t.suggest_float("l1",1e-4,20,log=True),
                   gamma=t.suggest_float("gamma",1e-4,8,log=True))
            rd=t.suggest_int("rounds",20,900,log=True)
        elif fam=="lgbm":
            p=dict(learning_rate=t.suggest_float("lr",.005,.30,log=True),
                   num_leaves=t.suggest_int("leaves",3,127),
                   min_child_samples=t.suggest_int("mcs",3,150),
                   subsample=t.suggest_float("sub",.5,1.),subsample_freq=1,
                   colsample_bytree=t.suggest_float("col",.3,1.),
                   reg_lambda=t.suggest_float("l2",1e-3,50,log=True),
                   reg_alpha=t.suggest_float("l1",1e-4,20,log=True))
            rd=t.suggest_int("rounds",20,900,log=True)
        else:
            p=dict(learning_rate=t.suggest_float("lr",.005,.30,log=True),
                   depth=t.suggest_int("depth",2,9),
                   l2_leaf_reg=t.suggest_float("l2",.3,60,log=True))
            rd=t.suggest_int("rounds",40,1200,log=True)
        return p,spw,rd,FS[fs],fs

    best={}
    for fam in ("lgbm","xgb","cat"):
        def obj(t,fam=fam):
            p,spw,rd,cols,_=spec(fam,t)
            m,_,_,_,_=evaluate(fam,p,spw,rd,X[cols],y,g,fold_seeds=SEL,n_jobs=6)
            return m
        st=optuna.create_study(direction="maximize",
              sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=35))
        st.optimize(obj,n_trials=N,show_progress_bar=False)
        print(f"\n### {fam}: best search {st.best_value:.4f}  fset={st.best_params['fset']} "
              f"spw={st.best_params['spw']:.3f}")
        top=sorted(st.trials,key=lambda x:-(x.value or -1))[:5]
        rows=[]
        for i,t_ in enumerate(top):
            p,spw,rd,cols,fs=spec(fam,optuna.trial.FixedTrial(t_.params))
            m2,_,_,_,_=evaluate(fam,p,spw,rd,X[cols],y,g,fold_seeds=HOLD,n_jobs=10)
            rows.append((m2,t_.value,t_.params,p,spw,rd,cols,fs))
            print(f"   #{i+1} search={t_.value:.4f} holdout={m2:.4f} drift={m2-t_.value:+.4f} "
                  f"fset={fs} spw={spw:.2f}")
        rows.sort(key=lambda r:-r[0])
        best[fam]=rows[0]
        print(f"   -> {fam} best-by-holdout {rows[0][0]:.4f} (fset={rows[0][7]}, spw={rows[0][4]:.3f})")

    print("\n=== blend the three per-family winners (verify seeds 300-319) ===")
    O={}
    for fam,(h,sv,params,p,spw,rd,cols,fs) in best.items():
        m,se,_,o,_=evaluate(fam,p,spw,rd,X[cols],y,g,fold_seeds=VER)
        O[fam]=o; print(f"  {fam:5s} {m:.4f} +/- {se:.4f}  fset={fs} spw={spw:.3f} rounds={rd}")
    keys=list(O)
    def bs(w):
        return float(np.mean([combo(y,sum(w[i]*O[k][s] for i,k in enumerate(keys)))
                              for s in range(O[keys[0]].shape[0])]))
    import itertools
    print()
    for nm,w in [("equal thirds",[1/3]*3),
                 ("lgbm-heavy 2:1:1",[.5,.25,.25] if keys[0]=="lgbm" else None),
                 ("lgbm only",[1 if k=="lgbm" else 0 for k in keys]),
                 ("lgbm+xgb",[.5 if k in("lgbm","xgb") else 0 for k in keys]),
                 ("lgbm+cat",[.5 if k in("lgbm","cat") else 0 for k in keys])]:
        if w is None: continue
        print(f"  {nm:20s} {bs(np.array(w,float)):.4f}")
    print(f"\n  reference: v8 blend 0.8245 | v7 trio 0.8229")
    json.dump({f:{"params":best[f][2],"spw":best[f][4],"rounds":best[f][5],"fset":best[f][7],
                  "holdout":best[f][0]} for f in best},
              open("artifacts/optuna2_best.json","w"),indent=2,default=str)

if __name__=="__main__":
    main()
