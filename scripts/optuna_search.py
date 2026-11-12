"""Optuna TPE search under the HONEST harness.

Why this is not a repeat of tune.py / tune2.py: those searched with early stopping
on the validation fold and a single fold seed, so their rankings reflected leakage
plus ~0.0013 of seed noise. Here every trial is scored with fixed rounds and the
mean over 6 fold seeds (SE ~0.0004), and -- critically -- the top trials are
RE-SCORED on held-out fold seeds 100-119.

That last step matters more than the search. With ~250 trials on an objective with
SE 0.0004, the winning trial's score is inflated by roughly 2-3 SE (~0.001) purely
by max-selection. Without a holdout pass Optuna will confidently report a gain that
does not exist -- exactly the mechanism that produced v5.
"""
import sys, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME, CTIME
from harness import evaluate, parts

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
MIN  = ["age","age_log1p","gender_bin","zone_bin","year"]
SEL, HOLD = range(6), range(100, 120)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 250

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True)
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    X0 = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    V4tr, _ = build_v4(tr, te)
    X = pd.concat([X0, V4tr], axis=1)
    FSETS = {"core": CORE, "min": MIN, "v3": CORE+V3_COLS,
             "core_regime": CORE+REGIME, "core_ctime_regime": CORE+CTIME+REGIME,
             "full": list(engineer(tr).columns)}

    def spec(t):
        fam = t.suggest_categorical("fam", ["xgb", "lgbm", "cat"])
        fs  = t.suggest_categorical("fset", list(FSETS))
        spw = t.suggest_float("spw", 1.2, 2.6)
        if fam == "xgb":
            p = dict(learning_rate=t.suggest_float("lr",.01,.30,log=True),
                     max_depth=t.suggest_int("depth",2,6),
                     min_child_weight=t.suggest_int("mcw",1,20),
                     subsample=t.suggest_float("sub",.5,1.0),
                     colsample_bytree=t.suggest_float("col",.4,1.0),
                     reg_lambda=t.suggest_float("l2",1e-3,20,log=True),
                     reg_alpha=t.suggest_float("l1",1e-4,10,log=True),
                     gamma=t.suggest_float("gamma",1e-4,5,log=True))
            rd = t.suggest_int("rounds",20,600,log=True)
        elif fam == "lgbm":
            p = dict(learning_rate=t.suggest_float("lr",.01,.30,log=True),
                     num_leaves=t.suggest_int("leaves",3,63),
                     min_child_samples=t.suggest_int("mcs",5,120),
                     subsample=t.suggest_float("sub",.5,1.0), subsample_freq=1,
                     colsample_bytree=t.suggest_float("col",.4,1.0),
                     reg_lambda=t.suggest_float("l2",1e-3,20,log=True),
                     reg_alpha=t.suggest_float("l1",1e-4,10,log=True))
            rd = t.suggest_int("rounds",20,600,log=True)
        else:
            p = dict(learning_rate=t.suggest_float("lr",.01,.30,log=True),
                     depth=t.suggest_int("depth",2,8),
                     l2_leaf_reg=t.suggest_float("l2",.5,30,log=True))
            rd = t.suggest_int("rounds",50,900,log=True)
        return fam, p, spw, rd, FSETS[fs]

    def objective(t):
        fam, p, spw, rd, cols = spec(t)
        m,_,_,_,_ = evaluate(fam, p, spw, rd, X[cols], y, groups, fold_seeds=SEL, n_jobs=6)
        return m

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=40))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"completed {len(study.trials)} trials")
    print(f"best SEARCH score {study.best_value:.4f}  params={study.best_params}\n")

    df = study.trials_dataframe().sort_values("value", ascending=False)
    df.to_csv("artifacts/optuna_trials.csv", index=False)

    print("=== HOLDOUT RE-SCORE of top-10 trials (fold seeds 100-119) ===")
    print(f"{'rank':>4s} {'search':>7s} {'holdout':>8s} {'drift':>8s}  fam  fset")
    rows=[]
    for i, tr_ in enumerate(sorted(study.trials, key=lambda x: -(x.value or -1))[:10]):
        fam, p, spw, rd, cols = spec(optuna.trial.FixedTrial(tr_.params))
        m2,se2,_,o2,_ = evaluate(fam, p, spw, rd, X[cols], y, groups, fold_seeds=HOLD, n_jobs=10)
        f1,auc = parts(y, o2.mean(0))
        rows.append((i+1, tr_.value, m2, m2-tr_.value, tr_.params["fam"], tr_.params["fset"], f1, auc,
                     json.dumps(tr_.params)))
        print(f"{i+1:4d} {tr_.value:7.4f} {m2:8.4f} {m2-tr_.value:+8.4f}  "
              f"{tr_.params['fam']:4s} {tr_.params['fset']}")
    pd.DataFrame(rows, columns=["rank","search","holdout","drift","fam","fset","f1","auc","params"]
                 ).to_csv("artifacts/optuna_holdout.csv", index=False)
    best_hold = max(r[2] for r in rows)
    print(f"\nbest holdout from Optuna : {best_hold:.4f}")
    print(f"incumbent xgb_core holdout: 0.8223")
    print(f"v7 blend holdout          : 0.8229     Bayes ceiling: 0.8237")
    try:
        imp = optuna.importance.get_param_importances(study)
        print("\nhyperparameter importances:")
        for k,v in list(imp.items())[:10]: print(f"  {k:10s} {v:.3f}")
    except Exception as e:
        print("importance unavailable:", e)

if __name__ == "__main__":
    main()
