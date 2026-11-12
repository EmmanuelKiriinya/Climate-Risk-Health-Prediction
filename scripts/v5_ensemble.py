"""v5 ensemble: add decorrelated members, stability-aware weight search.

Members evaluated under 3 fold seeds; weights maximize mean combo.
Test probs: average of fold-model predictions AND full-data seed-bagged models.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from features import load_data
from features_v2 import build_v2

SEEDS = [42, 7, 2026]
FOLD_SEEDS = [42, 7, 2026]
SPW = 1.86

CORE_COLS = ["age", "age_log1p", "is_infant", "is_under5", "is_child", "is_elderly",
             "gender_bin", "zone_bin", "year"]


def make(name, seed):
    if name == "core_xgb":
        return xgb.XGBClassifier(n_estimators=1500, learning_rate=0.06, max_depth=3,
                                 scale_pos_weight=1.86, subsample=0.85, colsample_bytree=0.9,
                                 reg_lambda=1.0, random_state=seed, eval_metric="logloss",
                                 early_stopping_rounds=100, verbosity=0), "es"
    if name == "xgb_spw15":
        return dict(cls="xgb"), None
    return None


def ctor(name, seed):
    if name.startswith("core"):
        m = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.06, max_depth=3,
                              scale_pos_weight=1.86, subsample=0.85, colsample_bytree=0.9,
                              reg_lambda=1.0, random_state=seed, eval_metric="logloss",
                              early_stopping_rounds=100, verbosity=0)
        return m, "xgb"
    spec = {
        "xgb_d3_lr05": ("xgb", dict(learning_rate=0.05, max_depth=3, min_child_weight=1)),
        "xgb_d3_lr03": ("xgb", dict(learning_rate=0.03, max_depth=3, min_child_weight=1)),
        "lgbm_l15": ("lgbm", dict(learning_rate=0.03, num_leaves=15, min_child_samples=40)),
        "cat_d4": ("cat", dict(learning_rate=0.03, depth=4)),
        "xgb_spw15": ("xgb", dict(learning_rate=0.05, max_depth=3, min_child_weight=1)),
        "xgb_spw24": ("xgb", dict(learning_rate=0.04, max_depth=3, min_child_weight=2)),
        "cat_d5": ("cat", dict(learning_rate=0.05, depth=5)),
        "lgbm_l31": ("lgbm", dict(learning_rate=0.04, num_leaves=31, min_child_samples=20)),
    }
    fam, p = spec[name]
    spw = 1.5 if name == "xgb_spw15" else 2.4 if name == "xgb_spw24" else SPW
    if name == "core_xgb":
        spw = SPW
    if fam == "xgb":
        return xgb.XGBClassifier(n_estimators=3000, scale_pos_weight=spw, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                                 eval_metric="logloss", early_stopping_rounds=100,
                                 verbosity=0, **p), fam
    if fam == "lgbm":
        return lgb.LGBMClassifier(n_estimators=3000, scale_pos_weight=spw, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=seed, verbose=-1, **p), fam
    return CatBoostClassifier(iterations=3000, scale_pos_weight=spw, l2_leaf_reg=3.0,
                              random_seed=seed, verbose=False, allow_writing_files=False,
                              early_stopping_rounds=100, **p), fam


def fit(m, fam, Xtr, ytr, Xva, yva):
    if fam == "lgbm":
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    elif fam == "cat":
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
    else:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)


def main():
    tr_raw, te_raw = load_data()
    y = tr_raw.is_climate_sensitive.reset_index(drop=True)
    Xall, Xte_all, _ = build_v2(tr_raw, te_raw)
    cols = list(Xall.columns[:43]) + [c for c in Xall.columns if c.startswith("season_")]
    Xf, Xfe = Xall[cols].reset_index(drop=True), Xte_all[cols].reset_index(drop=True)
    P_tr = pd.read_csv("artifacts/power_features_train.csv")[["rh_mean_30d_anom", "qv_mean_30d_anom"]]
    P_te = pd.read_csv("artifacts/power_features_test.csv")[["rh_mean_30d_anom", "qv_mean_30d_anom"]]
    Xn = pd.concat([Xf, P_tr], axis=1); Xne = pd.concat([Xfe, P_te], axis=1)

    village = pd.factorize(tr_raw.location.str.split(",").str[:2].str.join(","))[0]

    members = ["core_xgb", "xgb_d3_lr05", "xgb_d3_lr03", "lgbm_l15", "cat_d4",
               "xgb_spw15", "xgb_spw24", "cat_d5", "lgbm_l31"]
    featsel = {"core_xgb": CORE_COLS}
    which = {n: ("n" if n in ("n_xgb",) else "") for n in []}
    data = {}
    for n in members:
        if n == "core_xgb" or n not in ("xgb_d3_lr05", "xgb_d3_lr03", "lgbm_l15", "cat_d4"):
            data[n] = (Xn if n != "core_xgb" else Xn[featsel["core_xgb"]],
                       Xne if n != "core_xgb" else Xne[featsel["core_xgb"]])
        else:
            # half of classic members on plain base+season for decorrelation
            data[n] = (Xf, Xfe)

    oofs = {n: np.zeros((len(FOLD_SEEDS), len(y))) for n in members}
    tests = {n: np.zeros(len(Xne)) for n in members}

    for name in members:
        Xd, Xde = data[name]
        for fi, fs in enumerate(FOLD_SEEDS):
            splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(Xd, y, village))
            for si, seed in enumerate(SEEDS):
                te_acc = np.zeros(len(Xde))
                for tr_i, va_i in splits:
                    m, fam = ctor(name, seed * 10 + fi)
                    fit(m, fam, Xd.iloc[tr_i], y.iloc[tr_i], Xd.iloc[va_i], y.iloc[va_i])
                    oofs[name][fi][va_i] += m.predict_proba(Xd.iloc[va_i])[:, 1] / len(SEEDS)
                    te_acc += m.predict_proba(Xde)[:, 1] / len(splits)
                tests[name] += te_acc / len(FOLD_SEEDS)
        f1 = f1_score(y, (oofs[name].mean(0) >= .5).astype(int))
        auc = roc_auc_score(y, oofs[name].mean(0))
        print(f"{name}: combo={0.6*f1+.4*auc:.4f} F1={f1:.4f} AUC={auc:.4f}")

    np.savez_compressed("artifacts/v5_members.npz",
                        y=y.values, tests=np.vstack([tests[n] for n in members]),
                        keys=np.array(members),
                        **{f"oof_{k}_{fi}": oofs[k][fi] for k in members for fi in range(len(FOLD_SEEDS))})

    # stability-aware weight search: maximize MEAN combo across fold seeds
    O = np.vstack([oofs[n] for n in members])  # (members*seeds, n)

    def score(w):
        tot = 0.0
        for fi in range(len(FOLD_SEEDS)):
            rows = slice(fi * len(members), (fi + 1) * len(members))
            p = w @ O[rows]
            tot += 0.6 * f1_score(y, (p >= .5).astype(int)) + 0.4 * roc_auc_score(y, p)
        return tot / len(FOLD_SEEDS)

    rng = np.random.default_rng(42)
    best = (-1, None)
    for _ in range(30000):
        w = rng.dirichlet(np.ones(len(members)) * 2)
        s = score(w)
        if s > best[0]:
            best = (s, w)
    print("\nstability-aware best mean-combo=%.5f" % best[0])
    print(dict(zip(members, best[1].round(3))))
    tp = best[1] @ np.vstack([tests[n] for n in members])
    np.savez_compressed("artifacts/v5_blend.npz", weights=best[1], keys=np.array(members),
                        test_probs=tp, mean_combo=best[0])


if __name__ == "__main__":
    main()
