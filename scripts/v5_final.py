"""v5 final: core-feature family ensemble (xgb/lgbm/cat), stability-aware weights."""
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

SEEDS = [42, 7, 2026, 99, 5]
FOLDS = [42, 7, 2026]
CORE = ["age", "age_log1p", "is_infant", "is_under5", "is_child", "is_elderly",
        "gender_bin", "zone_bin", "year"]
MEMBERS = [("xgb", 1.5), ("lgbm", 1.5), ("cat", 1.86)]


def make(fam, spw, seed):
    if fam == "xgb":
        return xgb.XGBClassifier(n_estimators=2000, learning_rate=0.06, max_depth=3,
                                 scale_pos_weight=spw, subsample=0.85, colsample_bytree=0.9,
                                 reg_lambda=1.0, random_state=seed, eval_metric="logloss",
                                 early_stopping_rounds=100, verbosity=0)
    if fam == "lgbm":
        return lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.04, num_leaves=15,
                                  scale_pos_weight=spw, min_child_samples=40, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=seed, verbose=-1)
    return CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=4, scale_pos_weight=spw,
                              l2_leaf_reg=3.0, random_seed=seed, verbose=False,
                              allow_writing_files=False, early_stopping_rounds=100)


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
    X, Xte = Xall[CORE].reset_index(drop=True), Xte_all[CORE].reset_index(drop=True)
    village = pd.factorize(tr_raw.location.str.split(",").str[:2].str.join(","))[0]

    oofs = {n: np.zeros((len(FOLDS), len(y))) for n in MEMBERS}
    tests = {n: np.zeros(len(Xte)) for n in MEMBERS}
    stops = {n: [] for n in MEMBERS}

    for n in MEMBERS:
        fam, spw = n
        for fi, fs in enumerate(FOLDS):
            splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(X, y, village))
            for s in SEEDS:
                te_acc = np.zeros(len(Xte))
                for tr_i, va_i in splits:
                    m = make(fam, spw, s * 10 + fi)
                    fit(m, fam, X.iloc[tr_i], y.iloc[tr_i], X.iloc[va_i], y.iloc[va_i])
                    oofs[n][fi][va_i] += m.predict_proba(X.iloc[va_i])[:, 1] / len(SEEDS)
                    te_acc += m.predict_proba(Xte)[:, 1] / len(splits)
                    if fam == "xgb":
                        stops[n].append(m.get_booster().best_iteration + 1)
                    elif fam == "cat":
                        stops[n].append(m.get_best_iteration() + 1)
                    else:
                        stops[n].append(m.best_iteration_ + 1)
                tests[n] += te_acc / len(FOLDS)
        f1 = f1_score(y, (oofs[n].mean(0) >= .5).astype(int))
        auc = roc_auc_score(y, oofs[n].mean(0))
        print(f"core-{fam}(spw={spw}): combo={0.6*f1+.4*auc:.4f} F1={f1:.4f} AUC={auc:.4f} "
              f"median_rounds={int(np.median(stops[n]))}")

    def score(w):
        tot = 0.0
        for fi in range(len(FOLDS)):
            p = sum(w[i] * oofs[n][fi] for i, n in enumerate(MEMBERS))
            tot += 0.6 * f1_score(y, (p >= .5).astype(int)) + 0.4 * roc_auc_score(y, p)
        return tot / len(FOLDS)

    rng = np.random.default_rng(42)
    best = (-1, None)
    for _ in range(20000):
        w = rng.dirichlet(np.ones(len(MEMBERS)) * 2)
        s = score(w)
        if s > best[0]:
            best = (s, w)
    print("\nbest mean-combo=%.5f weights=%s" % (best[0], dict(zip([f[0] for f in MEMBERS], best[1].round(3)))))

    # full-data fits at CV-derived round counts
    tp = np.zeros(len(Xte))
    for i, n in enumerate(MEMBERS):
        fam, spw = n
        rounds = int(np.median(stops[n]) * 1.15)
        cfg_p = np.zeros(len(Xte))
        for s in SEEDS:
            if fam == "cat":
                m = make(fam, spw, s)
                m.set_params(iterations=rounds, early_stopping_rounds=None)
                m.fit(X, y)
            else:
                m = make(fam, spw, s)
                m.set_params(n_estimators=rounds)
                if fam == "xgb":
                    m.set_params(early_stopping_rounds=None)
                m.fit(X, y)
            cfg_p += m.predict_proba(Xte)[:, 1] / len(SEEDS)
        tp += best[1][i] * cfg_p
    sub = pd.DataFrame({"ID": te_raw.ID, "TargetF1": (tp >= .5).astype(int),
                        "TargetRAUC": np.asarray(tp, dtype=float)})
    sub.to_csv("submissions/submission_v5.csv", index=False)
    assert list(sub.columns) == ["ID", "TargetF1", "TargetRAUC"] and sub.ID.equals(te_raw.ID)
    prev = pd.read_csv("submissions/submission_v4.csv")
    print("saved | pos-rate %.3f | diff vs v4: %d rows" % (sub.TargetF1.mean(), (sub.TargetF1 != prev.TargetF1).sum()))


if __name__ == "__main__":
    main()
