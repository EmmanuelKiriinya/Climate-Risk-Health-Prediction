"""Honest evaluation harness.

Two fixes over the round-1..5 protocol:
  1. NO early stopping on the validation fold. n_estimators is a hyperparameter.
     ES-on-val selects the round count that looks best on the very rows being
     scored -> optimistic, and optimistic by a DIFFERENT amount per config,
     which corrupts config ranking.
  2. Repeated grouped CV over many fold seeds, so every reported number comes
     with a standard error. Deltas smaller than the SE are not decisions.
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

N_SPLITS = 5


def combo(y, p):
    return 0.6 * f1_score(y, (np.asarray(p) >= 0.5).astype(int)) + 0.4 * roc_auc_score(y, p)


def parts(y, p):
    return (f1_score(y, (np.asarray(p) >= 0.5).astype(int)), roc_auc_score(y, p))


def build(fam, params, spw, rounds, seed):
    """Construct a model with a FIXED round count and no early stopping."""
    if fam == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=rounds, scale_pos_weight=spw, random_state=seed,
            eval_metric="logloss", verbosity=0, n_jobs=1, **params)
    if fam == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=rounds, scale_pos_weight=spw, random_state=seed,
            verbose=-1, n_jobs=1, **params)
    if fam == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=rounds, scale_pos_weight=spw, random_seed=seed,
            verbose=False, allow_writing_files=False, thread_count=1, **params)
    if fam == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=rounds, class_weight={0: 1.0, 1: spw}, random_state=seed, **params)
    raise ValueError(fam)


def _one_fold_seed(job):
    """Run one fold-seed: 5 folds x model seeds -> OOF vector (+ optional test preds)."""
    fam, params, spw, rounds, model_seeds, fold_seed, Xv, yv, gv, Xtev = job
    X = pd.DataFrame(Xv[0], columns=Xv[1])
    y = pd.Series(yv)
    groups = gv
    Xte = pd.DataFrame(Xtev[0], columns=Xtev[1]) if Xtev is not None else None

    oof = np.zeros(len(y))
    te = np.zeros(len(Xte)) if Xte is not None else None
    splits = list(StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                       random_state=fold_seed).split(X, y, groups))
    for tr_i, va_i in splits:
        for s in model_seeds:
            m = build(fam, params, spw, rounds, s * 100 + fold_seed)
            m.fit(X.iloc[tr_i], y.iloc[tr_i])
            oof[va_i] += m.predict_proba(X.iloc[va_i])[:, 1] / len(model_seeds)
            if Xte is not None:
                te += m.predict_proba(Xte)[:, 1] / (len(model_seeds) * len(splits))
    return oof, te


def evaluate(fam, params, spw, rounds, X, y, groups, fold_seeds=range(10),
             model_seeds=(42,), Xte=None, n_jobs=10):
    """Repeated grouped CV. Returns (mean, se, scores, oofs, test_pred)."""
    Xv = (X.values, list(X.columns))
    Xtev = (Xte.values, list(Xte.columns)) if Xte is not None else None
    jobs = [(fam, params, spw, rounds, tuple(model_seeds), fs, Xv, y.values, groups, Xtev)
            for fs in fold_seeds]
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
        res = list(ex.map(_one_fold_seed, jobs))
    oofs = np.vstack([r[0] for r in res])
    scores = np.array([combo(y, o) for o in oofs])
    te = None
    if Xte is not None:
        te = np.mean([r[1] for r in res], axis=0)
    return scores.mean(), scores.std(ddof=1) / np.sqrt(len(scores)), scores, oofs, te
