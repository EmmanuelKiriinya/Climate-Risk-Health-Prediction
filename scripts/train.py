"""Model comparison with StratifiedGroupKFold (groups = village/coord) and StratifiedKFold.

Score = 0.6 * F1(@0.5) + 0.4 * ROC-AUC, mirroring the leaderboard.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, "scripts")
from features import load_data, engineer

SEED = 42
N_SPLITS = 5


def combo(y, p):
    return 0.6 * f1_score(y, (p >= 0.5).astype(int)) + 0.4 * roc_auc_score(y, p)


def cv_eval(name, fit_predict, X, y, groups, grouped=True):
    if grouped:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y, groups))
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y))
    oof = np.zeros(len(y))
    for tr_idx, va_idx in splits:
        Xtr_, ytr_ = X.iloc[tr_idx], y.iloc[tr_idx]
        Xva_, yva_ = X.iloc[va_idx], y.iloc[va_idx]
        oof[va_idx] = fit_predict(Xtr_, ytr_, Xva_, yva_)
    f1 = f1_score(y, (oof >= 0.5).astype(int))
    auc = roc_auc_score(y, oof)
    print(f"{name:34s} F1={f1:.4f}  AUC={auc:.4f}  combo={combo(y, oof):.4f}")
    return oof


def lgb_factory(spw=1.0, lr=0.05, leaves=31, extra=None):
    def fit_predict(Xtr, ytr, Xva, yva):
        model = lgb.LGBMClassifier(
            n_estimators=1500, learning_rate=lr, num_leaves=leaves,
            scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_lambda=1.0, random_state=SEED, verbose=-1,
            **(extra or {}),
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return model.predict_proba(Xva)[:, 1]
    return fit_predict


def xgb_factory(spw=1.0, lr=0.05):
    def fit_predict(Xtr, ytr, Xva, yva):
        model = xgb.XGBClassifier(
            n_estimators=1500, learning_rate=lr, max_depth=5,
            scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=SEED, eval_metric="logloss", early_stopping_rounds=100,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return model.predict_proba(Xva)[:, 1]
    return fit_predict


def lr_factory(C=1.0):
    def fit_predict(Xtr, ytr, Xva, yva=None):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
        model.fit(Xtr, ytr)
        return model.predict_proba(Xva)[:, 1]
    return fit_predict


def main():
    tr, te = load_data()
    X = engineer(tr)
    y = tr.is_climate_sensitive
    village = tr.location.str.split(",").str[:2].str.join(",")
    groups = pd.factorize(village)[0]

    print(f"train {X.shape}, positives {y.mean():.3f}, groups {len(np.unique(groups))}\n")

    results = {}
    for grouped in (True, False):
        tag = "GroupCV" if grouped else "StratCV"
        print(f"--- {tag} ---")
        results[(tag, "LR")] = cv_eval(f"[{tag}] LogisticReg", lr_factory(), X, y, groups, grouped)
        for spw in (1.0, 1.86, 2.6):
            results[(tag, f"LGBM spw{spw}")] = cv_eval(
                f"[{tag}] LGBM spw={spw}", lgb_factory(spw=spw), X, y, groups, grouped)
        results[(tag, "XGB")] = cv_eval(f"[{tag}] XGB spw=1.86", xgb_factory(spw=1.86), X, y, groups, grouped)

        # simple blend of best two
        lgb_oof = results[(tag, "LGBM spw1.86")]
        xgb_oof = results[(tag, "XGB")]
        blend = 0.5 * lgb_oof + 0.5 * xgb_oof
        f1 = f1_score(y, (blend >= 0.5).astype(int)); auc = roc_auc_score(y, blend)
        print(f"[{tag}] blend LGBM+XGB           F1={f1:.4f}  AUC={auc:.4f}  combo={combo(y, blend):.4f}")
        print()


if __name__ == "__main__":
    main()
