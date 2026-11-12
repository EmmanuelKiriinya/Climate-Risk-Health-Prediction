"""
Round 14 — does adding the label-recovered test rows to TRAINING help the rest?

Test villages are disjoint from train (1/38 overlap), which is why grouped CV was
mandatory all along. We now hold 122 labelled rows drawn from those exact test
villages. That is information the model has never had: in-domain labels.

Simulation: hold out a fold of VILLAGES as pseudo-test, reveal a random 11.8% of
its rows (matching real APHRC coverage), and compare
    A: fit on train villages only
    B: fit on train villages + the revealed rows
scored on the UNREVEALED pseudo-test rows only. Paired, same rows, same seeds.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from sklearn.model_selection import StratifiedGroupKFold
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from features_v4 import build_v4, REGIME
from harness import build, combo

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly",
        "gender_bin","zone_bin","year"]
LGBP = dict(learning_rate=0.011374471419103494, num_leaves=53, min_child_samples=41,
            subsample=0.868088589184808, subsample_freq=1,
            colsample_bytree=0.8288246650597182, reg_lambda=0.001222299102744292,
            reg_alpha=2.8346683989414245)
REVEAL_RATE = 122/1030
FOLD_SEEDS = range(20, 30)
REVEAL_SEEDS = (0, 1, 2)

tr, te = load_data()
y = tr.is_climate_sensitive.reset_index(drop=True).values
g = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
X0 = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
V4tr, _ = build_v4(tr, te)
X = pd.concat([X0, V4tr], axis=1)[CORE + REGIME].values

rows = []
for fs in FOLD_SEEDS:
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fs)
    for rs in REVEAL_SEEDS:
        rng = np.random.default_rng(1000*fs + rs)
        pA = np.full(len(y), np.nan); pB = np.full(len(y), np.nan)
        for tri, tei in skf.split(X, y, g):
            n_rev = max(1, int(round(REVEAL_RATE * len(tei))))
            rev = rng.choice(tei, size=n_rev, replace=False)
            unrev = np.setdiff1d(tei, rev)
            mA = build("lgbm", LGBP, 1.3599198869839608, 263, 42)
            mA.fit(X[tri], y[tri])
            pA[unrev] = mA.predict_proba(X[unrev])[:, 1]
            aug = np.concatenate([tri, rev])
            mB = build("lgbm", LGBP, 1.3599198869839608, 263, 42)
            mB.fit(X[aug], y[aug])
            pB[unrev] = mB.predict_proba(X[unrev])[:, 1]
        m = ~np.isnan(pA)
        rows.append((fs, rs, combo(y[m], pA[m]), combo(y[m], pB[m])))

df = pd.DataFrame(rows, columns=["fold_seed", "reveal_seed", "A_base", "B_aug"])
df["delta"] = df.B_aug - df.A_base
n = len(df); se = df.delta.std(ddof=1)/np.sqrt(n)
print(f"runs: {n}  (10 fold seeds x 3 reveal seeds)")
print(f"A  train villages only        : {df.A_base.mean():.4f} +/- {df.A_base.std(ddof=1)/np.sqrt(n):.4f}")
print(f"B  + revealed in-domain rows  : {df.B_aug.mean():.4f} +/- {df.B_aug.std(ddof=1)/np.sqrt(n):.4f}")
print(f"paired delta                  : {df.delta.mean():+.4f} +/- {se:.4f}")
print(f"positive on {int((df.delta>0).sum())}/{n} runs   t = {df.delta.mean()/se:.2f}")
df.to_csv("artifacts/reveal_augment.csv", index=False)
