"""PCA-compressed climate blocks.

Rationale: raw climate columns are individually weak (AUC 0.50-0.55) and adding
all 16-34 of them DILUTES a depth-3 GBM at n=3k (full43 = 0.8175 vs core 0.8223).
PCA keeps the shared variance in 1-5 dense columns instead of 16-34 sparse ones,
which is the standard fix for exactly that failure mode.

PCA is fit on train+test INPUTS only (unsupervised, no target) -- legitimate and
leak-free, same convention as features_v2.py.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from features import load_data, engineer, CLIMATE_COLS

OBS = ["avg_temperature","max_temperature","min_temperature","precipitation"]
GEO = ["latitude","longitude"]

def _fit_pca(tr_block, te_block, k, kind="pca", seed=42):
    sc = StandardScaler().fit(np.vstack([tr_block, te_block]))
    A, B = sc.transform(tr_block), sc.transform(te_block)
    if kind == "kpca":
        m = KernelPCA(n_components=k, kernel="rbf", gamma=1.0/A.shape[1], random_state=seed)
    else:
        m = PCA(n_components=k, random_state=seed)
    m.fit(np.vstack([A, B]))
    return m.transform(A), m.transform(B), m

def build_pca_blocks(tr, te, ks=(1,2,3,5,8)):
    """Return dict of block-name -> (train_df, test_df) for several PCA variants."""
    Xtr, Xte = engineer(tr).reset_index(drop=True), engineer(te).reset_index(drop=True)
    # engineer() drops avg_temperature (BASE_COLS[4:] off-by-one) -- take OBS from raw
    for c in OBS:
        if c not in Xtr.columns:
            Xtr[c] = tr[c].reset_index(drop=True); Xte[c] = te[c].reset_index(drop=True)
    live = [c for c in CLIMATE_COLS if Xtr[c].nunique() > 1]      # drops hot_days_30d
    out = {}
    sources = {
        "cli":  live,                                   # 15 provided climate cols
        "clig": live + GEO + OBS,                       # + coords + observed weather
        "all":  [c for c in Xtr.columns if Xtr[c].nunique() > 1],   # everything (43)
    }
    for sname, cols in sources.items():
        A, B = Xtr[cols].values, Xte[cols].values
        for k in ks:
            if k >= len(cols): continue
            a, b, m = _fit_pca(A, B, k)
            ev = m.explained_variance_ratio_.sum()
            names = [f"{sname}pc{i+1}" for i in range(k)]
            out[f"{sname}_pca{k}"] = (pd.DataFrame(a, columns=names),
                                      pd.DataFrame(b, columns=names), ev)
    # POWER block, if present
    try:
        Ptr = pd.read_csv("artifacts/power_features_train.csv")
        Pte = pd.read_csv("artifacts/power_features_test.csv")
        pc = [c for c in Ptr.columns if Ptr[c].nunique() > 1]
        for k in (2, 3, 5):
            a, b, m = _fit_pca(Ptr[pc].values, Pte[pc].values, k)
            names = [f"powpc{i+1}" for i in range(k)]
            out[f"power_pca{k}"] = (pd.DataFrame(a, columns=names),
                                    pd.DataFrame(b, columns=names),
                                    m.explained_variance_ratio_.sum())
    except FileNotFoundError:
        pass
    # kernel PCA on the climate block
    for k in (2, 3):
        a, b, _ = _fit_pca(Xtr[live].values, Xte[live].values, k, kind="kpca")
        names = [f"kpc{i+1}" for i in range(k)]
        out[f"cli_kpca{k}"] = (pd.DataFrame(a, columns=names), pd.DataFrame(b, columns=names), np.nan)
    return Xtr, Xte, out
