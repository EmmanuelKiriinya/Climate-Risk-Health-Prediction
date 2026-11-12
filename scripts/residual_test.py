"""Which JOIN KEYS still carry signal after the core model?

Any external dataset attaches to a record through some key: location, time, or
location x time. If, conditional on the core model's predictions, the target
shows no more between-cell variation than binomial chance at a given key, then
NO dataset indexed by that key can add information -- whatever it measures.

Test: OOF probs p_i from the core trio. For each cell c, O_c = observed positives,
E_c = sum p_i, V_c = sum p_i(1-p_i). Dispersion X2 = sum (O_c-E_c)^2 / V_c ~ chi2(df=#cells)
under the null of no residual cell effect. Reported alongside a permutation null,
because chi2 is only asymptotic at these cell sizes.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from scipy import stats
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from harness import evaluate

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP = dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP = dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True); yv = y.values
    Xtr = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    groups = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]

    # honest OOF from the v7 trio
    _,_,_,o1,_ = evaluate("xgb", XGBP, 1.86, 100, Xtr[CORE], y, groups, fold_seeds=range(10))
    _,_,_,o2,_ = evaluate("cat", CATP, 1.86, 250, Xtr[CORE], y, groups, fold_seeds=range(10))
    _,_,_,o3,_ = evaluate("xgb", XGBP, 1.70, 100, Xtr[CORE+V3_COLS], y, groups, fold_seeds=range(10))
    p = (o1.mean(0) + o2.mean(0) + o3.mean(0)) / 3
    print(f"core-trio OOF: mean p={p.mean():.4f}, observed rate={yv.mean():.4f}\n")

    d = tr.reset_index(drop=True)
    site = d.latitude.round(6).astype(str)+"_"+d.longitude.round(6).astype(str)
    dist = d.location.str.split(",").str[-2].str.strip()
    vil  = d.location.str.split(",").str[:2].str.join(",")
    yr   = d.deathdate.dt.year.astype(str); mo = d.deathdate.dt.month.astype(str)
    q    = d.deathdate.dt.to_period("Q").astype(str)

    keys = {
      "location only (site)":            site,
      "location only (village)":         vil,
      "location only (district)":        dist,
      "time only (year)":                yr,
      "time only (month-of-year)":       mo,
      "time only (year-quarter)":        q,
      "location x time (site x year)":   site+"|"+yr,
      "location x time (district x year)":dist+"|"+yr,
      "location x time (site x month)":  site+"|"+mo,
      "location x time (village x quarter)": vil+"|"+q,
    }
    rng = np.random.default_rng(42)
    print(f"{'join key':36s} {'cells':>6s} {'n>=10':>6s} {'X2/df':>7s} {'chi2 p':>9s} {'perm p':>7s}  verdict")
    print("-"*104)
    rows=[]
    for name, k in keys.items():
        kk = pd.factorize(k)[0]
        df_ = pd.DataFrame({"c":kk,"y":yv,"p":p})
        g = df_.groupby("c").agg(n=("y","size"), O=("y","sum"), E=("p","sum"),
                                 V=("p", lambda s: float((s*(1-s)).sum())))
        g = g[g.n >= 10]
        if len(g) < 3:
            print(f"{name:36s} {len(kk):6d} {len(g):6d}   (too few usable cells)"); continue
        X2 = float(((g.O-g.E)**2 / g.V).sum()); dfree = len(g)
        pv = 1 - stats.chi2.cdf(X2, dfree)
        # permutation null: reshuffle records across cells, keeping cell sizes
        perm=[]
        for _ in range(400):
            idx = rng.permutation(len(yv))
            t = pd.DataFrame({"c":kk,"y":yv[idx],"p":p[idx]}).groupby("c").agg(
                n=("y","size"),O=("y","sum"),E=("p","sum"),V=("p",lambda s: float((s*(1-s)).sum())))
            t=t[t.n>=10]; perm.append(float(((t.O-t.E)**2/t.V).sum())/len(t))
        perm=np.array(perm); pp=float((perm >= X2/dfree).mean())
        verdict = "RESIDUAL SIGNAL" if pp < 0.05 else "no signal beyond chance"
        rows.append((name,len(g),X2/dfree,pv,pp,verdict))
        print(f"{name:36s} {len(np.unique(kk)):6d} {len(g):6d} {X2/dfree:7.3f} {pv:9.4f} {pp:7.3f}  {verdict}")
    pd.DataFrame(rows,columns=["key","cells","X2_df","chi2_p","perm_p","verdict"]).to_csv(
        "artifacts/residual_test.csv",index=False)

if __name__ == "__main__":
    main()
