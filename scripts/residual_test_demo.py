"""DEMOGRAPHIC join keys: has the model left any age/sex/year structure unextracted?

External sources like IHME GBD cause fractions join on (age, sex, year) -- NOT on
location or time, so the Round-7b residual test did not cover them. If, conditional
on the core model, the target shows no more between-cell variation than binomial
chance at these keys, then no demographically-indexed external dataset can help
either -- whatever it measures.

Test: OOF probs p from the v7 trio. Per cell: O=observed positives, E=sum p,
V=sum p(1-p). Dispersion X2/df, judged against a PERMUTATION null (chi2 is invalid
here because spw=1.86 leaves p miscalibrated; the permutation preserves that).
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
    X = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    g = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    _,_,_,o1,_ = evaluate("xgb", XGBP, 1.86, 100, X[CORE], y, g, fold_seeds=range(10))
    _,_,_,o2,_ = evaluate("cat", CATP, 1.86, 250, X[CORE], y, g, fold_seeds=range(10))
    _,_,_,o3,_ = evaluate("xgb", XGBP, 1.70, 100, X[CORE+V3_COLS], y, g, fold_seeds=range(10))
    p = (o1.mean(0)+o2.mean(0)+o3.mean(0))/3

    d = tr.reset_index(drop=True)
    A = d.age.astype(int).astype(str); S = d.gender; Z = d.zone
    Y = d.deathdate.dt.year.astype(str); M = d.deathdate.dt.month.astype(str)
    E = (d.deathdate.dt.year>=2018).astype(str)
    B = pd.cut(d.age,[-1,0.9,4.9,17.9,59.9,200],labels=False).astype(str)
    keys = {
      "exact age":              A,
      "age x sex":              A+"|"+S,
      "age x zone":             A+"|"+Z,
      "age x year":             A+"|"+Y,
      "age x month":            A+"|"+M,
      "sex x year":             S+"|"+Y,
      "age-band x year":        B+"|"+Y,
      "age-band x sex x era":   B+"|"+S+"|"+E,
      "age x sex x year":       A+"|"+S+"|"+Y,
      "age x era x sex x zone": A+"|"+E+"|"+S+"|"+Z,
    }
    rng = np.random.default_rng(42)
    print(f"\n{'demographic join key':26s} {'cells':>6s} {'n>=10':>6s} {'X2/df':>7s} {'perm p':>7s}  verdict")
    print("-"*78)
    rows=[]
    for name,k in keys.items():
        kk = pd.factorize(k)[0]
        gg = pd.DataFrame({"c":kk,"y":yv,"p":p}).groupby("c").agg(
            n=("y","size"),O=("y","sum"),E=("p","sum"),V=("p",lambda s: float((s*(1-s)).sum())))
        gg = gg[gg.n>=10]
        if len(gg)<3:
            print(f"{name:26s} {len(np.unique(kk)):6d} {len(gg):6d}   (too few usable cells)"); continue
        X2=float(((gg.O-gg.E)**2/gg.V).sum()); df=len(gg)
        perm=[]
        for _ in range(400):
            i=rng.permutation(len(yv))
            t=pd.DataFrame({"c":kk,"y":yv[i],"p":p[i]}).groupby("c").agg(
                n=("y","size"),O=("y","sum"),E=("p","sum"),V=("p",lambda s: float((s*(1-s)).sum())))
            t=t[t.n>=10]; perm.append(float(((t.O-t.E)**2/t.V).sum())/len(t))
        pp=float((np.array(perm)>=X2/df).mean())
        v = "RESIDUAL SIGNAL" if pp<0.05 else "no signal beyond chance"
        rows.append((name,len(gg),X2/df,pp,v))
        print(f"{name:26s} {len(np.unique(kk)):6d} {len(gg):6d} {X2/df:7.3f} {pp:7.3f}  {v}")
    pd.DataFrame(rows,columns=["key","cells","X2_df","perm_p","verdict"]).to_csv(
        "artifacts/residual_test_demo.csv",index=False)

if __name__=="__main__":
    main()
