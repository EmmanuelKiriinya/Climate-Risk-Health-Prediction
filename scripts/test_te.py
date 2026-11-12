"""Out-of-fold Bayesian-shrunk target encoding + monotone constraints.

TE is computed strictly INSIDE each training fold (never on validation rows), so
this is leak-free. Smoothing: rate_k = (sum_y_k + m*prior) / (n_k + m).

Motivation: exact-age target rate is a sharp non-monotone hump
(0->0.788, 1->0.936, 2->0.959, 3->0.899, 4->0.797) that a depth-3 tree has to
approximate with axis splits; a shrunk encoding represents it directly, and
shrinkage handles the sparse high-age cells that trees smooth over badly.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import StratifiedGroupKFold
from features import load_data, engineer
from harness import build, combo, parts

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP = dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
SEEDS=(42,7)
_G={}
def _init(p): _G.update(p)

def te_fit_apply(keys_tr, y_tr, keys_va, m, prior):
    df = pd.DataFrame({"k": keys_tr, "y": y_tr})
    g = df.groupby("k").y.agg(["sum","size"])
    rate = ((g["sum"] + m*prior) / (g["size"] + m))
    return pd.Series(keys_va).map(rate).fillna(prior).values

def _job(j):
    tag, keyspec, m, fold_seed, mono = j
    X=_G["X"]; y=_G["y"]; yv=_G["yv"]; g=_G["g"]; KEYS=_G["KEYS"]
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=fold_seed).split(X,y,g))
    oof=np.zeros(len(y))
    for tr_i,va_i in splits:
        Xtr,Xva = X.iloc[tr_i].copy(), X.iloc[va_i].copy()
        prior = float(yv[tr_i].mean())
        for kn in keyspec:
            k = KEYS[kn]
            Xtr[f"te_{kn}"] = te_fit_apply(k[tr_i], yv[tr_i], k[tr_i], m, prior)
            Xva[f"te_{kn}"] = te_fit_apply(k[tr_i], yv[tr_i], k[va_i], m, prior)
        a=np.zeros(len(va_i))
        for s in SEEDS:
            kw = dict(monotone_constraints="("+",".join("0" for _ in Xtr.columns)+")") if False else {}
            mdl=build("xgb",{**XGBP,**({"monotone_constraints":mono} if mono else {})},1.86,100,s)
            mdl.fit(Xtr,y.iloc[tr_i]); a+=mdl.predict_proba(Xva)[:,1]/len(SEEDS)
        oof[va_i]=a
    return tag, m, fold_seed, combo(yv,oof)

def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True); yv=y.values
    X=engineer(tr).reset_index(drop=True)[CORE]
    g=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    age=tr.age.values.astype(int); yr=tr.deathdate.dt.year.values; era=(yr>=2018).astype(int)
    band=np.select([age==0,(age>=1)&(age<5),(age>=5)&(age<18),(age>=18)&(age<60)],[0,1,2,3],default=4)
    KEYS={"age":np.minimum(age,85),
          "age_era":np.minimum(age,85)*2+era,
          "band_year":band*100+yr,
          "band_era":band*2+era}
    mono_year = "(" + ",".join("-1" if c=="year" else "0" for c in CORE) + ")"

    RUNS=[("baseline (no TE)",(),0,None),
          ("TE age",("age",),20,None), ("TE age m=5",("age",),5,None),
          ("TE age m=50",("age",),50,None),
          ("TE age_era",("age_era",),20,None), ("TE age_era m=50",("age_era",),50,None),
          ("TE band_year",("band_year",),20,None),
          ("TE band_era",("band_era",),20,None),
          ("TE age + band_year",("age","band_year"),20,None),
          ("monotone year<=0",(),0,mono_year),
          ("TE age + monotone year",("age",),20,mono_year)]
    jobs=[(t,ks,m,fs,mo) for (t,ks,m,mo) in RUNS for fs in list(range(6))+list(range(100,106))]
    print(f"running {len(jobs)} jobs on 10 workers...\n")
    got={}
    with ProcessPoolExecutor(10,initializer=_init,
            initargs=({"X":X,"y":y,"yv":yv,"g":g,"KEYS":KEYS},)) as ex:
        for tag,m,fs,sc in ex.map(_job,jobs): got.setdefault((tag,fs<100),[]).append(sc)
    print(f"{'variant':26s} {'sel(0-5)':>9s} {'hold(100-105)':>14s} {'drift':>8s}")
    rows=[]
    for (t,ks,m,mo) in RUNS:
        a=float(np.mean(got[(t,True)])); b=float(np.mean(got[(t,False)]))
        rows.append((t,a,b,b-a)); print(f"{t:26s} {a:9.4f} {b:14.4f} {b-a:+8.4f}")
    pd.DataFrame(rows,columns=["variant","sel","hold","drift"]).to_csv("artifacts/te_results.csv",index=False)

if __name__=="__main__":
    main()
