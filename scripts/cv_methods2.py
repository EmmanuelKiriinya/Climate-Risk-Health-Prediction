"""Alternative CV designs, batch 2 -- probing axes batch 1 did not.

  fold count (k=3/5/10/20)      : does resolution change the ranking?
  leave-P-villages-out (P=2..19): how much held-out breadth matters
  leave-one-YEAR-out            : generalisation to an unseen year
  purged + embargoed time CV    : de Prado style, no temporal adjacency leak
  combinatorial purged CV       : many backtest paths instead of one
  test-composition-matched      : validation reweighted to the TEST's
                                  (age-band x era x zone) marginal -- the most
                                  direct answer to "does CV represent the test?"
  bootstrap .632+               : out-of-bag rather than fold-based
"""
import sys, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import StratifiedGroupKFold, LeavePGroupsOut
from sklearn.metrics import f1_score, roc_auc_score
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from harness import build

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP = dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP = dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
SEEDS=(42,7)
_G={}
def _init(p): _G.update(p)

def wcombo(y,p,m,w=None):
    ww = None if w is None else w[m]
    if len(np.unique(y[m]))<2: return np.nan
    return (0.6*f1_score(y[m],(p[m]>=.5).astype(int),sample_weight=ww)
            + 0.4*roc_auc_score(y[m],p[m],sample_weight=ww))

def _job(j):
    sname,mname,rep,ck,fam,pr,spw,rd,splits,usew = j
    X=_G["COLS"][ck]; y=_G["y"]; yv=_G["yv"]; w=_G["tw"]
    oof=np.zeros(len(y)); seen=np.zeros(len(y),bool)
    for tr_i,va_i in splits:
        a=np.zeros(len(va_i))
        for s in SEEDS:
            m=build(fam,pr,spw,rd,s); m.fit(X.iloc[tr_i],y.iloc[tr_i])
            a+=m.predict_proba(X.iloc[va_i])[:,1]/len(SEEDS)
        oof[va_i]=a; seen[va_i]=True
    return sname,mname,rep,wcombo(yv,oof,seen,w if usew else None)

def main():
    tr,te=load_data(); y=tr.is_climate_sensitive.reset_index(drop=True); yv=y.values
    Xtr=add_v3(engineer(tr).reset_index(drop=True),tr.reset_index(drop=True))
    Xte=add_v3(engineer(te).reset_index(drop=True),te.reset_index(drop=True))
    COLS={"core":Xtr[CORE],"v3":Xtr[CORE+V3_COLS],"full":Xtr[list(engineer(tr).columns)]}
    vil=pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    yr=tr.deathdate.dt.year.values; t=tr.deathdate.values.astype("datetime64[D]").astype(int)

    # ---- test-composition weights on (age-band x era x zone) ----
    def cells(d):
        a=d.age.values; e=(d.deathdate.dt.year.values>=2018).astype(int)
        b=np.select([a==0,(a>=1)&(a<5),(a>=5)&(a<18),(a>=18)&(a<60)],[0,1,2,3],default=4)
        z=(d.zone.values=="Rural").astype(int)
        return b*4+e*2+z
    ctr,cte=cells(tr),cells(te)
    ptr=pd.Series(ctr).value_counts(normalize=True); pte=pd.Series(cte).value_counts(normalize=True)
    tw=np.array([pte.get(c,0.)/max(ptr.get(c,1e-9),1e-9) for c in ctr]); tw=tw/tw.mean()
    print(f"test-composition weights: mean={tw.mean():.3f} p10={np.percentile(tw,10):.3f} "
          f"p90={np.percentile(tw,90):.3f} max={tw.max():.3f}")

    def kfold(k):  return lambda s: list(StratifiedGroupKFold(k,shuffle=True,random_state=s).split(Xtr,y,vil))
    def lpgo(P):
        def f(s):
            rng=np.random.default_rng(s); g=np.unique(vil); out=[]
            for _ in range(6):
                ho=rng.choice(g,P,replace=False); m=np.isin(vil,ho)
                if y[m].nunique()>1: out.append((np.where(~m)[0],np.where(m)[0]))
            return out
        return f
    def loyo(s): return [(np.where(yr!=u)[0],np.where(yr==u)[0]) for u in np.unique(yr)
                         if (yr==u).sum()>=30 and y[yr==u].nunique()>1]
    def purged(s, emb=60):
        out=[]
        for cut in (2012,2014,2016,2018,2020):
            lo=np.datetime64(f"{cut+1}-01-01").astype(int)
            a=np.where(t < lo-emb)[0]; b=np.where(yr==cut+1)[0]
            if len(b)>30 and y[b].nunique()>1: out.append((a,b))
        return out
    def cpcv(s):
        blocks=np.array_split(np.argsort(t),6); out=[]
        for combo in itertools.combinations(range(6),2):
            va=np.concatenate([blocks[i] for i in combo])
            tr_i=np.concatenate([blocks[i] for i in range(6) if i not in combo])
            if y[va].nunique()>1: out.append((tr_i,va))
        return out
    def boot(s):
        rng=np.random.default_rng(s); out=[]
        for _ in range(8):
            b=rng.choice(len(y),len(y),replace=True); oob=np.setdiff1d(np.arange(len(y)),b)
            if len(oob)>50 and y[oob].nunique()>1: out.append((b,oob))
        return out

    SCH={"GroupVillage k=3":(kfold(3),2,False),"GroupVillage k=5":(kfold(5),2,False),
         "GroupVillage k=10":(kfold(10),2,False),"GroupVillage k=20":(kfold(20),1,False),
         "LeaveP=2VillagesOut":(lpgo(2),2,False),"LeaveP=5VillagesOut":(lpgo(5),2,False),
         "LeaveP=10VillagesOut":(lpgo(10),2,False),"LeaveP=19VillagesOut":(lpgo(19),2,False),
         "LeaveOneYearOut":(loyo,1,False),"PurgedEmbargoTime":(purged,1,False),
         "CombinatorialPurgedCV":(cpcv,1,False),"Bootstrap .632 (OOB)":(boot,2,False),
         "TestCompositionMatched":(kfold(5),3,True)}
    MOD={"xgb_core":("xgb",XGBP,1.86,100,"core"),"cat_core":("cat",CATP,1.86,250,"core"),
         "xgb_v3":("xgb",XGBP,1.70,100,"v3"),"xgb_full":("xgb",XGBP,1.86,100,"full")}

    jobs=[]
    for sn,(fn,reps,uw) in SCH.items():
        for mn,(fam,pr,spw,rd,ck) in MOD.items():
            for r in range(reps): jobs.append((sn,mn,r,ck,fam,pr,spw,rd,fn(42+r*17),uw))
    print(f"running {len(jobs)} jobs on 10 workers...\n")
    got={}
    with ProcessPoolExecutor(10,initializer=_init,
                             initargs=({"COLS":COLS,"y":y,"yv":yv,"tw":tw},)) as ex:
        for sn,mn,r,sc in ex.map(_job,jobs): got.setdefault((sn,mn),[]).append(sc)

    hdr=f"{'scheme':26s}"+"".join(f"{m:>10s}" for m in MOD)+f"{'core-full':>11s}"
    print(hdr); print("-"*len(hdr)); rows=[]
    for sn in SCH:
        sc={m:float(np.nanmean(got[(sn,m)])) for m in MOD}
        gap=np.mean([sc["xgb_core"],sc["cat_core"]])-sc["xgb_full"]
        rows.append((sn,*[sc[m] for m in MOD],gap))
        print(f"{sn:26s}"+"".join(f"{sc[m]:10.4f}" for m in MOD)+f"{gap:+11.4f}")
    res=pd.DataFrame(rows,columns=["scheme",*MOD,"core_minus_full"])
    res.to_csv("artifacts/cv_methods2.csv",index=False)
    print(f"\ncore > full in {(res.core_minus_full>0).sum()}/{len(res)} schemes")

if __name__=="__main__":
    main()
