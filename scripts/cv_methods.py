"""Ten CV schemes x five models. Does ANY validation design flip core vs full?

Schemes deliberately span the axes that could matter here:
  random / grouped-by-village / grouped-by-site / grouped-by-district /
  leave-one-village-out / Monte-Carlo / temporal rolling-origin /
  stratified-by-(age-band x era) / adversarially reweighted to look like TEST.

Rolling-origin scores ONLY rows that actually received a prediction -- the bug in
experiment.py was scoring a partially-filled OOF vector against the full y.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold, GroupKFold,
                                     StratifiedShuffleSplit, GroupShuffleSplit)
from sklearn.metrics import f1_score, roc_auc_score
from features import load_data, engineer
from features_v3 import add_v3, V3_COLS
from harness import build
from concurrent.futures import ProcessPoolExecutor

CORE = ["age","age_log1p","is_infant","is_under5","is_child","is_elderly","gender_bin","zone_bin","year"]
XGBP = dict(learning_rate=.05,max_depth=3,min_child_weight=1,subsample=.85,colsample_bytree=.9,reg_lambda=1.)
CATP = dict(learning_rate=.05,depth=4,l2_leaf_reg=3.)
LGBP = dict(learning_rate=.05,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1.)
SEEDS = (42, 7)
_G = {}

def _init(payload):
    _G.update(payload)

def _run_job(job):
    sname, mname, rep, ck, fam, p, spw, rd, adv, splits = job
    X = _G["COLS"][ck]; y = _G["y"]; yv = _G["yv"]; advw = _G["advw"]
    oof = np.zeros(len(y)); seen = np.zeros(len(y), bool)
    for tr_i, va_i in splits:
        a = np.zeros(len(va_i))
        for s in SEEDS:
            m = build(fam, p, spw, rd, s); m.fit(X.iloc[tr_i], y.iloc[tr_i])
            a += m.predict_proba(X.iloc[va_i])[:, 1] / len(SEEDS)
        oof[va_i] = a; seen[va_i] = True
    return sname, mname, rep, wcombo(yv, oof, seen, advw if adv else None)

def wcombo(y, p, m=None, w=None):
    if m is None: m = np.ones(len(y), bool)
    return (0.6*f1_score(y[m], (p[m] >= .5).astype(int), sample_weight=None if w is None else w[m])
            + 0.4*roc_auc_score(y[m], p[m], sample_weight=None if w is None else w[m]))

def main():
    tr, te = load_data()
    y = tr.is_climate_sensitive.reset_index(drop=True); yv = y.values
    Xtr = add_v3(engineer(tr).reset_index(drop=True), tr.reset_index(drop=True))
    Xte = add_v3(engineer(te).reset_index(drop=True), te.reset_index(drop=True))
    FULL = list(engineer(tr).columns)
    COLS = {"core":CORE, "v3":CORE+V3_COLS, "full":FULL}

    village = pd.factorize(tr.location.str.split(",").str[:2].str.join(","))[0]
    site    = pd.factorize(tr.latitude.round(6).astype(str)+"_"+tr.longitude.round(6).astype(str))[0]
    dist    = pd.factorize(tr.location.str.split(",").str[-2].str.strip())[0]
    year    = tr.deathdate.dt.year.values
    band    = np.select([tr.age.values==0,(tr.age.values>=1)&(tr.age.values<5)],[0,1],default=2)
    cell    = band*2 + (year>=2018).astype(int)
    strat_cell = cell*2 + yv

    # adversarial validation: how test-like is each train row?
    advy = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    for tag, cs in [("FULL (incl. lat/lon)", FULL), ("CORE (what the model sees)", CORE)]:
        aX = pd.concat([Xtr[cs], Xte[cs]], ignore_index=True)
        am = build("xgb", dict(learning_rate=.05,max_depth=4,subsample=.9,colsample_bytree=.9), 1.0, 200, 42)
        am.fit(aX, advy)
        auc_sep = roc_auc_score(advy, am.predict_proba(aX)[:,1])
        pt = am.predict_proba(Xtr[cs])[:,1]
        w = np.clip(pt/(1-pt+1e-9), 0.05, 20.0)
        print(f"adversarial train-vs-test AUC on {tag:28s} = {auc_sep:.4f}  "
              f"weights mean={w.mean():.3f} p10={np.percentile(w,10):.3f} p90={np.percentile(w,90):.3f}")
        if cs is CORE:
            advw = w / w.mean()
    print("  (AUC 1.0 on FULL means lat/lon identify the split outright -> those weights are\n"
          "   degenerate; the reweighted scheme below uses CORE-space density ratios.)\n")

    def sk(seed):  return list(StratifiedKFold(5,shuffle=True,random_state=seed).split(Xtr,y))
    def gv(seed):  return list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(Xtr,y,village))
    def gs(seed):  return list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(Xtr,y,site))
    def gd(seed):  return [(np.where(dist!=d)[0], np.where(dist==d)[0]) for d in np.unique(dist)]
    def lovo(seed):return [(np.where(village!=v)[0], np.where(village==v)[0]) for v in np.unique(village)]
    def mc(seed):  return list(StratifiedShuffleSplit(5,test_size=.2,random_state=seed).split(Xtr,y))
    def gmc(seed): return list(GroupShuffleSplit(5,test_size=.25,random_state=seed).split(Xtr,y,village))
    def scell(seed):return list(StratifiedKFold(5,shuffle=True,random_state=seed).split(Xtr,strat_cell))
    def roll(seed):
        s=[]
        for cut in (2013,2015,2017,2019):
            a,b=np.where(year<=cut)[0], np.where(year==cut+1)[0]
            if len(b)>30 and y.iloc[b].nunique()>1: s.append((a,b))
        return s
    SCHEMES = {"StratKFold":(sk,3),"GroupVillage":(gv,3),"GroupSite":(gs,3),
               "LeaveOneDistrictOut":(gd,1),"LeaveOneVillageOut":(lovo,1),
               "MonteCarlo(strat)":(mc,3),"GroupShuffle(village)":(gmc,3),
               "StratByAgeEraCell":(scell,3),"RollingOrigin(time)":(roll,1),
               "GroupVillage+advWeight":(gv,3)}
    MODELS = {"xgb_core":("xgb",XGBP,1.86,100,"core"), "cat_core":("cat",CATP,1.86,250,"core"),
              "xgb_v3":("xgb",XGBP,1.70,100,"v3"), "xgb_full":("xgb",XGBP,1.86,100,"full"),
              "cat_full":("cat",CATP,1.86,250,"full")}

    jobs=[]
    for sname,(fn,reps) in SCHEMES.items():
        adv = sname.endswith("advWeight")
        for mname,(fam,p,spw,rd,ck) in MODELS.items():
            for r in range(reps):
                jobs.append((sname,mname,r,ck,fam,p,spw,rd,adv,fn(42+r*17)))
    print(f"running {len(jobs)} scheme x model x rep jobs on 10 workers...\n")
    got={}
    payload = {"COLS": {k: Xtr[v] for k, v in COLS.items()}, "y": y, "yv": yv, "advw": advw}
    with ProcessPoolExecutor(max_workers=10, initializer=_init, initargs=(payload,)) as ex:
        for sname,mname,rep,sc in ex.map(_run_job, jobs):
            got.setdefault((sname,mname),[]).append(sc)

    rows=[]
    hdr = f"{'scheme':24s}" + "".join(f"{m:>10s}" for m in MODELS) + f"{'core-full':>11s}"
    print(hdr); print("-"*len(hdr))
    for sname,(fn,reps) in SCHEMES.items():
        scores={m: float(np.mean(got[(sname,m)])) for m in MODELS}
        gap = np.mean([scores["xgb_core"],scores["cat_core"]]) - np.mean([scores["xgb_full"],scores["cat_full"]])
        rows.append((sname,*[scores[m] for m in MODELS],gap))
        print(f"{sname:24s}" + "".join(f"{scores[m]:10.4f}" for m in MODELS) + f"{gap:+11.4f}")
    res=pd.DataFrame(rows,columns=["scheme",*MODELS,"core_minus_full"])
    res.to_csv("artifacts/cv_methods.csv",index=False)
    print("\ncore_minus_full > 0 in %d/%d schemes"%((res.core_minus_full>0).sum(),len(res)))

if __name__ == "__main__":
    main()
