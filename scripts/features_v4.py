"""v4 features: record-density / cohort-composition / continuous-time.

All computed from train+test INPUTS ONLY (dates, coords, age, gender) -- no target
anywhere, so this is transductive but leak-free (same convention as features_v2).

Three ideas, each aimed at structure found in earlier rounds but never exploited:

1. OUTBREAK DENSITY. Village-months show target overdispersion (within-group sd
   0.266 vs 0.249 iid). Deaths that cluster in space-time are more likely to share
   a cause (malaria/diarrhoeal outbreak). Counts of nearby deaths are input-only.
2. SURVEILLANCE REGIME. The label drift is a collection/coding artifact: infant
   share of records goes ~25% -> 80% after 2018. The cohort composition AT a
   site-year tells the model which regime a record sits in, at finer resolution
   than `year` alone.
3. CONTINUOUS TIME. `year` is a step function; the 2018 break and the strong
   within-post-2018 trend (0.556/0.416/0.565/0.589/0.760) may be better located on
   a continuous day index.
"""
import numpy as np
import pandas as pd

EPS = 1e-6

def build_v4(tr_raw, te_raw):
    """Return (train_block, test_block) of input-only density/composition features."""
    comb = pd.concat([tr_raw.assign(_src=0), te_raw.assign(_src=1)], ignore_index=True)
    comb["_site"] = (comb.latitude.round(6).astype(str) + "_" + comb.longitude.round(6).astype(str))
    comb["_t"] = (comb.deathdate - pd.Timestamp("2007-01-01")).dt.days.astype(float)
    comb["_yr"] = comb.deathdate.dt.year
    f = pd.DataFrame(index=comb.index)

    # --- 3. continuous time ---
    f["t_days"] = comb["_t"]
    f["t_frac_year"] = comb.deathdate.dt.dayofyear / 365.25

    # --- 1. space-time death density (self-excluded) ---
    for w in (30, 90, 365):
        cnt = np.zeros(len(comb))
        for s, idx in comb.groupby("_site").groups.items():
            t = comb.loc[idx, "_t"].values
            order = np.argsort(t); ts = t[order]
            lo = np.searchsorted(ts, ts - w, "left")
            hi = np.searchsorted(ts, ts + w, "right")
            c = (hi - lo - 1).astype(float)          # exclude self
            out = np.empty_like(c); out[order] = c
            cnt[comb.index.get_indexer(idx)] = out
        f[f"n_site_{w}d"] = cnt
    # local burst relative to the site's own baseline rate
    f["dens_ratio_30d"] = f.n_site_30d / (f.n_site_365d / 12.0 + EPS)
    f["dens_ratio_90d"] = f.n_site_90d / (f.n_site_365d / 4.0 + EPS)

    # gap to neighbouring deaths at the same site
    prv = np.zeros(len(comb)); nxt = np.zeros(len(comb))
    for s, idx in comb.groupby("_site").groups.items():
        t = np.sort(comb.loc[idx, "_t"].values)
        order = np.argsort(comb.loc[idx, "_t"].values)
        dp = np.r_[np.nan, np.diff(t)]; dn = np.r_[np.diff(t), np.nan]
        p_ = np.empty(len(t)); n_ = np.empty(len(t))
        p_[order] = dp; n_[order] = dn
        pos = comb.index.get_indexer(idx)
        prv[pos] = np.nan_to_num(p_, nan=9999.); nxt[pos] = np.nan_to_num(n_, nan=9999.)
    f["days_prev_death_site"] = prv
    f["days_next_death_site"] = nxt

    # --- 2. surveillance regime: cohort composition at site x year ---
    comb["_u1"] = (comb.age == 0).astype(float)
    comb["_u5"] = (comb.age < 5).astype(float)
    comb["_fem"] = (comb.gender == "Female").astype(float)
    gsy = comb.groupby(["_site", "_yr"])
    for col, src in [("sy_n", "_u1"), ("sy_infshare", "_u1"), ("sy_u5share", "_u5"),
                     ("sy_femshare", "_fem"), ("sy_meanage", "age")]:
        agg = gsy[src].transform("size" if col == "sy_n" else "mean")
        f[col] = agg.values
    gy = comb.groupby("_yr")
    f["y_n"] = gy["_u1"].transform("size").values            # national surveillance volume
    f["y_infshare"] = gy["_u1"].transform("mean").values
    f["y_meanage"] = gy["age"].transform("mean").values
    # where does this record sit within its own year's age distribution?
    f["age_pctile_in_year"] = gy["age"].rank(pct=True).values
    f["age_minus_year_mean"] = comb.age.values - f["y_meanage"].values

    n = len(tr_raw)
    return (f.iloc[:n].reset_index(drop=True), f.iloc[n:].reset_index(drop=True))

DENSITY = ["n_site_30d","n_site_90d","n_site_365d","dens_ratio_30d","dens_ratio_90d",
           "days_prev_death_site","days_next_death_site"]
REGIME  = ["sy_n","sy_infshare","sy_u5share","sy_femshare","sy_meanage",
           "y_n","y_infshare","y_meanage","age_pctile_in_year","age_minus_year_mean"]
CTIME   = ["t_days","t_frac_year"]
