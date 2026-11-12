"""Build record-level features from cached NASA POWER daily series."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from features import load_data

CACHE = "data_external/power_cache"


def site_features(la, lo):
    df = pd.read_csv(f"{CACHE}/{la}_{lo}.csv", parse_dates=["date"]).set_index("date")
    # sentinel values (-999) -> nan
    df = df.mask(df < -900)

    f = pd.DataFrame(index=df.index)
    rh, t2m, pr = df.RH2M, df.T2M, df.PRECTOTCORR

    f["rh_mean_7d"] = rh.rolling(7).mean()
    f["rh_mean_30d"] = rh.rolling(30).mean()
    f["rh_mean_90d"] = rh.rolling(90).mean()
    f["qv_mean_30d"] = df.QV2M.rolling(30).mean()

    svp = 0.6108 * np.exp(17.27 * t2m / (t2m + 237.3))
    f["vpd_mean_7d"] = (svp * (1 - rh / 100)).rolling(7).mean()
    f["vpd_mean_30d"] = (svp * (1 - rh / 100)).rolling(30).mean()

    f["wind_mean_30d"] = df.WS2M.rolling(30).mean()
    f["wind_max_30d"] = df.WS2M_MAX.rolling(30).max()
    f["insol_mean_30d"] = df.ALLSKY_SFC_SW_DWN.rolling(30).mean()
    f["insol_sum_90d"] = df.ALLSKY_SFC_SW_DWN.rolling(90).sum()

    f["precip_pow_sum_30d"] = pr.rolling(30).sum()
    f["precip_pow_sum_90d"] = pr.rolling(90).sum()
    f["wet_days_30d"] = (pr >= 1).rolling(30).sum().values

    dry = (pr < 1).astype(int)
    grp = dry * (dry != dry.shift()).cumsum()
    streak = dry.groupby(grp).cumsum() * dry
    f["dry_spell_max_30d"] = streak.rolling(30).max()

    f["t2m_range_mean_30d"] = df.T2M_RANGE.rolling(30).mean()

    # site-month normals -> anomalies
    m = df.index.month
    for c in ["rh_mean_30d", "qv_mean_30d", "insol_mean_30d"]:
        norm = f[c].groupby(m).transform(lambda s: s.mean())
        f[f"{c}_anom"] = f[c] - norm

    return f


def build():
    tr, te = load_data()
    comb = pd.concat([tr.assign(src="tr"), te.assign(src="te")], ignore_index=True)
    comb["site"] = list(zip(comb.latitude.round(4), comb.longitude.round(4)))

    parts = {}
    for (la, lo) in sorted(set(comb.site)):
        sf = site_features(la, lo)
        parts[(la, lo)] = sf.stack().rename("v")

    long = pd.concat(parts, names=["la", "lo", "date", "feat"]).reset_index()
    wide = (long.pivot_table(index=["la", "lo", "date"], columns="feat", values="v")
            .reset_index().rename(columns={"la": "lat", "lo": "lon"}))

    comb["date"] = pd.to_datetime(comb.deathdate)
    rec = comb[["ID", "date"]].copy()
    rec["lat"] = comb.site.str[0]
    rec["lon"] = comb.site.str[1]
    merged = rec.merge(wide, on=["lat", "lon", "date"], how="left").drop(columns=["lat", "lon", "date"])
    assert not merged.isna().any().any(), merged.isna().sum()

    out_tr = merged[merged.ID.isin(tr.ID)].drop(columns="ID").reset_index(drop=True)
    out_te = merged[merged.ID.isin(te.ID)].drop(columns="ID").reset_index(drop=True)
    out_tr.to_csv("artifacts/power_features_train.csv", index=False)
    out_te.to_csv("artifacts/power_features_test.csv", index=False)
    print(out_tr.shape, out_te.shape)


if __name__ == "__main__":
    build()
