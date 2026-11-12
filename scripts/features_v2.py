"""Feature engineering v2: seasonal climate anomalies, spatial clusters, age-climate interactions.

All normals/frequencies are computed on combined train+test INPUT features only (no target).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, "scripts")
from features import load_data, engineer, CLIMATE_COLS

N_CLUSTERS = 10
ANOM_COLS = ["tavg_7d", "tavg_30d", "tavg_90d", "tmax_30d", "tmin_30d",
             "rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "ndvi_30d", "ndvi_90d"]


def _season(month):
    # Uganda: long rains Mar-May, short rains Sep-Nov
    return np.select([month.isin([3, 4, 5]), month.isin([9, 10, 11]),
                      month.isin([12, 1, 2])], [0, 1, 2], default=3)  # 3 = Jun-Aug dry


class FeatureBuilderV2:
    """Fit on combined train+test; transform each split."""

    def fit(self, tr_raw, te_raw):
        comb = pd.concat([tr_raw.assign(src=0), te_raw.assign(src=1)], ignore_index=True)
        month = comb.deathdate.dt.month
        self.comb_size_ = len(comb)

        coords = comb[["latitude", "longitude"]].values
        self.km_ = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit(coords)
        comb["cluster"] = self.km_.labels_

        # monthly normals per cluster (and global fallback)
        g = comb.groupby(["cluster", month.rename("m")])[ANOM_COLS].mean()
        self.normals_ = g
        self.month_means_ = comb.groupby(month)[ANOM_COLS].mean()

        # coarse grid-cell frequency (~11km cells at these latitudes)
        cell = (comb.latitude.round(1).astype(str) + "_" + comb.longitude.round(1).astype(str))
        self.cell_freq_ = cell.value_counts()
        return self

    def transform(self, raw):
        X = engineer(raw)
        month = raw.deathdate.dt.month
        season = _season(month)

        coords = raw[["latitude", "longitude"]].values
        clus = self.km_.predict(coords)

        # spatial block
        blocks = {}
        sp = pd.DataFrame(index=X.index)
        for k in range(N_CLUSTERS):
            sp[f"clus_{k}"] = (clus == k).astype(int)
        cent = self.km_.cluster_centers_
        for k in range(N_CLUSTERS):
            sp[f"dist_c{k}"] = np.hypot(coords[:, 0] - cent[k][0], coords[:, 1] - cent[k][1])
        sp["lat_sq"] = raw.latitude ** 2
        sp["lon_sq"] = raw.longitude ** 2
        sp["lat_lon"] = raw.latitude * raw.longitude
        blocks["spatial"] = sp

        # anomaly block
        an = pd.DataFrame(index=X.index)
        key = list(zip(clus, month))
        for c in ANOM_COLS:
            cl_m = self.normals_[c]
            normal = np.array([cl_m.get((k, m), np.nan) for k, m in key], dtype=float)
            fallback = month.map(self.month_means_[c]).values
            normal = np.where(np.isnan(normal), fallback, normal)
            an[f"{c}_anom"] = X[c] - normal
        mu = an[[f"{c}_anom" for c in ANOM_COLS]].abs().mean().mean()
        an["anom_mag"] = np.sqrt(sum(an[f"{c}_anom"] ** 2 for c in ANOM_COLS)) / np.sqrt(len(ANOM_COLS))
        blocks["anomaly"] = an

        # season block
        se = pd.DataFrame(index=X.index)
        for s, name in enumerate(["long_rains", "short_rains", "dec_feb", "jun_aug"]):
            se[f"season_{name}"] = (season == s).astype(int)
        blocks["season"] = se

        # interactions block
        it = pd.DataFrame(index=X.index)
        u5, inf = X.is_under5, X.is_infant
        it["u5_x_rain30"] = u5 * an.rain_sum_30d_anom
        it["u5_x_tmax30"] = u5 * an.tmax_30d_anom
        it["inf_x_tavg30"] = inf * an.tavg_30d_anom
        it["agelog_x_ndvi"] = X.age_log1p * X.ndvi_30d
        it["u5_x_dryness"] = u5 * X.dryness_idx
        it["elderly_x_tmax"] = X.is_elderly * X.tmax_30d
        it["u5_x_short_rains"] = u5 * se.season_short_rains
        it["u5_x_long_rains"] = u5 * se.season_long_rains
        blocks["interact"] = it

        out = pd.concat([X] + list(blocks.values()), axis=1)
        meta = {name: list(df.columns) for name, df in blocks.items()}
        return out, meta


def build_v2(tr_raw=None, te_raw=None):
    if tr_raw is None or te_raw is None:
        tr_raw, te_raw = load_data()
    fb = FeatureBuilderV2().fit(tr_raw, te_raw)
    Xtr, meta = fb.transform(tr_raw)
    Xte, _ = fb.transform(te_raw)
    return Xtr, Xte, meta
