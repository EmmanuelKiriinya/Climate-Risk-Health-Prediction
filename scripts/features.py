"""Shared feature engineering for the Climate Risk & Health challenge."""
import numpy as np
import pandas as pd

CLIMATE_COLS = [
    "rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "rain_days_30d",
    "max_daily_rain_30d", "tavg_7d", "tavg_30d", "tavg_90d", "tmax_30d",
    "tmin_30d", "hot_days_30d", "temp_range_mean_30d", "ndvi_30d", "ndvi_90d",
    "elevation", "slope",
]

BASE_COLS = [
    "age", "gender", "zone",
    "avg_temperature", "max_temperature", "min_temperature", "precipitation",
    "latitude", "longitude",
]


def load_data(data_dir="Data"):
    tr = pd.read_csv(f"{data_dir}/Train.csv", parse_dates=["deathdate"])
    te = pd.read_csv(f"{data_dir}/Test.csv", parse_dates=["deathdate"])
    cl = pd.read_csv(f"{data_dir}/climate_features.csv").drop(columns=["deathdate"])
    tr = tr.merge(cl, on="ID", how="left")
    te = te.merge(cl, on="ID", how="left")
    return tr, te


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # date / seasonality
    out["year"] = out.deathdate.dt.year
    month = out.deathdate.dt.month
    doy = out.deathdate.dt.dayofyear
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # age transforms (monotonic helpers)
    out["age_log1p"] = np.log1p(out["age"])
    out["is_infant"] = (out["age"] <= 1).astype(int)
    out["is_under5"] = (out["age"] < 5).astype(int)
    out["is_child"] = ((out["age"] >= 5) & (out["age"] < 18)).astype(int)
    out["is_elderly"] = (out["age"] >= 60).astype(int)

    # categorical encodings
    out["gender_bin"] = (out["gender"] == "Female").astype(int)
    out["zone_bin"] = (out["zone"] == "Rural").astype(int)

    # climate deltas / ratios
    eps = 1e-6
    out["t_anom_7d_30d"] = out["tavg_7d"] - out["tavg_30d"]
    out["t_anom_30d_90d"] = out["tavg_30d"] - out["tavg_90d"]
    out["tx_tavg_gap"] = out["tmax_30d"] - out["tavg_30d"]
    out["rain_share_7d_30d"] = out["rain_sum_7d"] / (out["rain_sum_30d"] + eps)
    out["rain_share_30d_90d"] = out["rain_sum_30d"] / (out["rain_sum_90d"] + eps)
    out["dryness_idx"] = out["rain_sum_30d"] / (out["tavg_30d"] + eps)
    out["ndvi_delta"] = out["ndvi_30d"] - out["ndvi_90d"]

    # observed-weather vs reanalysis agreement
    out["tavg_obs_minus_rean"] = out["avg_temperature"] - out["tavg_30d"]
    out["precip_obs_vs_rain30"] = out["precipitation"] / (out["rain_sum_30d"] + eps) * 30

    feature_cols = (
        BASE_COLS[:1]                      # age
        + ["gender_bin", "zone_bin", "year",
           "month_sin", "month_cos", "doy_sin", "doy_cos",
           "age_log1p", "is_infant", "is_under5", "is_child", "is_elderly"]
        + BASE_COLS[4:]                    # temps, precip, lat, lon
        + CLIMATE_COLS
        + ["t_anom_7d_30d", "t_anom_30d_90d", "tx_tavg_gap",
           "rain_share_7d_30d", "rain_share_30d_90d", "dryness_idx",
           "ndvi_delta", "tavg_obs_minus_rean", "precip_obs_vs_rain30"]
    )
    return out[feature_cols]


def make_submission(ids, proba, path="submissions/submission.csv"):
    pred = (np.asarray(proba) >= 0.5).astype(int)
    sub = pd.DataFrame({"ID": ids, "TargetF1": pred, "TargetRAUC": np.asarray(proba, dtype=float)})
    sub.to_csv(path, index=False)
    return sub
