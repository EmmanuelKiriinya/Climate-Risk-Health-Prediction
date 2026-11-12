import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
tr = pd.read_csv("Data/Train.csv", parse_dates=["deathdate"])
te = pd.read_csv("Data/Test.csv", parse_dates=["deathdate"])
cl = pd.read_csv("Data/climate_features.csv")

df = tr.merge(cl.drop(columns=["deathdate"]), on="ID", how="left")
print("=== shapes ===")
print("train", tr.shape, "test", te.shape, "climate", cl.shape)
print("\n=== target by zone / gender ===")
print(df.groupby("zone")["is_climate_sensitive"].agg(["mean", "count"]))
print(df.groupby("gender")["is_climate_sensitive"].agg(["mean", "count"]))

print("\n=== age vs target ===")
print(df.groupby("is_climate_sensitive")["age"].describe().round(2))

print("\n=== year coverage train/test ===")
print("train:", tr.deathdate.dt.year.min(), "-", tr.deathdate.dt.year.max())
print("test :", te.deathdate.dt.year.min(), "-", te.deathdate.dt.year.max())
print("\ntrain deaths/yr:")
print(tr.deathdate.dt.year.value_counts().sort_index().to_string())
print("\ntest deaths/yr:")
print(te.deathdate.dt.year.value_counts().sort_index().to_string())

print("\n=== district extraction (train) ===")
tr["district"] = tr.location.str.split(",").str[-2].str.strip()
te["district"] = te.location.str.split(",").str[-2].str.strip()
dtr = tr.groupby("district").agg(n=("ID", "count"), rate=("is_climate_sensitive", "mean")).sort_values("n", ascending=False)
print(dtr.to_string())
print("\ndistricts in test not in train:", set(te.district) - set(tr.district))
print("districts in train not in test:", set(tr.district) - set(te.district))

num = ["age", "avg_temperature", "max_temperature", "min_temperature", "precipitation",
       "latitude", "longitude", "rain_sum_7d", "rain_sum_30d", "rain_sum_90d",
       "rain_days_30d", "max_daily_rain_30d", "tavg_7d", "tavg_30d", "tavg_90d",
       "tmax_30d", "tmin_30d", "hot_days_30d", "temp_range_mean_30d", "ndvi_30d",
       "ndvi_90d", "elevation", "slope"]
print("\n=== point-biserial corr with target ===")
corr = df[num + ["is_climate_sensitive"]].corr()["is_climate_sensitive"].drop("is_climate_sensitive").sort_values(key=abs, ascending=False)
print(corr.round(3).to_string())

print("\n=== climate features: missing in merged ===")
print(cl.isna().sum()[lambda s: s > 0])
print("climate rows not in train+test:", (~cl.ID.isin(set(tr.ID) | set(te.ID))).sum())

# quick univariate AUC for each numeric feature
from sklearn.metrics import roc_auc_score
y = df.is_climate_sensitive
print("\n=== univariate ROC-AUC (abs) ===")
aucs = {}
for c in num:
    v = df[c]
    aucs[c] = max(roc_auc_score(y, v.fillna(v.median())), 1 - roc_auc_score(y, v.fillna(v.median())))
for k, v in sorted(aucs.items(), key=lambda kv: -kv[1]):
    print(f"{k:28s} {v:.4f}")

# month seasonality
print("\n=== target rate by month ===")
print(df.groupby(df.deathdate.dt.month)["is_climate_sensitive"].agg(["mean", "count"]).round(3).to_string())

# ID leakage check
print("\n=== duplicate coordinates ===")
print("train dup lat/lon pairs:", tr.duplicated(subset=["latitude", "longitude"]).sum())
print("exact feature dupes across train:", tr.drop(columns=["ID", "deathdate"]).duplicated().sum())
