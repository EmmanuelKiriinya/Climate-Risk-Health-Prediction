"""External features: WorldPop density + distances to cities / Lake Victoria."""
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, "scripts")
from features import load_data

# city / landmark coords (lat, lon)
CITIES = {
    "kampala": (0.3476, 32.5825),
    "jinja": (0.4478, 33.2026),
    "iganga": (0.6093, 33.4686),
    "mbale": (1.0827, 34.1755),
    "tororo": (0.6928, 34.1808),
    "entebbe": (0.0512, 32.4633),
    "mukono": (0.3533, 32.7553),
}
LAKE_POINTS = [(0.05, 32.45), (0.29, 32.63), (-0.05, 32.90), (-0.40, 33.15),
               (0.00, 33.30), (0.20, 32.50), (-0.55, 33.10)]


def hav(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main():
    tr, te = load_data()
    out = {}
    with rasterio.open("data_external/uga_ppp_2020_1km_UNadj.tif") as src:
        print("raster:", src.crs, src.shape)
        for name, df in [("train", tr), ("test", te)]:
            vals = np.array(list(src.sample(df[["longitude", "latitude"]].values.astype(float))))[:, 0]
            out[f"{name}_pop"] = np.clip(np.nan_to_num(vals.astype(float), nan=0.0), 0, None)

    rows = []
    for name, df in [("train", tr), ("test", te)]:
        f = pd.DataFrame({"ID": df.ID})
        f["pop"] = out[f"{name}_pop"]
        f["log_pop"] = np.log1p(f["pop"])
        for c, (la, lo) in CITIES.items():
            f[f"dist_{c}"] = hav(df.latitude.values, df.longitude.values, la, lo)
        f["dist_lake"] = np.min([hav(df.latitude.values, df.longitude.values, la, lo)
                                 for la, lo in LAKE_POINTS], axis=0)
        f["near_kampala_logpop"] = f.log_pop * np.exp(-f.dist_kampala / 50)
        rows.append(f)

    feat = pd.concat(rows, ignore_index=True)
    feat.to_csv("artifacts/external_features.csv", index=False)
    print(feat.describe().round(2).T[["mean", "min", "max"]].to_string())
    print("saved", len(feat), "rows")


if __name__ == "__main__":
    main()
