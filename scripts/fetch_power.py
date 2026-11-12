"""Fetch NASA POWER daily series (2006-12..2023-01) for every unique site."""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, "scripts")
from features import load_data

CACHE = Path("data_external/power_cache")
CACHE.mkdir(parents=True, exist_ok=True)
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "RH2M,QV2M,T2M,WS2M,WS2M_MAX,ALLSKY_SFC_SW_DWN,PRECTOTCORR,T2M_RANGE"


def main():
    tr, te = load_data()
    comb = pd.concat([tr[["latitude", "longitude"]], te[["latitude", "longitude"]]])
    sites = sorted(set(zip(comb.latitude.round(4), comb.longitude.round(4))))
    print(len(sites), "sites")
    fails = []
    for i, (la, lo) in enumerate(sites):
        f = CACHE / f"{la}_{lo}.csv"
        if f.exists() and f.stat().st_size > 5000:
            continue
        p = dict(parameters=PARAMS, community="AG", latitude=la, longitude=lo,
                 start="20061201", end="20230131", format="JSON")
        ok = False
        for attempt in range(5):
            try:
                r = requests.get(URL, params=p, timeout=180)
                r.raise_for_status()
                d = r.json()["properties"]["parameter"]
                df = pd.DataFrame(d)
                df.index.name = "date"
                df.to_csv(f)
                print(f"[{i+1}/{len(sites)}] {la},{lo} rows={len(df)}")
                ok = True
                break
            except Exception as e:
                wait = 20 * (attempt + 1)
                print(f"retry {la},{lo} in {wait}s: {str(e)[:60]}")
                time.sleep(wait)
        if not ok:
            fails.append((la, lo))
        time.sleep(1.0)
    print("FAILED:", fails)


if __name__ == "__main__":
    main()
