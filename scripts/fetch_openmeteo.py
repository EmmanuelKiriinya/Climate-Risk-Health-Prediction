"""Fetch Open-Meteo ERA5 daily series, per site-year chunks, threaded."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, "scripts")
from features import load_data

CACHE = Path("data_external/om_cache2")
CACHE.mkdir(parents=True, exist_ok=True)
URL = "https://archive-api.open-meteo.com/v1/archive"
FIELDS = ("temperature_2m_mean,relative_humidity_2m_mean,dew_point_2m_mean,"
          "et0_fao_evapotranspiration,wind_speed_10m_max,sunshine_duration,"
          "precipitation_hours,precipitation_sum")


def fetch_one(args):
    la, lo, year = args
    f = CACHE / f"{la}_{lo}_{year}.csv"
    if f.exists() and f.stat().st_size > 100:
        return f"{la},{lo},{year} cached"
    params = dict(latitude=la, longitude=lo,
                  start_date=f"{year}-01-01", end_date=f"{year}-12-31",
                  daily=FIELDS, timezone="GMT")
    for attempt in range(3):
        try:
            r = requests.get(URL, params=params, timeout=90)
            r.raise_for_status()
            d = r.json()["daily"]
            pd.DataFrame(d).to_csv(f, index=False)
            return f"{la},{lo},{year} ok"
        except Exception as e:
            err = str(e)[:80]
            time.sleep(3 * (attempt + 1))
    return f"{la},{lo},{year} FAIL {err}"


def main():
    tr, te = load_data()
    comb = pd.concat([tr[["latitude", "longitude"]], te[["latitude", "longitude"]]])
    sites = sorted(set(zip(comb.latitude.round(4), comb.longitude.round(4))))
    jobs = [(la, lo, y) for (la, lo) in sites for y in range(2006, 2024)]
    print(f"{len(sites)} sites, {len(jobs)} jobs")
    done = fail = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for fut in as_completed([ex.submit(fetch_one, j) for j in jobs]):
            msg = fut.result()
            if "FAIL" in msg:
                fail += 1
                print(msg)
            else:
                done += 1
            if (done + fail) % 50 == 0:
                print(f"progress {done+fail}/{len(jobs)}")
    print(f"DONE ok={done} fail={fail}")


if __name__ == "__main__":
    main()
