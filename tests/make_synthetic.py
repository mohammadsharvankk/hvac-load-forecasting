"""Write small ASHRAE-format CSVs for SMOKE TESTING ONLY.

These numbers are invented. Never report results from this data.
    python tests/make_synthetic.py --out data/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--buildings", type=int, default=3)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ts = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
    h, doy = ts.hour.to_numpy(), ts.dayofyear.to_numpy()
    temp = 28 + 8 * np.sin(2 * np.pi * (doy - 100) / 365) + 6 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 1, len(ts))

    pd.DataFrame({"site_id": 0, "timestamp": ts, "air_temperature": temp,
                  "cloud_coverage": np.nan, "dew_temperature": temp - 8 + rng.normal(0, 1, len(ts)),
                  "precip_depth_1_hr": 0, "sea_level_pressure": 1010 + rng.normal(0, 3, len(ts)),
                  "wind_direction": 180, "wind_speed": np.abs(rng.normal(3, 1, len(ts)))}
                 ).to_csv(out / "weather_train.csv", index=False)
    pd.DataFrame({"site_id": 0, "building_id": range(a.buildings), "primary_use": "Office",
                  "square_feet": 50000, "year_built": 2000, "floor_count": 5}
                 ).to_csv(out / "building_metadata.csv", index=False)

    rows = []
    for b in range(a.buildings):
        occ = ((h >= 8) & (h <= 18) & (ts.dayofweek < 5)).astype(float)
        load = np.clip(50 * (b + 1) + 6 * (temp - 22) + 40 * occ + rng.normal(0, 5, len(ts)), 0, None)
        load[1000:1050] = 0  # simulated meter outage
        rows.append(pd.DataFrame({"building_id": b, "meter": 1, "timestamp": ts, "meter_reading": load}))
    pd.concat(rows).to_csv(out / "train.csv", index=False)
    print(f"synthetic data written to {out}")


if __name__ == "__main__":
    main()
