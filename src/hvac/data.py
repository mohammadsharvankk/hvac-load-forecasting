"""Data loading, cleaning and windowing for ASHRAE Great Energy Predictor III.

Dataset: https://www.kaggle.com/c/ashrae-energy-prediction (download to data/raw/)
Files used: train.csv, building_metadata.csv, weather_train.csv

Default target is meter 1 (chilled water) = the building's cooling load,
which is the closest proxy for HVAC / climate-control demand.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}

WEATHER_COLS = ["air_temperature", "dew_temperature", "cloud_coverage",
                "wind_speed", "sea_level_pressure"]
TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
FEATURE_COLS = ["target"] + WEATHER_COLS + TIME_COLS  # target (lagged) is feature 0

# Chronological split: no shuffling, no future leakage.
TRAIN_END = "2016-08-31 23:00"
VAL_END = "2016-10-31 23:00"


def load_raw(data_dir: str | Path, meter: int = 1,
             building_ids: list[int] | None = None) -> pd.DataFrame:
    """Merge meter readings with building metadata and site weather."""
    data_dir = Path(data_dir)
    meters = pd.read_csv(data_dir / "train.csv", parse_dates=["timestamp"])
    meters = meters[meters["meter"] == meter]
    if building_ids is not None:
        meters = meters[meters["building_id"].isin(building_ids)]
    if meters.empty:
        raise ValueError(f"No readings for meter={meter} and buildings={building_ids}")

    buildings = pd.read_csv(data_dir / "building_metadata.csv")
    weather = pd.read_csv(data_dir / "weather_train.csv", parse_dates=["timestamp"])

    df = meters.merge(buildings[["building_id", "site_id"]], on="building_id", how="left")
    df = df.merge(weather[["site_id", "timestamp"] + WEATHER_COLS],
                  on=["site_id", "timestamp"], how="left")
    return df.rename(columns={"meter_reading": "target"})


def clean_building(g: pd.DataFrame) -> pd.DataFrame:
    """Clean one building's series onto a regular hourly grid."""
    g = g.set_index("timestamp").sort_index()
    g = g[~g.index.duplicated(keep="first")]
    full = pd.date_range(g.index.min(), g.index.max(), freq="h")
    g = g.reindex(full)
    g.index.name = "timestamp"

    # Long runs of exact zeros are meter outages, not real zero load.
    zero = g["target"].eq(0)
    run_id = (zero != zero.shift()).cumsum()
    run_len = zero.groupby(run_id).transform("sum")
    g.loc[zero & (run_len >= 24), "target"] = np.nan

    # Short sensor dropouts: interpolate up to 3 hours; longer gaps stay NaN
    # and are skipped when windows are built.
    g["target"] = g["target"].interpolate(limit=3, limit_area="inside")
    g[WEATHER_COLS] = g[WEATHER_COLS].interpolate(limit=6, limit_area="inside")
    # Cloud cover / pressure are often entirely missing for a site.
    for c in ["cloud_coverage", "sea_level_pressure"]:
        if g[c].isna().all():
            g[c] = 0.0

    h, d = g.index.hour, g.index.dayofweek
    g["hour_sin"], g["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    g["dow_sin"], g["dow_cos"] = np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)
    g["is_weekend"] = (d >= 5).astype(float)
    g["building_id"] = g["building_id"].ffill().bfill()
    return g.reset_index()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    parts = [clean_building(g) for _, g in df.groupby("building_id")]
    out = pd.concat(parts, ignore_index=True)
    out["target"] = np.log1p(out["target"].clip(lower=0))  # heavy right tail
    return out


@dataclass
class Scaler:
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Scaler":
        x = df[FEATURE_COLS]
        return cls(x.mean(), x.std().replace(0, 1.0).fillna(1.0))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[FEATURE_COLS] = (df[FEATURE_COLS] - self.mean) / self.std
        return df

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        """Scaled log target -> original meter units."""
        return np.expm1(y * self.std["target"] + self.mean["target"])


def split_by_time(df: pd.DataFrame):
    ts = df["timestamp"]
    return (df[ts <= TRAIN_END],
            df[(ts > TRAIN_END) & (ts <= VAL_END)],
            df[ts > VAL_END])


def make_windows(df: pd.DataFrame, lookback: int = 24, horizon: int = 1):
    """Sliding windows per building. X: past `lookback` hours of all features,
    y: target `horizon` hours after the window. Windows with any NaN are dropped,
    and windows never cross buildings."""
    X_all, y_all = [], []
    for _, g in df.groupby("building_id"):
        arr = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        n = len(arr) - lookback - horizon + 1
        if n <= 0:
            continue
        idx = np.arange(lookback)[None, :] + np.arange(n)[:, None]
        X = arr[idx]                                  # (n, lookback, F)
        y = arr[lookback + horizon - 1:, 0][:n]       # (n,)
        ok = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y)
        X_all.append(X[ok]); y_all.append(y[ok])
    if not X_all:
        return np.empty((0, lookback, len(FEATURE_COLS)), np.float32), np.empty(0, np.float32)
    return np.concatenate(X_all), np.concatenate(y_all)


def build_datasets(data_dir, meter=1, building_ids=None, lookback=24, horizon=1):
    """End-to-end: raw CSVs -> scaled train/val/test windows + fitted scaler."""
    df = prepare(load_raw(data_dir, meter, building_ids))
    train, val, test = split_by_time(df)
    scaler = Scaler.fit(train)  # fit on training period only
    sets = {name: make_windows(scaler.transform(part), lookback, horizon)
            for name, part in [("train", train), ("val", val), ("test", test)]}
    return sets, scaler
