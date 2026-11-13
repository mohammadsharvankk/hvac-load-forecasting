"""Build the windowed dataset once, save it, and let every model load the identical arrays.

The cache needs only NumPy + json to load, so the TensorFlow environment
doesn't need pandas or the raw CSVs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path("artifacts/cache")


def cache_key(meter, buildings, lookback, horizon) -> str:
    b = "all" if not buildings else ",".join(map(str, sorted(buildings)))
    raw = f"m{meter}|b{b}|lb{lookback}|h{horizon}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


class TargetScaler:
    """Inverse transform for the target only (scaled log1p -> original units)."""

    def __init__(self, mean: float, std: float):
        self.mean, self.std = mean, std

    def inverse_target(self, y):
        return np.expm1(np.asarray(y) * self.std + self.mean)


def build_and_save(data_dir, meter, buildings, lookback, horizon) -> Path:
    from .data import FEATURE_COLS, build_datasets  # pandas only needed here

    sets, scaler = build_datasets(data_dir, meter, buildings, lookback, horizon)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(meter, buildings, lookback, horizon)
    path = CACHE_DIR / f"windows_{key}.npz"
    np.savez_compressed(path, **{f"X_{k}": v[0] for k, v in sets.items()},
                        **{f"y_{k}": v[1] for k, v in sets.items()})
    meta = {"meter": meter, "buildings": buildings, "lookback": lookback, "horizon": horizon,
            "features": FEATURE_COLS,
            "target_mean": float(scaler.mean["target"]), "target_std": float(scaler.std["target"])}
    (CACHE_DIR / f"windows_{key}.json").write_text(json.dumps(meta, indent=2))
    return path


def load(meter, buildings, lookback, horizon, data_dir=None):
    """Return ({'train': (X, y), 'val': ..., 'test': ...}, TargetScaler, meta)."""
    key = cache_key(meter, buildings, lookback, horizon)
    path = CACHE_DIR / f"windows_{key}.npz"
    if not path.exists():
        if data_dir is None:
            raise FileNotFoundError(f"{path} not found. Run src/prepare_data.py first with the same "
                                    f"--meter/--buildings/--lookback/--horizon.")
        print("cache miss: building windows from raw CSVs ...")
        build_and_save(data_dir, meter, buildings, lookback, horizon)
    z = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    sets = {k: (z[f"X_{k}"], z[f"y_{k}"]) for k in ("train", "val", "test")}
    return sets, TargetScaler(meta["target_mean"], meta["target_std"]), meta
