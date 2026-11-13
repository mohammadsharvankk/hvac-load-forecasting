"""Shared CLI options, target modes and result saving, so every model is run the same way."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import regression_metrics

RESULTS_DIR = Path("artifacts/results")


def add_data_args(p: argparse.ArgumentParser):
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--meter", type=int, default=1, help="0 elec, 1 chilled water (cooling), 2 steam, 3 hot water")
    p.add_argument("--buildings", type=int, nargs="*", default=None)
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)


def to_training_target(X, y, mode: str):
    """level: predict scaled load directly. delta: predict change vs. last observed hour."""
    return y - X[:, -1, 0] if mode == "delta" else y


def from_prediction(X, pred, mode: str):
    return pred + X[:, -1, 0] if mode == "delta" else pred


def evaluate_and_save(name: str, X_test, y_test, pred_scaled, scaler, extra: dict | None = None,
                      include_naive: bool = True) -> dict:
    y_true = scaler.inverse_target(y_test)
    results = {name: regression_metrics(y_true, scaler.inverse_target(pred_scaled))}
    if include_naive:
        results["naive_last_hour"] = regression_metrics(y_true, scaler.inverse_target(X_test[:, -1, 0]))
        if X_test.shape[1] >= 24:
            results["naive_same_hour_yesterday"] = regression_metrics(
                y_true, scaler.inverse_target(X_test[:, -24, 0]))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps({"results": results, **(extra or {})}, indent=2))
    print("\nTest period results (original units):")
    for k, m in results.items():
        print(f"  {k:30s} RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  MAPE={m['mape_pct']:.1f}%")
    return results
