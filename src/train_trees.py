"""Tree baselines (Random Forest, XGBoost) on lag features taken from the SAME windows as the LSTMs.

    python src/train_trees.py --buildings 166 168 ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hvac.cache import load  # noqa: E402
from hvac.common import add_data_args, evaluate_and_save  # noqa: E402

LAGS = [1, 2, 3, 6, 12, 24]


def lag_features(X: np.ndarray, feature_names: list[str]):
    """Turn (n, lookback, F) windows into a flat table a tree model can use."""
    lb = X.shape[1]
    lags = [k for k in LAGS if k <= lb]
    target = X[:, :, 0]
    cols, names = [], []
    for k in lags:
        cols.append(target[:, -k]); names.append(f"load_lag{k}")
    cols += [target.mean(1), target.std(1), target.min(1), target.max(1),
             target[:, -1] - target[:, -2] if lb >= 2 else np.zeros(len(X))]
    names += ["load_mean_win", "load_std_win", "load_min_win", "load_max_win", "load_diff1"]
    for j, f in enumerate(feature_names[1:], start=1):  # weather + time at the latest hour
        cols.append(X[:, -1, j]); names.append(f"{f}_now")
    for j, f in enumerate(feature_names[1:], start=1):
        if f in ("air_temperature", "dew_temperature"):
            cols.append(X[:, :, j].mean(1)); names.append(f"{f}_mean_win")
    return np.column_stack(cols).astype(np.float32), names


def main():
    p = argparse.ArgumentParser()
    add_data_args(p)
    p.add_argument("--models", nargs="*", default=["random_forest", "xgboost"])
    a = p.parse_args()

    sets, scaler, meta = load(a.meter, a.buildings, a.lookback, a.horizon, a.data_dir)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = sets["train"], sets["val"], sets["test"]
    Ftr, names = lag_features(Xtr, meta["features"])
    Fva, _ = lag_features(Xva, meta["features"])
    Fte, _ = lag_features(Xte, meta["features"])
    print(f"tree features: {len(names)}  train={len(Ftr):,}")

    if "random_forest" in a.models:
        from sklearn.ensemble import RandomForestRegressor
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                   n_jobs=-1, random_state=a.seed)
        rf.fit(Ftr, ytr)
        print(f"random forest trained in {time.time() - t0:.1f}s")
        imp = sorted(zip(names, rf.feature_importances_), key=lambda t: -t[1])[:10]
        evaluate_and_save("random_forest", Xte, yte, rf.predict(Fte), scaler,
                          {"top_features": [[n, float(v)] for n, v in imp]}, include_naive=False)

    if "xgboost" in a.models:
        import xgboost as xgb
        t0 = time.time()
        model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, early_stopping_rounds=100, random_state=a.seed,
                                 n_jobs=-1, tree_method="hist")
        model.fit(Ftr, ytr, eval_set=[(Fva, yva)], verbose=False)  # early-stopped on Sep–Oct only
        print(f"xgboost trained in {time.time() - t0:.1f}s, best iteration {model.best_iteration}")
        imp = sorted(zip(names, model.feature_importances_), key=lambda t: -t[1])[:10]
        evaluate_and_save("xgboost", Xte, yte, model.predict(Fte), scaler,
                          {"best_iteration": int(model.best_iteration),
                           "top_features": [[n, float(v)] for n, v in imp]}, include_naive=False)


if __name__ == "__main__":
    main()
