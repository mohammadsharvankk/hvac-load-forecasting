import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics in original meter units. MAPE skips near-zero loads, where it explodes."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    mask = np.abs(y_true) > max(1e-6, 0.05 * np.median(np.abs(y_true)))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": float(np.mean(np.abs(err[mask] / y_true[mask])) * 100) if mask.any() else float("nan"),
        "n": int(len(y_true)),
    }
