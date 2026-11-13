"""Train the PyTorch LSTM and evaluate against naive baselines on a held-out time period.

Example:
    python src/train_torch.py --data-dir data/raw --buildings 0 1 2 3 4 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from hvac.data import FEATURE_COLS, build_datasets  # noqa: E402
from hvac.metrics import regression_metrics  # noqa: E402
from hvac.models_torch import LSTMForecaster  # noqa: E402


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def predict(model, X, device, batch=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch]).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--meter", type=int, default=1, help="0 elec, 1 chilled water (cooling), 2 steam, 3 hot water")
    p.add_argument("--buildings", type=int, nargs="*", default=None, help="building_ids; default = all for this meter")
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="artifacts/torch")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sets, scaler = build_datasets(args.data_dir, args.meter, args.buildings, args.lookback, args.horizon)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = sets["train"], sets["val"], sets["test"]
    print(f"windows  train={len(Xtr):,}  val={len(Xva):,}  test={len(Xte):,}  device={device}")

    model = LSTMForecaster(len(FEATURE_COLS), args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=args.batch_size, shuffle=True)  # shuffling windows within train period is fine

    best, bad, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(Xtr)
        val_loss = float(np.mean((predict(model, Xva, device) - yva) ** 2))
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"epoch {epoch:3d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}  ({time.time() - t0:.1f}s)")
        if val_loss < best - 1e-5:
            best, bad = val_loss, 0
            torch.save(model.state_dict(), out_dir / "lstm_best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping"); break

    model.load_state_dict(torch.load(out_dir / "lstm_best.pt", map_location=device))

    # Evaluate on the untouched test period, in original meter units.
    y_true = scaler.inverse_target(yte)
    results = {
        "lstm_pytorch": regression_metrics(y_true, scaler.inverse_target(predict(model, Xte, device))),
        # Persistence: next hour = last observed hour.
        "naive_last_hour": regression_metrics(y_true, scaler.inverse_target(Xte[:, -1, 0])),
    }
    if args.lookback >= 24 and args.horizon == 1:
        # Seasonal naive: same hour yesterday.
        results["naive_same_hour_yesterday"] = regression_metrics(
            y_true, scaler.inverse_target(Xte[:, -24, 0]))

    report = {"config": vars(args), "results": results, "history": history}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print("\nTest period results (original units):")
    for name, m in results.items():
        print(f"  {name:28s} RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  MAPE={m['mape_pct']:.1f}%")


if __name__ == "__main__":
    main()
