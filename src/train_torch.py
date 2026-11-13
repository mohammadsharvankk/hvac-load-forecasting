"""PyTorch LSTM for next-hour cooling load.

    python src/train_torch.py --buildings 166 168 ... --target-mode delta
Results: artifacts/results/lstm_pytorch_<mode>.json, weights in artifacts/torch/.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from hvac.cache import load  # noqa: E402
from hvac.common import add_data_args, evaluate_and_save, from_prediction, to_training_target  # noqa: E402
from hvac.models_torch import LSTMForecaster  # noqa: E402


def predict(model, X, device, batch=2048):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch]).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def main():
    p = argparse.ArgumentParser()
    add_data_args(p)
    p.add_argument("--target-mode", choices=["level", "delta"], default="level")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("artifacts/torch"); out_dir.mkdir(parents=True, exist_ok=True)
    name = f"lstm_pytorch_{a.target_mode}"

    sets, scaler, meta = load(a.meter, a.buildings, a.lookback, a.horizon, a.data_dir)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = sets["train"], sets["val"], sets["test"]
    ttr = to_training_target(Xtr, ytr, a.target_mode).astype(np.float32)
    print(f"{name}: train={len(Xtr):,} val={len(Xva):,} test={len(Xte):,} device={device}")

    model = LSTMForecaster(Xtr.shape[2], a.hidden, a.layers, a.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ttr)),
                        batch_size=a.batch_size, shuffle=True)

    best, bad, history, ckpt = float("inf"), 0, [], out_dir / f"{name}.pt"
    for epoch in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        # Validation measured on the load level in both modes, so epochs are comparable.
        val_pred = from_prediction(Xva, predict(model, Xva, device), a.target_mode)
        val_mse = float(np.mean((val_pred - yva) ** 2))
        sched.step(val_mse)
        history.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_mse": val_mse})
        print(f"epoch {epoch:3d}  train_loss={total / len(Xtr):.4f}  val_mse={val_mse:.4f}  ({time.time() - t0:.1f}s)")
        if val_mse < best - 1e-5:
            best, bad = val_mse, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    pred = from_prediction(Xte, predict(model, Xte, device), a.target_mode)
    evaluate_and_save(name, Xte, yte, pred, scaler, {"config": vars(a), "history": history})


if __name__ == "__main__":
    main()
