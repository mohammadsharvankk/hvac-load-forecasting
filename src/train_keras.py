"""TensorFlow/Keras LSTM: same architecture, data windows and split as the PyTorch model.

Needs only numpy + tensorflow, and reads the cache made by src/prepare_data.py.
    python src/train_keras.py --buildings 166 168 ... --target-mode delta
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from hvac.cache import load  # noqa: E402
from hvac.common import add_data_args, evaluate_and_save, from_prediction, to_training_target  # noqa: E402


def build_model(lookback, n_features, hidden=64, layers=2, dropout=0.2, lr=1e-3):
    """Mirrors hvac.models_torch.LSTMForecaster: stacked LSTM -> Dense(hidden/2, relu) -> Dense(1)."""
    inp = tf.keras.Input(shape=(lookback, n_features))
    x = inp
    for i in range(layers):
        last = i == layers - 1
        x = tf.keras.layers.LSTM(hidden, return_sequences=not last)(x)
        if not last and dropout:
            x = tf.keras.layers.Dropout(dropout)(x)  # PyTorch applies dropout between LSTM layers too
    x = tf.keras.layers.Dense(hidden // 2, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0), loss="mse")
    return model


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

    tf.keras.utils.set_random_seed(a.seed)
    name = f"lstm_keras_{a.target_mode}"
    out_dir = Path("artifacts/keras"); out_dir.mkdir(parents=True, exist_ok=True)

    sets, scaler, _ = load(a.meter, a.buildings, a.lookback, a.horizon)  # cache only: no pandas here
    (Xtr, ytr), (Xva, yva), (Xte, yte) = sets["train"], sets["val"], sets["test"]
    ttr = to_training_target(Xtr, ytr, a.target_mode)
    tva = to_training_target(Xva, yva, a.target_mode)
    print(f"{name}: train={len(Xtr):,} val={len(Xva):,} test={len(Xte):,} tf={tf.__version__}")

    model = build_model(Xtr.shape[1], Xtr.shape[2], a.hidden, a.layers, a.dropout, a.lr)
    ckpt = out_dir / f"{name}.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
    ]
    hist = model.fit(Xtr, ttr, validation_data=(Xva, tva), epochs=a.epochs,
                     batch_size=a.batch_size, shuffle=True, callbacks=callbacks, verbose=2)

    pred = from_prediction(Xte, model.predict(Xte, batch_size=2048, verbose=0).ravel(), a.target_mode)
    history = [{"epoch": i + 1, "train_loss": float(l), "val_loss": float(v)}
               for i, (l, v) in enumerate(zip(hist.history["loss"], hist.history["val_loss"]))]
    evaluate_and_save(name, Xte, yte, pred, scaler, {"config": vars(a), "history": history})


if __name__ == "__main__":
    main()
