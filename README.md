# HVAC Cooling-Load Forecasting

Forecasting next-hour building cooling load (chilled-water meter) from past load and weather, using a **PyTorch LSTM**, benchmarked against naive and tree-based baselines. A TensorFlow/Keras version, PySpark preprocessing, a FastAPI service and drift monitoring are added in later stages.

> Status: **Stage 1 — data pipeline + PyTorch LSTM.** Results below get filled in from real runs only.

## Why this setup
- **Real data:** ASHRAE Great Energy Predictor III — about 1,400 buildings, hourly meter readings for 2016, plus site weather.
- **Target:** meter 1 (chilled water), i.e. how much cooling the building uses. This is the closest public proxy for climate-control demand.
- **Chronological split:** train Jan–Aug, validate Sep–Oct, test Nov–Dec. Random splits leak future information into training for time series, so they're not used here.
- **Baselines first:** "same as last hour" and "same hour yesterday" are hard to beat for hourly load. A model only counts as useful if it beats them.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Download the competition data from Kaggle (https://www.kaggle.com/c/ashrae-energy-prediction). Put `train.csv`, `building_metadata.csv` and `weather_train.csv` in `data/raw/`. Via the Kaggle CLI:
```bash
kaggle competitions download -c ashrae-energy-prediction -p data/raw
```
You need to accept the competition rules on Kaggle once before the download works.

## Run
```bash
# Smoke test on synthetic data (code check only — do not report these numbers)
python tests/make_synthetic.py --out data/synthetic
python src/train_torch.py --data-dir data/synthetic --epochs 3

# Real run: a handful of buildings first, then scale up
python src/train_torch.py --data-dir data/raw --buildings 0 1 2 3 4 5 6 7 8 9 --epochs 30
```
Outputs go to `artifacts/torch/`: the best model weights (`lstm_best.pt`) and `metrics.json`, which holds the config, per-epoch loss history and test metrics.

## Project structure
```
src/hvac/data.py           load, clean (outage zeros, short-gap interpolation), features, time split, windows
src/hvac/models_torch.py   2-layer LSTM
src/hvac/metrics.py        RMSE / MAE / MAPE in original meter units
src/train_torch.py         training loop, early stopping, evaluation vs baselines
tests/make_synthetic.py    ASHRAE-format fake data for smoke tests
```

## Roadmap
- [x] Stage 1 — Data pipeline + PyTorch LSTM + naive baselines
- [ ] Stage 2 — TensorFlow/Keras LSTM (same windows, same split) + Random Forest / XGBoost on lag features
- [ ] Stage 3 — Clustering of daily load profiles (k-means) to group building behaviour
- [ ] Stage 4 — PySpark version of the preprocessing for the full dataset
- [ ] Stage 5 — FastAPI `/predict` endpoint in Docker
- [ ] Stage 6 — Drift monitoring (PSI / KS test on inputs, rolling error on predictions)

## Results
10 buildings (IDs 166–192, chilled-water meter), test period Nov–Dec 2016, original meter units.

| Model | RMSE | MAE | MAPE |  |---|---|---|---|
| Naive (last hour) | 70.89 | 30.00 | 25.4% |
| Naive (same hour yesterday) | 103.89 | 48.70 | 42.5% |
| LSTM (PyTorch) | **63.26** | **29.17** | **21.9%** |