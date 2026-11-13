# HVAC Cooling-Load Forecasting

Forecasting next-hour building cooling load (chilled-water meter) from past load and weather. The same data windows feed a **PyTorch LSTM**, a **TensorFlow/Keras LSTM**, **Random Forest** and **XGBoost**, all benchmarked against naive baselines. PySpark preprocessing, a FastAPI service and drift monitoring are added in later stages.

> Status: **Stage 2 — multi-framework model comparison.** Results come from real runs only.

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
B="166 168 169 170 173 174 178 181 183 192"   # PowerShell: just paste the IDs after --buildings

# 1. Build the windows once; every model reads the same cached arrays
python src/prepare_data.py --data-dir data/raw --buildings $B

# 2. Train
python src/train_torch.py --buildings $B --target-mode level
python src/train_torch.py --buildings $B --target-mode delta
python src/train_trees.py --buildings $B
python src/train_keras.py --buildings $B --target-mode delta   # in the TensorFlow env, see below

# 3. One comparison table
python src/compare.py
```
`--target-mode delta` makes a model predict the change from the last observed hour instead of the raw load.

**TensorFlow environment:** TensorFlow has no stable Windows wheel for Python 3.14 yet, so the Keras model runs in its own environment on Python 3.12 or 3.13:
```bash
py -3.13 -m venv .venv-tf
.venv-tf\Scripts\activate
pip install -r requirements-tf.txt
```

**Smoke test** on synthetic data. This only checks the code runs, so never report these numbers:
```bash
python tests/make_synthetic.py --out data/synthetic
python src/prepare_data.py --data-dir data/synthetic
python src/train_torch.py --epochs 3
```
Outputs: cached windows in `artifacts/cache/`, per-model metrics in `artifacts/results/<model>.json`, the comparison table in `artifacts/results/comparison.md`.

## Project structure
```
src/hvac/data.py           load, clean (outage zeros, short-gap interpolation), features, time split, windows
src/hvac/cache.py          build windows once, save/load as .npz so every model sees identical data
src/hvac/common.py         shared CLI options, level/delta targets, evaluation + result saving
src/hvac/models_torch.py   2-layer LSTM (PyTorch)
src/hvac/metrics.py        RMSE / MAE / MAPE in original meter units
src/prepare_data.py        raw CSVs -> cached windows
src/train_torch.py         PyTorch LSTM
src/train_keras.py         TensorFlow/Keras LSTM, same architecture
src/train_trees.py         Random Forest + XGBoost on lag features from the same windows
src/compare.py             all results -> one Markdown table
tests/make_synthetic.py    ASHRAE-format fake data for smoke tests
```

## Roadmap
- [x] Stage 1 — Data pipeline + PyTorch LSTM + naive baselines
- [x] Stage 2 — TensorFlow/Keras LSTM (same windows, same split) + Random Forest / XGBoost on lag features
- [ ] Stage 3 — Clustering of daily load profiles (k-means) to group building behaviour
- [ ] Stage 4 — PySpark version of the preprocessing for the full dataset
- [ ] Stage 5 — FastAPI `/predict` endpoint in Docker
- [ ] Stage 6 — Drift monitoring (PSI / KS test on inputs, rolling error on predictions)

## Results
10 buildings (IDs 166–192, chilled-water meter), test period Nov–Dec 2016, original meter units. Every model uses the same cached windows and the same chronological split.

| Model | RMSE | MAE | MAPE | RMSE vs last-hour |
|---|---|---|---|---|
| Random Forest | **62.48** | **25.97** | 22.1% | +11.9% |
| LSTM PyTorch (delta) | 62.87 | 27.55 | **21.6%** | +11.3% |
| LSTM PyTorch (level) | 63.26 | 29.17 | 21.9% | +10.8% |
| XGBoost | 64.35 | 29.13 | 21.6% | +9.2% |
| LSTM Keras (delta) | 64.47 | 27.92 | 22.7% | +9.1% |
| Naive (last hour) | 70.89 | 30.00 | 25.4% | – |
| Naive (same hour yesterday) | 103.89 | 48.70 | 42.5% | -46.5% |

**Takeaways**
- Random Forest on lag features matches the LSTMs and trains in under a minute, against ~15 minutes for the LSTM on CPU.
- Predicting the change from the last hour (delta) cut LSTM MAE by 6% compared with predicting the load directly.
- The PyTorch and Keras LSTMs share an architecture but differ slightly, due to framework defaults (initialisation, early-stopping point).
- MAPE stays around 22% for every model because the test period is low-load winter, while training was mostly summer. That seasonal shift is the motivation for drift monitoring (Stage 6).
- Runs are reproducible: a fixed seed gives identical PyTorch results run to run.