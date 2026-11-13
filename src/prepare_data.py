"""Build windows from the raw CSVs once and cache them for all models.

    python src/prepare_data.py --data-dir data/raw --buildings 166 168 169 170 173 174 178 181 183 192
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hvac.cache import build_and_save, load  # noqa: E402
from hvac.common import add_data_args  # noqa: E402

p = argparse.ArgumentParser()
add_data_args(p)
a = p.parse_args()
path = build_and_save(a.data_dir, a.meter, a.buildings, a.lookback, a.horizon)
sets, _, _ = load(a.meter, a.buildings, a.lookback, a.horizon)
print(f"saved {path}")
for k, (X, y) in sets.items():
    print(f"  {k:5s} X={X.shape}  y={y.shape}")
