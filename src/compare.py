"""Collect every artifacts/results/*.json into one Markdown table, best RMSE first.

    python src/compare.py
"""
import json
from pathlib import Path

rows = {}
for f in sorted(Path("artifacts/results").glob("*.json")):
    for name, m in json.loads(f.read_text())["results"].items():
        rows.setdefault(name, m)  # naive baselines appear in several files; identical, keep one

if not rows:
    raise SystemExit("No results yet. Run a training script first.")

ref = rows.get("naive_last_hour")
lines = ["| Model | RMSE | MAE | MAPE | RMSE vs last-hour |", "|---|---|---|---|---|"]
for name, m in sorted(rows.items(), key=lambda kv: kv[1]["rmse"]):
    gain = f"{(1 - m['rmse'] / ref['rmse']) * 100:+.1f}%" if ref and name != "naive_last_hour" else "–"
    lines.append(f"| {name} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['mape_pct']:.1f}% | {gain} |")
table = "\n".join(lines)
print(table)
Path("artifacts/results/comparison.md").write_text(table + "\n")
