#!/usr/bin/env python3
# Build runs_plus/asset_names.csv from data/*.csv symbols.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)

if __name__ == "__main__":
    syms = sorted([p.stem.replace("_prices", "").upper() for p in DATA.glob("*.csv") if p.is_file()])
    pd.DataFrame({"asset": syms}).to_csv(RUNS / "asset_names.csv", index=False)
    print(f"✅ Wrote {RUNS/'asset_names.csv'} ({len(syms)} symbols)")
