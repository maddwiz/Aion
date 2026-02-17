#!/usr/bin/env python3
# Build runs_plus/asset_names.csv from data/*.csv and data_new/*.csv symbols.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [ROOT / "data", ROOT / "data_new"]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)

if __name__ == "__main__":
    by_sym = {}
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.csv")):
            if not p.is_file():
                continue
            sym = p.stem.replace("_prices", "").upper().strip()
            if sym:
                by_sym[sym] = str(p)
    syms = sorted(by_sym.keys())
    pd.DataFrame({"asset": syms}).to_csv(RUNS / "asset_names.csv", index=False)
    print(f"✅ Wrote {RUNS/'asset_names.csv'} ({len(syms)} symbols)")
