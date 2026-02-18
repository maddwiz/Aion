#!/usr/bin/env python3
"""
Build external untouched holdout files from q/data source CSVs.

Writes:
  - q/data_holdout/<SYMBOL>.csv (Date,Close)
  - runs_plus/external_holdout_build_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _read_asset_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    col = df.columns[0]
    vals = [str(x).strip().upper() for x in df[col].tolist() if str(x).strip()]
    return vals


def _read_price(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    dcol = None
    for c in ["DATE", "Date", "date", "timestamp", "Timestamp"]:
        if c in df.columns:
            dcol = c
            break
    if dcol is None:
        return None

    ccol = None
    for c in ["Close", "Adj Close", "close", "adj_close", "value", "Value", "PRICE", "price"]:
        if c in df.columns:
            ccol = c
            break
    if ccol is None:
        return None

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[dcol], errors="coerce"),
            "Close": pd.to_numeric(df[ccol], errors="coerce"),
        }
    ).dropna()
    if out.empty:
        return None
    out = out.sort_values("Date")
    out = out.drop_duplicates(subset=["Date"], keep="last")
    return out


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def main() -> int:
    holdout_dir = Path(str(os.getenv("Q_EXTERNAL_HOLDOUT_DIR", str(ROOT / "data_holdout"))).strip())
    source_dir = Path(str(os.getenv("Q_EXTERNAL_HOLDOUT_SOURCE_DIR", str(ROOT / "data"))).strip())
    build_rows = int(np.clip(int(float(os.getenv("Q_EXTERNAL_HOLDOUT_BUILD_ROWS", "504"))), 126, 5000))
    min_symbols = int(np.clip(int(float(os.getenv("Q_EXTERNAL_HOLDOUT_BUILD_MIN_SYMBOLS", "8"))), 1, 2000))
    required = _bool_env("Q_EXTERNAL_HOLDOUT_REQUIRED", _bool_env("Q_EXTERNAL_HOLDOUT_BUILD_REQUIRE", True))

    names = _read_asset_names(RUNS / "asset_names.csv")
    if not names and source_dir.exists():
        names = sorted([p.stem.upper() for p in source_dir.glob("*.csv")])

    built = []
    missing = []
    short = []
    holdout_dir.mkdir(parents=True, exist_ok=True)

    for sym in names:
        src = source_dir / f"{sym}.csv"
        if not src.exists():
            missing.append(sym)
            continue
        px = _read_price(src)
        if px is None or len(px) < build_rows:
            short.append(sym)
            continue
        out = px.iloc[-build_rows:].copy()
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
        out_path = holdout_dir / f"{sym}.csv"
        out.to_csv(out_path, index=False)
        built.append(sym)

    info = {
        "ok": len(built) >= min_symbols,
        "required": bool(required),
        "source_dir": str(source_dir),
        "holdout_dir": str(holdout_dir),
        "build_rows": int(build_rows),
        "min_symbols": int(min_symbols),
        "built_count": int(len(built)),
        "missing_count": int(len(missing)),
        "short_count": int(len(short)),
        "built_symbols": built,
        "missing_symbols": missing[:100],
        "short_symbols": short[:100],
    }
    (RUNS / "external_holdout_build_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"✅ Wrote {RUNS/'external_holdout_build_info.json'}")
    print(f"External holdout build: built={len(built)} min_required={min_symbols} rows={build_rows}")

    if required and len(built) < min_symbols:
        print("(!) External holdout build requirement not met.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
