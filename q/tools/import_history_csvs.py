#!/usr/bin/env python3
# Import and normalize historical CSVs into Q's data folder.
#
# Usage:
#   python tools/import_history_csvs.py --src "/path/to/csv_folder"
#
# Writes normalized files into ./data as:
#   SYMBOL.csv with columns DATE,Close
# Also writes:
#   runs_plus/import_history_summary.json

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
DATA = ROOT / "data"
RUNS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

DATE_CANDS = ["date", "datetime", "timestamp", "time", "Date", "DATE", "Timestamp"]
PRICE_CANDS = [
    "Adj Close",
    "adj_close",
    "AdjClose",
    "Close",
    "close",
    "last",
    "Last",
    "price",
    "Price",
    "PX_LAST",
]


def _norm_symbol(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "", str(name or "").strip().upper())
    s = s.replace("-", "").replace("/", "")
    return s or "UNKNOWN"


def _pick_col(cols, candidates):
    cset = list(cols)
    low = {c.lower(): c for c in cset}
    for c in candidates:
        if c in cset:
            return c
        lc = c.lower()
        if lc in low:
            return low[lc]
    return None


def _import_one(src: Path, dst_dir: Path):
    try:
        df = pd.read_csv(src)
    except Exception as e:
        return {"file": src.name, "ok": False, "reason": f"read_error: {e}"}
    if df.empty:
        return {"file": src.name, "ok": False, "reason": "empty_csv"}

    dcol = _pick_col(df.columns, DATE_CANDS)
    pcol = _pick_col(df.columns, PRICE_CANDS)
    if dcol is None:
        return {"file": src.name, "ok": False, "reason": "date_col_not_found"}
    if pcol is None:
        # fallback: first numeric column that is not date-like
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        num = [c for c in num if str(c) != str(dcol)]
        if not num:
            return {"file": src.name, "ok": False, "reason": "price_col_not_found"}
        pcol = num[-1]

    out = pd.DataFrame(
        {
            "DATE": pd.to_datetime(df[dcol], errors="coerce"),
            "Close": pd.to_numeric(df[pcol], errors="coerce"),
        }
    )
    out = out.dropna(subset=["DATE", "Close"]).sort_values("DATE")
    if len(out) < 20:
        return {"file": src.name, "ok": False, "reason": "too_few_rows_after_clean"}
    out["DATE"] = out["DATE"].dt.normalize()
    out = out.drop_duplicates(subset=["DATE"], keep="last")

    sym = _norm_symbol(src.stem.replace("_prices", ""))
    dst = dst_dir / f"{sym}.csv"
    out.to_csv(dst, index=False)
    return {"file": src.name, "ok": True, "symbol": sym, "rows": int(len(out)), "out": str(dst)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source folder containing historical CSVs.")
    ap.add_argument("--dst", default=str(DATA), help="Destination folder for normalized data CSVs.")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern under source (default: *.csv).")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Source folder not found: {src_dir}")

    rows = []
    for fp in sorted(src_dir.glob(args.glob)):
        if not fp.is_file():
            continue
        rows.append(_import_one(fp, dst_dir))

    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    summary = {
        "source": str(src_dir),
        "dest": str(dst_dir),
        "total_files": int(len(rows)),
        "imported": int(len(ok)),
        "failed": int(len(bad)),
        "results": rows,
    }
    (RUNS / "import_history_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"✅ Imported {len(ok)}/{len(rows)} files into {dst_dir}")
    print(f"✅ Wrote {RUNS/'import_history_summary.json'}")
