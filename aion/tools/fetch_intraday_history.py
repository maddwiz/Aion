#!/usr/bin/env python3
"""
Fetch and store intraday (1-minute) history from IBKR for day_skimmer backtesting.

Outputs one file per symbol/day:
  aion/data/intraday/SYMBOL_1m_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aion import config as cfg
from aion.data.ib_client import disconnect, hist_bars_cached, ib


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    x = df.copy()
    rename = {}
    for c in x.columns:
        lc = str(c).strip().lower()
        if lc in {"date", "datetime", "timestamp", "time"}:
            rename[c] = "datetime"
        elif lc in {"open", "high", "low", "close", "volume"}:
            rename[c] = lc
    if rename:
        x = x.rename(columns=rename)
    if "datetime" not in x.columns:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    x["datetime"] = pd.to_datetime(x["datetime"], errors="coerce")
    x = x.dropna(subset=["datetime"]).copy()
    for c in ["open", "high", "low", "close"]:
        if c not in x.columns:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if "volume" not in x.columns:
        x["volume"] = 0.0
    x["volume"] = pd.to_numeric(x["volume"], errors="coerce").fillna(0.0)
    x = x.dropna(subset=["open", "high", "low", "close"]).copy()
    x = x.sort_values("datetime")
    return x[["datetime", "open", "high", "low", "close", "volume"]]


def _load_default_symbols() -> list[str]:
    p = Path(cfg.AION_HOME) / "config" / "skimmer_watchlist.csv"
    if p.exists():
        vals = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            s = str(ln).strip()
            if not s or s.startswith("#") or s.lower() in {"symbol", "ticker"}:
                continue
            vals.append(s.split(",")[0].strip().upper())
        if vals:
            return vals
    return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "AMD", "GOOGL"]


def fetch_intraday_history(
    *,
    symbols: list[str],
    out_dir: Path,
    duration: str = "3 D",
    pause_sec: float = 1.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ib()
    fetched = 0
    written = 0
    for raw in symbols:
        sym = str(raw).strip().upper()
        if not sym:
            continue
        fetched += 1
        df = hist_bars_cached(sym, duration=duration, barSize="1 min", ttl_seconds=0)
        bars = _normalize_bars(df)
        if bars.empty:
            print(f"[fetch-intraday] {sym}: no bars")
            continue
        bars["session_date"] = bars["datetime"].dt.strftime("%Y%m%d")
        for sess, chunk in bars.groupby("session_date", sort=True):
            out = out_dir / f"{sym}_1m_{sess}.csv"
            chunk.drop(columns=["session_date"]).to_csv(out, index=False)
            written += 1
        print(f"[fetch-intraday] {sym}: sessions={bars['session_date'].nunique()} rows={len(bars)}")
        time.sleep(max(0.0, float(pause_sec)))
    disconnect()
    print(f"[fetch-intraday] symbols={fetched} files_written={written} out_dir={out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch 1-minute IBKR history and store per-symbol/session CSV files.")
    p.add_argument("--symbols", default="", help="Comma-separated symbols. Defaults to skimmer watchlist.")
    p.add_argument("--out-dir", default=str(Path(cfg.AION_HOME) / "data" / "intraday"))
    p.add_argument("--duration", default="3 D")
    p.add_argument("--pause-sec", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    symbols = [x.strip().upper() for x in str(args.symbols).split(",") if x.strip()]
    if not symbols:
        symbols = _load_default_symbols()
    fetch_intraday_history(
        symbols=symbols,
        out_dir=Path(args.out_dir),
        duration=str(args.duration),
        pause_sec=float(args.pause_sec),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
