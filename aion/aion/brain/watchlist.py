"""
Day-skimmer watchlist management.

Builds a tradable intraday shortlist from a static watchlist and
session-time filters (liquidity, spread, ATR, earnings blackout),
then prioritizes symbols with strong Q overlay bias.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.ib_client import hist_bars_cached, ib

DEFAULT_STATIC_WATCHLIST = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "META",
    "AMD",
    "GOOGL",
]


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _norm_symbol(x: str) -> str:
    out = "".join(ch for ch in str(x).upper() if ch.isalnum())
    return out


def _dedupe_symbols(vals: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for raw in vals:
        s = _norm_symbol(raw)
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def _normalize_bars(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    x = df.copy()
    rename = {}
    for c in x.columns:
        lc = str(c).strip().lower()
        if lc in {"date", "datetime", "timestamp", "time"}:
            rename[c] = "date"
        elif lc in {"open", "high", "low", "close", "volume"}:
            rename[c] = lc
    if rename:
        x = x.rename(columns=rename)

    if "date" in x.columns:
        idx = pd.to_datetime(x["date"], errors="coerce")
        x = x.loc[~idx.isna()].copy()
        x.index = pd.DatetimeIndex(idx[~idx.isna()])
    elif not isinstance(x.index, pd.DatetimeIndex):
        idx = pd.to_datetime(x.index, errors="coerce")
        x = x.loc[~idx.isna()].copy()
        x.index = pd.DatetimeIndex(idx[~idx.isna()])
    x = x.sort_index()

    for c in ["open", "high", "low", "close"]:
        if c not in x.columns:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        x[c] = pd.to_numeric(x[c], errors="coerce")

    if "volume" not in x.columns:
        x["volume"] = 0.0
    x["volume"] = pd.to_numeric(x["volume"], errors="coerce").fillna(0.0)
    x = x.dropna(subset=["open", "high", "low", "close"])
    if x.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return x[["open", "high", "low", "close", "volume"]]


def _resample_5m(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
    )


def _atr_5m(df_1m: pd.DataFrame, period: int = 14) -> float:
    if df_1m.empty:
        return 0.0
    df = _resample_5m(df_1m)
    if len(df) < 3:
        return 0.0
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c_prev = pd.to_numeric(df["close"], errors="coerce").shift(1)
    tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=max(2, int(period)), adjust=False).mean()
    if atr.empty:
        return 0.0
    return float(max(0.0, _safe_float(atr.iloc[-1], 0.0)))


def _avg_daily_volume(df_1m: pd.DataFrame) -> float:
    if df_1m.empty or not isinstance(df_1m.index, pd.DatetimeIndex):
        return 0.0
    vol = pd.to_numeric(df_1m["volume"], errors="coerce").fillna(0.0)
    if vol.empty:
        return 0.0
    daily = vol.groupby(df_1m.index.date).sum()
    if daily.empty:
        return 0.0
    return float(max(0.0, _safe_float(daily.mean(), 0.0)))


def _default_quote_loader(symbol: str) -> tuple[float | None, float | None]:
    try:
        client = ib()
        from ib_insync import Stock  # local import to keep tests lightweight

        contracts = client.qualifyContracts(Stock(str(symbol).upper(), "SMART", "USD"))
        if not contracts:
            return None, None
        ticker = client.reqTickers(contracts[0])[0]
        bid = _safe_float(getattr(ticker, "bid", None), float("nan"))
        ask = _safe_float(getattr(ticker, "ask", None), float("nan"))
        if not math.isfinite(bid) or not math.isfinite(ask):
            return None, None
        return float(bid), float(ask)
    except Exception:
        return None, None


def _spread_bps(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    b = _safe_float(bid, float("nan"))
    a = _safe_float(ask, float("nan"))
    if (not math.isfinite(b)) or (not math.isfinite(a)) or b <= 0 or a <= 0 or a <= b:
        return None
    mid = (a + b) / 2.0
    if mid <= 0:
        return None
    return float(((a - b) / mid) * 10000.0)


def _load_earnings_blackout_symbols(path: Path, lookahead_days: int, today: dt.date | None = None) -> set[str]:
    if not path.exists():
        return set()
    today = today or dt.datetime.now().date()
    lo = today - dt.timedelta(days=max(0, int(lookahead_days)))
    hi = today + dt.timedelta(days=max(0, int(lookahead_days)))
    out: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = _norm_symbol(row.get("symbol", ""))
                date_raw = str(row.get("date", "")).strip()
                if not sym or not date_raw:
                    continue
                try:
                    d = dt.date.fromisoformat(date_raw[:10])
                except Exception:
                    continue
                if lo <= d <= hi:
                    out.add(sym)
    except Exception:
        return set()
    return out


def _overlay_bias_map(overlay_bundle: dict | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(overlay_bundle, dict):
        return out
    signals = overlay_bundle.get("signals", {})
    if not isinstance(signals, dict):
        return out
    global_bias = 0.0
    g = signals.get("__GLOBAL__", {})
    if isinstance(g, dict):
        global_bias = float(np.clip(_safe_float(g.get("bias", 0.0), 0.0), -1.0, 1.0))
    for k, v in signals.items():
        if not isinstance(v, dict):
            continue
        sym = _norm_symbol(k)
        if not sym or sym == "GLOBAL":
            continue
        out[sym] = float(np.clip(_safe_float(v.get("bias", global_bias), global_bias), -1.0, 1.0))
    out["__GLOBAL__"] = global_bias
    return out


@dataclass
class WatchlistEntry:
    symbol: str
    avg_volume: float
    spread_bps: float | None
    atr_5m: float
    q_bias: float
    earnings_blackout: bool
    pass_volume: bool
    pass_spread: bool
    pass_atr: bool
    selected: bool

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "avg_volume": float(self.avg_volume),
            "spread_bps": (None if self.spread_bps is None else float(self.spread_bps)),
            "atr_5m": float(self.atr_5m),
            "q_bias": float(self.q_bias),
            "earnings_blackout": bool(self.earnings_blackout),
            "pass_volume": bool(self.pass_volume),
            "pass_spread": bool(self.pass_spread),
            "pass_atr": bool(self.pass_atr),
            "selected": bool(self.selected),
        }


class SkimmerWatchlistManager:
    def __init__(
        self,
        cfg_mod,
        *,
        bars_loader=None,
        quote_loader=None,
    ):
        self.cfg = cfg_mod
        self._bars_loader = bars_loader or hist_bars_cached
        self._quote_loader = quote_loader or _default_quote_loader
        self._cached_symbols: list[str] = []
        self._cached_entries: list[WatchlistEntry] = []
        self._last_refresh_monotonic = 0.0

    def _static_file(self) -> Path:
        p = getattr(self.cfg, "SKIMMER_WATCHLIST_FILE", None)
        if p:
            return Path(p)
        return Path(self.cfg.AION_HOME) / "config" / "skimmer_watchlist.csv"

    def _fallback_watchlist_txt(self) -> Path:
        return Path(self.cfg.STATE_DIR) / "watchlist.txt"

    def _runtime_json_path(self) -> Path:
        return Path(self.cfg.STATE_DIR) / "skimmer_watchlist.json"

    def _load_static_symbols(self) -> list[str]:
        p = self._static_file()
        if p.exists():
            try:
                lines = p.read_text(encoding="utf-8").splitlines()
                vals = []
                for ln in lines:
                    s = str(ln).strip()
                    if not s or s.startswith("#"):
                        continue
                    if "," in s:
                        tok = s.split(",")[0].strip()
                    else:
                        tok = s
                    if tok.lower() in {"symbol", "ticker"}:
                        continue
                    vals.append(tok)
                syms = _dedupe_symbols(vals)
                if syms:
                    return syms
            except Exception:
                pass

        fb = self._fallback_watchlist_txt()
        if fb.exists():
            try:
                syms = _dedupe_symbols(fb.read_text(encoding="utf-8").splitlines())
                if syms:
                    return syms
            except Exception:
                pass
        return list(DEFAULT_STATIC_WATCHLIST)

    def _refresh_needed(self) -> bool:
        refresh_min = max(1, int(getattr(self.cfg, "SKIMMER_WATCHLIST_REFRESH_MIN", 30)))
        if not self._cached_symbols:
            return True
        elapsed = time.monotonic() - float(self._last_refresh_monotonic)
        return elapsed >= float(refresh_min * 60)

    def _build_entries(self, overlay_bundle: dict | None) -> list[WatchlistEntry]:
        symbols = self._load_static_symbols()
        bias_map = _overlay_bias_map(overlay_bundle)
        global_bias = float(bias_map.get("__GLOBAL__", 0.0))

        min_volume = float(max(0.0, getattr(self.cfg, "SKIMMER_WATCHLIST_MIN_AVG_VOLUME", 10_000_000.0)))
        max_spread_bps = float(max(0.0, getattr(self.cfg, "SKIMMER_WATCHLIST_MAX_SPREAD_BPS", 5.0)))
        require_spread = bool(getattr(self.cfg, "SKIMMER_WATCHLIST_REQUIRE_SPREAD", False))
        min_atr = float(max(0.0, getattr(self.cfg, "SKIMMER_WATCHLIST_MIN_ATR_5M", 0.15)))
        earnings_days = int(max(0, getattr(self.cfg, "SKIMMER_WATCHLIST_EARNINGS_DAYS", 1)))
        earnings_file = Path(getattr(self.cfg, "SKIMMER_WATCHLIST_EARNINGS_FILE", Path(self.cfg.STATE_DIR) / "earnings_calendar.csv"))
        duration = str(getattr(self.cfg, "SKIMMER_WATCHLIST_DURATION", "3 D"))
        ttl = int(max(0, getattr(self.cfg, "SKIMMER_WATCHLIST_BARS_CACHE_SEC", 60)))

        earnings_blackout = _load_earnings_blackout_symbols(earnings_file, earnings_days)

        entries: list[WatchlistEntry] = []
        for sym in symbols:
            bars_raw = self._bars_loader(sym, duration=duration, barSize="1 min", ttl_seconds=ttl)
            bars = _normalize_bars(bars_raw)
            avg_volume = _avg_daily_volume(bars)
            atr_5m = _atr_5m(bars, period=int(max(2, getattr(self.cfg, "SKIMMER_ATR_PERIOD", 14))))

            bid = ask = None
            spread_bps = None
            try:
                bid, ask = self._quote_loader(sym)
                spread_bps = _spread_bps(bid, ask)
            except Exception:
                spread_bps = None

            pass_volume = bool(avg_volume >= min_volume)
            pass_atr = bool(atr_5m >= min_atr)
            pass_spread = bool(spread_bps is None and (not require_spread))
            if spread_bps is not None:
                pass_spread = bool(spread_bps <= max_spread_bps)
            in_earnings_blackout = bool(_norm_symbol(sym) in earnings_blackout)

            selected = bool(pass_volume and pass_atr and pass_spread and (not in_earnings_blackout))
            entries.append(
                WatchlistEntry(
                    symbol=str(sym).upper(),
                    avg_volume=float(avg_volume),
                    spread_bps=(None if spread_bps is None else float(spread_bps)),
                    atr_5m=float(atr_5m),
                    q_bias=float(np.clip(bias_map.get(sym, global_bias), -1.0, 1.0)),
                    earnings_blackout=in_earnings_blackout,
                    pass_volume=pass_volume,
                    pass_spread=pass_spread,
                    pass_atr=pass_atr,
                    selected=selected,
                )
            )
        return entries

    def _rank_symbols(self, entries: list[WatchlistEntry]) -> list[str]:
        max_symbols = int(max(1, getattr(self.cfg, "SKIMMER_WATCHLIST_MAX_SYMBOLS", 15)))
        q_th = float(max(0.0, getattr(self.cfg, "SKIMMER_WATCHLIST_Q_BIAS_THRESHOLD", 0.30)))
        selected = [e for e in entries if e.selected]
        if not selected:
            return self._load_static_symbols()[:max_symbols]

        selected.sort(
            key=lambda e: (
                abs(float(e.q_bias)) >= q_th,
                abs(float(e.q_bias)),
                float(e.atr_5m),
                float(e.avg_volume),
            ),
            reverse=True,
        )
        return [e.symbol for e in selected[:max_symbols]]

    def _write_runtime_info(self, symbols: list[str], entries: list[WatchlistEntry]):
        payload = {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "selected_count": int(len(symbols)),
            "selected_symbols": list(symbols),
            "entries": [e.to_dict() for e in entries],
        }
        p = self._runtime_json_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            return

    def get_active_symbols(self, overlay_bundle: dict | None = None) -> list[str]:
        if not self._refresh_needed():
            return list(self._cached_symbols)

        entries = self._build_entries(overlay_bundle)
        symbols = self._rank_symbols(entries)
        self._cached_entries = list(entries)
        self._cached_symbols = list(symbols)
        self._last_refresh_monotonic = time.monotonic()
        self._write_runtime_info(symbols, entries)
        return list(symbols)
