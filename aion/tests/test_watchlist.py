import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aion.brain.watchlist import SkimmerWatchlistManager


def _make_cfg(tmp_path):
    aion_home = tmp_path / "aion_home"
    state_dir = tmp_path / "state"
    config_dir = aion_home / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    watchlist_file = config_dir / "skimmer_watchlist.csv"
    return SimpleNamespace(
        AION_HOME=aion_home,
        STATE_DIR=state_dir,
        SKIMMER_ATR_PERIOD=14,
        SKIMMER_WATCHLIST_FILE=watchlist_file,
        SKIMMER_WATCHLIST_MIN_AVG_VOLUME=10_000_000.0,
        SKIMMER_WATCHLIST_MAX_SPREAD_BPS=5.0,
        SKIMMER_WATCHLIST_REQUIRE_SPREAD=True,
        SKIMMER_WATCHLIST_MIN_ATR_5M=0.15,
        SKIMMER_WATCHLIST_MAX_SYMBOLS=12,
        SKIMMER_WATCHLIST_REFRESH_MIN=1,
        SKIMMER_WATCHLIST_Q_BIAS_THRESHOLD=0.30,
        SKIMMER_WATCHLIST_EARNINGS_DAYS=1,
        SKIMMER_WATCHLIST_EARNINGS_FILE=state_dir / "earnings_calendar.csv",
        SKIMMER_WATCHLIST_DURATION="3 D",
        SKIMMER_WATCHLIST_BARS_CACHE_SEC=0,
    )


def _bars(days: int = 2, volume_per_bar: float = 30_000.0, move: float = 0.30) -> pd.DataFrame:
    periods = 390 * days
    idx = pd.date_range("2026-01-05 09:30:00", periods=periods, freq="1min")
    base = 100.0 + np.arange(periods) * float(move) / 40.0
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_.values, close.values) + abs(move) * 0.45
    low = np.minimum(open_.values, close.values) - abs(move) * 0.45
    return pd.DataFrame(
        {
            "open": open_.values,
            "high": high,
            "low": low,
            "close": close.values,
            "volume": float(volume_per_bar),
        },
        index=idx,
    )


def test_watchlist_filters_volume_spread_atr_and_earnings(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.SKIMMER_WATCHLIST_FILE.write_text("symbol\nAAA\nBBB\nCCC\nDDD\nEEE\n", encoding="utf-8")
    today = dt.datetime.now().date().isoformat()
    cfg.SKIMMER_WATCHLIST_EARNINGS_FILE.write_text(f"symbol,date\nEEE,{today}\n", encoding="utf-8")

    bars_map = {
        "AAA": _bars(volume_per_bar=35_000, move=0.30),  # pass
        "BBB": _bars(volume_per_bar=2_000, move=0.30),  # low volume fail
        "CCC": _bars(volume_per_bar=35_000, move=0.30),  # spread fail
        "DDD": _bars(volume_per_bar=35_000, move=0.01),  # low ATR fail
        "EEE": _bars(volume_per_bar=35_000, move=0.30),  # earnings blackout fail
    }
    quote_map = {
        "AAA": (100.00, 100.03),  # ~3 bps
        "BBB": (100.00, 100.03),
        "CCC": (100.00, 100.20),  # ~20 bps
        "DDD": (100.00, 100.03),
        "EEE": (100.00, 100.03),
    }

    def bars_loader(symbol, duration=None, barSize=None, ttl_seconds=0):
        return bars_map.get(str(symbol).upper(), pd.DataFrame())

    def quote_loader(symbol):
        return quote_map.get(str(symbol).upper(), (None, None))

    m = SkimmerWatchlistManager(cfg, bars_loader=bars_loader, quote_loader=quote_loader)
    syms = m.get_active_symbols(overlay_bundle={"signals": {"__GLOBAL__": {"bias": 0.0}}})
    assert syms == ["AAA"]


def test_watchlist_prioritizes_strong_q_bias_with_cap(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.SKIMMER_WATCHLIST_MAX_SYMBOLS = 2
    cfg.SKIMMER_WATCHLIST_FILE.write_text("symbol\nAAA\nBBB\nCCC\n", encoding="utf-8")
    bars_df = _bars(volume_per_bar=35_000, move=0.30)

    def bars_loader(symbol, duration=None, barSize=None, ttl_seconds=0):
        return bars_df

    def quote_loader(symbol):
        return (100.0, 100.03)

    overlay = {
        "signals": {
            "__GLOBAL__": {"bias": 0.0},
            "AAA": {"bias": 0.10},
            "BBB": {"bias": 0.80},
            "CCC": {"bias": -0.40},
        }
    }
    m = SkimmerWatchlistManager(cfg, bars_loader=bars_loader, quote_loader=quote_loader)
    syms = m.get_active_symbols(overlay_bundle=overlay)
    assert syms == ["BBB", "CCC"]


def test_watchlist_falls_back_to_static_when_no_symbols_pass(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.SKIMMER_WATCHLIST_MAX_SYMBOLS = 2
    cfg.SKIMMER_WATCHLIST_FILE.write_text("symbol\nAAA\nBBB\nCCC\n", encoding="utf-8")

    def bars_loader(symbol, duration=None, barSize=None, ttl_seconds=0):
        return pd.DataFrame()

    def quote_loader(symbol):
        return (None, None)

    m = SkimmerWatchlistManager(cfg, bars_loader=bars_loader, quote_loader=quote_loader)
    syms = m.get_active_symbols(overlay_bundle=None)
    assert syms == ["AAA", "BBB"]
