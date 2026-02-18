from types import SimpleNamespace

import numpy as np
import pandas as pd

from aion.brain.bar_engine import TimeframeData
from aion.brain.intraday_confluence import IntradaySignalBundle, score_intraday_entry
from aion.brain.session_analyzer import SessionLevels, SessionPhase, SessionState, SessionType


def _cfg(th: float = 0.58):
    return SimpleNamespace(SKIMMER_ENTRY_THRESHOLD=th)


def _tfd(close_vals, vol_vals, ema_fast_last, ema_slow_last, rsi_last, atr_last=1.0):
    close_vals = np.asarray(close_vals, float)
    vol_vals = np.asarray(vol_vals, float)
    idx = pd.date_range("2026-01-05 10:00:00", periods=len(close_vals), freq="5min")
    bars = pd.DataFrame({"close": close_vals, "volume": vol_vals}, index=idx)
    ema_fast = pd.Series(np.linspace(ema_fast_last - 0.2, ema_fast_last, len(close_vals)), index=idx)
    ema_slow = pd.Series(np.linspace(ema_slow_last - 0.2, ema_slow_last, len(close_vals)), index=idx)
    rsi = pd.Series(np.full(len(close_vals), float(rsi_last)), index=idx)
    atr = pd.Series(np.full(len(close_vals), float(atr_last)), index=idx)
    vol_ma = pd.Series(np.full(len(close_vals), max(1.0, float(np.mean(vol_vals)))), index=idx)
    return TimeframeData(
        bars=bars,
        timeframe="x",
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        atr=atr,
        rsi=rsi,
        vwap=None,
        volume_ma=vol_ma,
    )


def _session(phase=SessionPhase.RANGE_EXTENSION, trade_allowed=True):
    levels = SessionLevels(
        open_price=100.0,
        session_high=112.0,
        session_low=99.0,
        opening_range_high=101.0,
        opening_range_low=99.5,
        initial_balance_high=103.0,
        initial_balance_low=99.2,
        vwap=111.0,
        poc=110.9,
        value_area_high=110.7,
        value_area_low=109.8,
        prior_day_high=111.1,
        prior_day_low=98.0,
        prior_day_close=100.5,
        prior_day_vwap=100.0,
    )
    return SessionState(
        phase=phase,
        session_type=SessionType.TREND_DAY,
        levels=levels,
        minutes_into_session=180,
        range_pct=2.0,
        ib_range_pct=1.0,
        range_extension_up=True,
        range_extension_down=False,
        vwap_slope=0.01,
        volume_profile_skew=0.4,
        relative_volume=1.8,
        trade_allowed=trade_allowed,
        aggression_scalar=0.95,
    )


def _bundle(side="LONG", q_bias=0.7, q_conf=0.9, phase=SessionPhase.RANGE_EXTENSION, trade_allowed=True):
    bars = {
        "1m": _tfd([110.7, 110.9, 111.0], [2000, 2100, 2300], 110.95, 110.2, 62.0, atr_last=1.0),
        "5m": _tfd([109.0, 109.8, 110.7], [1000, 1000, 2400], 110.8, 109.9, 64.0, atr_last=1.0),
        "15m": _tfd([107.5, 108.9, 110.4], [1200, 1200, 2300], 110.3, 109.6, 60.0, atr_last=1.1),
        "1H": _tfd([104.0, 106.2, 109.5], [1100, 1300, 2200], 109.9, 108.8, 58.0, atr_last=1.3),
    }
    patterns = {
        "pin_bar": {"detected": True, "direction": 1 if side == "LONG" else -1, "strength": 0.95},
        "momentum_burst": {"detected": True, "direction": 1 if side == "LONG" else -1, "strength": 0.90},
        "inside_bar_breakout": {"detected": True, "direction": 1 if side == "LONG" else -1, "strength": 0.95},
        "compression_breakout": {"detected": True, "direction": 1 if side == "LONG" else -1, "strength": 0.90},
    }
    return IntradaySignalBundle(
        symbol="AAPL",
        side=side,
        session=_session(phase=phase, trade_allowed=trade_allowed),
        patterns=patterns,
        bars=bars,
        q_overlay_bias=q_bias,
        q_overlay_confidence=q_conf,
    )


def test_max_confluence_scores_high():
    b = _bundle(side="LONG", q_bias=0.9, q_conf=1.0)
    out = score_intraday_entry(b, _cfg(0.58))
    assert out.score > 0.85
    assert out.entry_allowed is True


def test_midday_lull_blocks_entry_even_if_score_high():
    b = _bundle(side="LONG", q_bias=0.9, q_conf=1.0, phase=SessionPhase.MIDDAY_LULL, trade_allowed=False)
    out = score_intraday_entry(b, _cfg(0.40))
    assert out.score > 0.40
    assert out.entry_allowed is False


def test_opposing_q_overlay_reduces_score():
    aligned = score_intraday_entry(_bundle(side="LONG", q_bias=0.8, q_conf=1.0), _cfg(0.58))
    opposed = score_intraday_entry(_bundle(side="LONG", q_bias=-0.8, q_conf=1.0), _cfg(0.58))
    assert opposed.score < aligned.score


def test_trend_day_wrong_direction_penalizes():
    b = _bundle(side="SHORT", q_bias=-0.2, q_conf=0.8)  # trend day modeled bullish
    out = score_intraday_entry(b, _cfg(0.58))
    assert out.category_scores["session_structure"] < 0.8


def test_threshold_behavior_just_above_and_below():
    b = _bundle(side="LONG", q_bias=0.4, q_conf=0.7)
    mid = score_intraday_entry(b, _cfg(0.58))
    allow = score_intraday_entry(b, _cfg(mid.score - 0.01))
    block = score_intraday_entry(b, _cfg(mid.score + 0.01))
    assert allow.entry_allowed is True
    assert block.entry_allowed is False
