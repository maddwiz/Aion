"""
Intraday pattern detectors for day_skimmer mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _empty(level: str = "") -> dict:
    return {"detected": False, "direction": 0, "strength": 0.0, "level": level}


def _atr_value(atr: pd.Series | None, default: float = 1.0) -> float:
    if atr is None or len(atr) == 0:
        return float(default)
    try:
        v = float(pd.to_numeric(atr, errors="coerce").iloc[-1])
    except Exception:
        return float(default)
    if not np.isfinite(v) or v <= 1e-9:
        return float(default)
    return float(v)


def _last_bars(df: pd.DataFrame, n: int) -> pd.DataFrame | None:
    if df is None or df.empty or len(df) < n:
        return None
    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        return None
    out = df.tail(n).copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def detect_engulfing(bars_5m: pd.DataFrame, atr: pd.Series | None = None) -> dict:
    win = _last_bars(bars_5m, 2)
    if win is None:
        return _empty("candle")
    prev = win.iloc[-2]
    cur = win.iloc[-1]
    body_prev = abs(float(prev["close"] - prev["open"]))
    body_cur = abs(float(cur["close"] - cur["open"]))
    if body_prev <= 1e-9 or body_cur <= 1e-9:
        return _empty("candle")

    bull = (
        float(prev["close"]) < float(prev["open"])
        and float(cur["close"]) > float(cur["open"])
        and float(cur["close"]) >= float(prev["open"])
        and float(cur["open"]) <= float(prev["close"])
    )
    bear = (
        float(prev["close"]) > float(prev["open"])
        and float(cur["close"]) < float(cur["open"])
        and float(cur["close"]) <= float(prev["open"])
        and float(cur["open"]) >= float(prev["close"])
    )
    if not (bull or bear):
        return _empty("candle")

    av = _atr_value(atr)
    strength = min(1.0, 0.5 * (body_cur / max(body_prev, 1e-9)) + 0.5 * (body_cur / max(av, 1e-9)))
    return {
        "detected": True,
        "direction": 1 if bull else -1,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "level": "candle",
    }


def detect_pin_bar(
    bars_5m: pd.DataFrame,
    atr: pd.Series | None = None,
    level_prices: dict[str, float] | None = None,
    level_tol_atr: float = 0.35,
) -> dict:
    win = _last_bars(bars_5m, 1)
    if win is None:
        return _empty("none")
    row = win.iloc[-1]
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    body = abs(c - o)
    rng = max(1e-9, h - l)
    upper = h - max(o, c)
    lower = min(o, c) - l
    dominant = max(upper, lower)
    minor = min(upper, lower)
    if body <= 1e-9:
        body = 0.01 * rng

    wick_ratio = dominant / max(body, 1e-9)
    if wick_ratio < 2.0 or minor > dominant * 0.7:
        return _empty("none")

    direction = -1 if upper > lower else 1
    av = _atr_value(atr)
    near_level = "none"
    if isinstance(level_prices, dict) and level_prices:
        tol = abs(float(level_tol_atr)) * av
        price_ref = c
        best_key = None
        best_dist = None
        for k, v in level_prices.items():
            if v is None:
                continue
            try:
                lv = float(v)
            except Exception:
                continue
            d = abs(price_ref - lv)
            if (best_dist is None) or (d < best_dist):
                best_dist = d
                best_key = str(k)
        if best_key is not None and best_dist is not None and best_dist <= tol:
            near_level = best_key

    strength = 0.45 + 0.30 * min(1.0, wick_ratio / 3.5) + 0.25 * min(1.0, dominant / max(av, 1e-9))
    return {
        "detected": True,
        "direction": int(direction),
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "level": near_level,
    }


def detect_three_bar_reversal(bars_5m: pd.DataFrame, atr: pd.Series | None = None) -> dict:
    win = _last_bars(bars_5m, 3)
    if win is None:
        return _empty("three_bar")
    a, b, c = win.iloc[-3], win.iloc[-2], win.iloc[-1]
    av = _atr_value(atr)

    bull = (
        float(a["low"]) < min(float(b["low"]), float(c["low"]))
        and abs(float(b["close"] - b["open"])) <= 0.45 * max(float(b["high"] - b["low"]), 1e-9)
        and float(c["close"]) > float(a["open"])
    )
    bear = (
        float(a["high"]) > max(float(b["high"]), float(c["high"]))
        and abs(float(b["close"] - b["open"])) <= 0.45 * max(float(b["high"] - b["low"]), 1e-9)
        and float(c["close"]) < float(a["open"])
    )
    if not (bull or bear):
        return _empty("three_bar")

    impulse = abs(float(c["close"] - a["open"])) / max(av, 1e-9)
    strength = 0.5 + 0.5 * min(1.0, impulse)
    return {
        "detected": True,
        "direction": 1 if bull else -1,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "level": "three_bar",
    }


def detect_inside_bar_breakout(bars_5m: pd.DataFrame, atr: pd.Series | None = None) -> dict:
    win = _last_bars(bars_5m, 5)
    if win is None:
        return _empty("inside_bar")
    h = win["high"].values.astype(float)
    l = win["low"].values.astype(float)
    c = win["close"].values.astype(float)

    # Detect at least two-bar compression before the breakout bar.
    ib1 = (h[-3] <= h[-4]) and (l[-3] >= l[-4])
    ib2 = (h[-2] <= h[-3]) and (l[-2] >= l[-3])
    if not (ib1 and ib2):
        return _empty("inside_bar")

    box_hi = max(h[-4], h[-3], h[-2])
    box_lo = min(l[-4], l[-3], l[-2])
    bull = c[-1] > box_hi
    bear = c[-1] < box_lo
    if not (bull or bear):
        return _empty("inside_bar")

    av = _atr_value(atr)
    move = abs(c[-1] - (box_hi if bull else box_lo))
    strength = 0.5 + 0.5 * min(1.0, move / max(av, 1e-9))
    return {
        "detected": True,
        "direction": 1 if bull else -1,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "level": "inside_bar",
    }


def detect_momentum_burst(bars_5m: pd.DataFrame, atr: pd.Series | None = None) -> dict:
    win = _last_bars(bars_5m, 3)
    if win is None:
        return _empty("momentum")
    o = win["open"].values.astype(float)
    c = win["close"].values.astype(float)
    v = win["volume"].values.astype(float) if "volume" in win.columns else np.array([1.0, 1.0, 1.0], dtype=float)
    dir_bars = np.sign(c - o)

    bull = np.all(dir_bars > 0) and (v[0] <= v[1] <= v[2])
    bear = np.all(dir_bars < 0) and (v[0] <= v[1] <= v[2])
    if not (bull or bear):
        return _empty("momentum")

    av = _atr_value(atr)
    total_move = abs(c[-1] - o[0])
    vol_boost = (v[-1] / max(v[0], 1e-9)) - 1.0
    strength = 0.45 + 0.35 * min(1.0, total_move / max(av, 1e-9)) + 0.20 * min(1.0, max(0.0, vol_boost))
    return {
        "detected": True,
        "direction": 1 if bull else -1,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "level": "momentum",
    }


def detect_failed_breakout(
    bars_5m: pd.DataFrame,
    atr: pd.Series | None = None,
    lookback: int = 20,
    hold_bars: int = 2,
) -> dict:
    n = max(lookback, hold_bars + 4)
    win = _last_bars(bars_5m, n)
    if win is None:
        return _empty("breakout_level")
    h = win["high"].values.astype(float)
    l = win["low"].values.astype(float)
    c = win["close"].values.astype(float)
    # Use level from bars before the breakout attempt.
    ref_hi = float(np.max(h[: -(hold_bars + 1)]))
    ref_lo = float(np.min(l[: -(hold_bars + 1)]))

    # Bearish failed upside breakout.
    upside_break = c[-(hold_bars + 1)] > ref_hi
    upside_hold = np.all(c[-hold_bars:] >= ref_hi)
    upside_fail = c[-1] < ref_hi
    bear = upside_break and (upside_hold or c[-2] > ref_hi) and upside_fail

    # Bullish failed downside breakdown.
    downside_break = c[-(hold_bars + 1)] < ref_lo
    downside_hold = np.all(c[-hold_bars:] <= ref_lo)
    downside_fail = c[-1] > ref_lo
    bull = downside_break and (downside_hold or c[-2] < ref_lo) and downside_fail

    if not (bull or bear):
        return _empty("breakout_level")

    av = _atr_value(atr)
    move = abs(c[-1] - (ref_lo if bull else ref_hi))
    strength = 0.55 + 0.45 * min(1.0, move / max(av, 1e-9))
    return {
        "detected": True,
        "direction": 1 if bull else -1,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "level": "breakout_level",
    }


def detect_compression_breakout(bars_5m: pd.DataFrame, atr: pd.Series | None = None, lookback: int = 24) -> dict:
    n = max(lookback, 16)
    win = _last_bars(bars_5m, n)
    if win is None:
        return _empty("compression")
    rng = (win["high"] - win["low"]).astype(float)
    if len(rng) < 12:
        return _empty("compression")

    recent = rng.iloc[-10:-1]
    if recent.empty:
        return _empty("compression")
    base_recent = float(recent.median())
    compressed = base_recent <= float(rng.iloc[:-1].quantile(0.25))
    av = _atr_value(atr, default=float(recent.mean()) if len(recent) else 1.0)
    burst_abs = float(rng.iloc[-1]) >= 1.5 * max(av, 1e-9)
    burst_rel = float(rng.iloc[-1]) >= 1.25 * max(base_recent, 1e-9)
    burst = burst_abs and burst_rel
    if not (compressed and burst):
        return _empty("compression")

    direction = 1 if float(win["close"].iloc[-1]) >= float(win["open"].iloc[-1]) else -1
    strength = 0.5 + 0.5 * min(1.0, float(rng.iloc[-1]) / max(2.0 * av, 1e-9))
    return {
        "detected": True,
        "direction": int(direction),
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "level": "compression",
    }


def detect_wick_rejection(
    bars_5m: pd.DataFrame,
    atr: pd.Series | None = None,
    vwap: float | None = None,
    vwap_tol_atr: float = 0.35,
) -> dict:
    win = _last_bars(bars_5m, 1)
    if win is None:
        return _empty("vwap")
    row = win.iloc[-1]
    if vwap is None:
        if "vwap" in bars_5m.columns:
            try:
                vwap = float(pd.to_numeric(bars_5m["vwap"], errors="coerce").iloc[-1])
            except Exception:
                vwap = None
    if vwap is None or (not np.isfinite(float(vwap))):
        return _empty("vwap")

    av = _atr_value(atr)
    tol = abs(float(vwap_tol_atr)) * av
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l

    near_vwap = abs(c - float(vwap)) <= tol or (l <= float(vwap) <= h)
    if not near_vwap:
        return _empty("vwap")

    bull = (lower >= 2.0 * max(body, 1e-9)) and (c >= float(vwap))
    bear = (upper >= 2.0 * max(body, 1e-9)) and (c <= float(vwap))
    if not (bull or bear):
        return _empty("vwap")

    strength = 0.45 + 0.55 * min(1.0, max(upper, lower) / max(av, 1e-9))
    return {
        "detected": True,
        "direction": 1 if bull else -1,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "level": "vwap",
    }


def detect_all_intraday_patterns(bars_5m: pd.DataFrame, atr: pd.Series | None) -> dict:
    return {
        "engulfing": detect_engulfing(bars_5m, atr),
        "pin_bar": detect_pin_bar(bars_5m, atr),
        "three_bar_reversal": detect_three_bar_reversal(bars_5m, atr),
        "inside_bar_breakout": detect_inside_bar_breakout(bars_5m, atr),
        "momentum_burst": detect_momentum_burst(bars_5m, atr),
        "failed_breakout": detect_failed_breakout(bars_5m, atr),
        "compression_breakout": detect_compression_breakout(bars_5m, atr),
        "wick_rejection": detect_wick_rejection(bars_5m, atr),
    }
