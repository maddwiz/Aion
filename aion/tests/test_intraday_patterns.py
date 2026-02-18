import numpy as np
import pandas as pd

from aion.brain.intraday_patterns import (
    detect_all_intraday_patterns,
    detect_failed_breakout,
    detect_inside_bar_breakout,
    detect_momentum_burst,
    detect_pin_bar,
)


def _bars(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05 10:00:00", periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_pin_bar_detects_long_wick_rejection():
    df = _bars(
        [
            (100.0, 100.3, 99.8, 100.2, 900),
            (100.2, 100.5, 100.0, 100.4, 920),
            (100.3, 100.6, 100.2, 100.5, 950),
            (100.4, 100.7, 100.3, 100.6, 980),
            # body=0.10, lower wick=0.55 => 5.5x body
            (100.50, 100.62, 99.95, 100.60, 1300),
        ]
    )
    atr = pd.Series(np.full(len(df), 0.35), index=df.index)
    out = detect_pin_bar(df, atr)
    assert out["detected"] is True
    assert int(out["direction"]) == 1
    assert float(out["strength"]) > 0.0


def test_inside_bar_compression_breakout_detects():
    df = _bars(
        [
            (100.0, 100.5, 99.7, 100.2, 1000),
            (100.2, 100.8, 99.9, 100.3, 1000),
            (100.3, 101.0, 99.8, 100.4, 1000),  # mother
            (100.4, 100.9, 100.0, 100.5, 950),   # inside
            (100.5, 100.8, 100.1, 100.55, 920),  # inside
            (100.6, 101.3, 100.3, 101.2, 1300),  # breakout
        ]
    )
    atr = pd.Series(np.full(len(df), 0.40), index=df.index)
    out = detect_inside_bar_breakout(df, atr)
    assert out["detected"] is True
    assert int(out["direction"]) == 1


def test_momentum_burst_detects_three_bar_volume_push():
    df = _bars(
        [
            (100.0, 100.4, 99.8, 100.3, 1000),
            (100.3, 100.8, 100.1, 100.7, 1400),
            (100.7, 101.2, 100.6, 101.1, 1900),
        ]
    )
    atr = pd.Series(np.full(len(df), 0.30), index=df.index)
    out = detect_momentum_burst(df, atr)
    assert out["detected"] is True
    assert int(out["direction"]) == 1
    assert float(out["strength"]) >= 0.5


def test_failed_breakout_detects_rejection_after_break():
    rows = []
    for _ in range(17):
        rows.append((99.8, 100.2, 99.5, 99.9, 1000.0))
    rows.extend(
        [
            (100.1, 101.5, 100.0, 101.2, 1600.0),  # breakout above ref high
            (101.1, 101.3, 100.7, 100.9, 1400.0),  # holds above
            (100.8, 101.0, 99.2, 99.4, 1900.0),    # fails back below
        ]
    )
    df = _bars(rows)
    atr = pd.Series(np.full(len(df), 0.50), index=df.index)
    out = detect_failed_breakout(df, atr, lookback=20, hold_bars=2)
    assert out["detected"] is True
    assert int(out["direction"]) == -1


def test_no_false_positives_on_flat_data():
    df = _bars([(100.0, 100.1, 99.9, 100.0, 1000.0) for _ in range(30)])
    atr = pd.Series(np.full(len(df), 0.05), index=df.index)
    out = detect_all_intraday_patterns(df, atr)
    assert all(bool(v.get("detected", False)) is False for v in out.values())

