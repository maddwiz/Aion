from types import SimpleNamespace

import numpy as np
import pandas as pd

from aion.brain.bar_engine import BarEngine


def _cfg(**kwargs):
    base = {
        "SKIMMER_EMA_FAST": 9,
        "SKIMMER_EMA_SLOW": 21,
        "SKIMMER_ATR_PERIOD": 14,
        "SKIMMER_RSI_PERIOD": 14,
        "SKIMMER_VOL_MA_PERIOD": 20,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_resample_1m_to_5m_has_correct_ohlc():
    idx = pd.date_range("2026-01-05 09:30:00", periods=10, freq="min")
    o = 100.0 + np.arange(10, dtype=float)
    df = pd.DataFrame(
        {
            "open": o,
            "high": o + np.array([0.8, 0.4, 1.1, 0.6, 1.2, 0.7, 1.3, 0.5, 1.0, 0.9]),
            "low": o - np.array([0.5, 0.3, 0.4, 0.6, 0.2, 0.5, 0.3, 0.7, 0.4, 0.6]),
            "close": o + 0.25,
            "volume": np.arange(1, 11, dtype=float),
        },
        index=idx,
    )

    be = BarEngine(_cfg())
    frames = be.update(df)
    b5 = frames["5m"].bars

    assert len(b5) == 2
    assert float(b5["open"].iloc[0]) == float(df["open"].iloc[0])
    assert float(b5["high"].iloc[0]) == float(df["high"].iloc[:5].max())
    assert float(b5["low"].iloc[0]) == float(df["low"].iloc[:5].min())
    assert float(b5["close"].iloc[0]) == float(df["close"].iloc[4])
    assert float(b5["volume"].iloc[0]) == float(df["volume"].iloc[:5].sum())


def test_vwap_resets_on_session_boundary():
    idx1 = pd.date_range("2026-01-05 09:30:00", periods=3, freq="min")
    idx2 = pd.date_range("2026-01-06 09:30:00", periods=3, freq="min")
    idx = idx1.append(idx2)

    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 200, 201, 202],
            "high": [101, 102, 103, 200, 202, 203],
            "low": [99, 100, 101, 200, 200, 201],
            "close": [100.5, 101.5, 102.5, 200.0, 201.0, 202.0],
            "volume": [10, 10, 10, 5, 5, 5],
        },
        index=idx,
    )

    be = BarEngine(_cfg())
    tfd = be.update(df)["1m"]
    vwap = tfd.vwap

    day2_first = idx2[0]
    tp_first_day2 = float((df.loc[day2_first, "high"] + df.loc[day2_first, "low"] + df.loc[day2_first, "close"]) / 3.0)
    assert abs(float(vwap.loc[day2_first]) - tp_first_day2) < 1e-9


def test_atr_rsi_ema_are_computed_on_synthetic_data():
    idx = pd.date_range("2026-01-07 09:30:00", periods=60, freq="min")
    close = np.linspace(100.0, 115.0, num=60)
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(60, 1000.0),
        },
        index=idx,
    )

    be = BarEngine(_cfg(SKIMMER_EMA_FAST=3, SKIMMER_EMA_SLOW=8, SKIMMER_ATR_PERIOD=7, SKIMMER_RSI_PERIOD=7))
    tfd = be.update(df)["1m"]

    assert float(tfd.ema_fast.iloc[-1]) > float(tfd.ema_slow.iloc[-1])
    assert np.isfinite(tfd.atr.values).all()
    assert float(tfd.atr.iloc[-1]) >= 0.0
    assert 0.0 <= float(tfd.rsi.iloc[-1]) <= 100.0
    assert float(tfd.rsi.iloc[-1]) > 70.0
    assert abs(float(be.latest_price()) - float(close[-1])) < 1e-12


def test_empty_input_is_safe():
    be = BarEngine(_cfg())
    out = be.update(pd.DataFrame())
    assert out == {}
    assert be.latest_price() is None

