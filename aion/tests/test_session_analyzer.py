from types import SimpleNamespace

import numpy as np
import pandas as pd

from aion.brain.session_analyzer import SessionAnalyzer, SessionPhase, SessionType


def _cfg(**kwargs):
    base = {
        "SKIMMER_OPENING_RANGE_MIN": 15,
        "SKIMMER_INITIAL_BALANCE_MIN": 60,
        "SKIMMER_LULL_START_HOUR": 11,
        "SKIMMER_LULL_START_MIN": 30,
        "SKIMMER_LULL_END_HOUR": 14,
        "SKIMMER_CLOSING_DRIVE_MIN": 60,
        "SKIMMER_VP_BINS": 24,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _bars_for_day(day: str, close: np.ndarray, volume: np.ndarray | None = None) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30:00", periods=len(close), freq="min")
    c = np.asarray(close, float)
    v = np.asarray(volume, float) if volume is not None else np.full(len(c), 1000.0)
    return pd.DataFrame(
        {
            "open": c - 0.05,
            "high": c + 0.20,
            "low": c - 0.20,
            "close": c,
            "volume": v,
        },
        index=idx,
    )


def test_phase_detection_across_session():
    sa = SessionAnalyzer(_cfg())

    opening = _bars_for_day("2026-01-05", np.linspace(100.0, 101.0, 10))
    st_open = sa.update(opening)
    assert st_open is not None and st_open.phase == SessionPhase.OPENING_DRIVE

    ib = _bars_for_day("2026-01-05", np.linspace(100.0, 102.0, 45))
    st_ib = sa.update(ib)
    assert st_ib is not None and st_ib.phase == SessionPhase.INITIAL_BALANCE

    midday = _bars_for_day("2026-01-05", np.linspace(100.0, 103.0, 150))
    st_mid = sa.update(midday)
    assert st_mid is not None and st_mid.phase == SessionPhase.MIDDAY_LULL
    assert st_mid.trade_allowed is False

    closing = _bars_for_day("2026-01-05", np.linspace(100.0, 104.0, 350))
    st_close = sa.update(closing)
    assert st_close is not None and st_close.phase == SessionPhase.CLOSING_DRIVE


def test_trend_day_classification():
    sa = SessionAnalyzer(_cfg())
    close = np.linspace(100.0, 112.0, 220)
    st = sa.update(_bars_for_day("2026-01-06", close))
    assert st is not None
    assert st.session_type == SessionType.TREND_DAY


def test_range_day_classification():
    sa = SessionAnalyzer(_cfg())
    x = np.linspace(0.0, np.pi, 140)
    close = 100.0 + 0.6 * np.sin(x)
    st = sa.update(_bars_for_day("2026-01-07", close))
    assert st is not None
    assert st.session_type == SessionType.RANGE_DAY


def test_reversal_day_detection():
    sa = SessionAnalyzer(_cfg())
    seg1 = np.linspace(100.0, 110.0, 60)
    seg2 = np.linspace(110.0, 95.0, 90)
    close = np.r_[seg1, seg2]
    st = sa.update(_bars_for_day("2026-01-08", close))
    assert st is not None
    assert st.session_type == SessionType.REVERSAL_DAY


def test_volume_profile_poc_moves_toward_high_volume_prices():
    sa = SessionAnalyzer(_cfg(SKIMMER_VP_BINS=20))
    close = np.linspace(100.0, 110.0, 50)
    vol = np.full(50, 10.0)
    vol[-6:] = 5000.0
    sess = _bars_for_day("2026-01-09", close, volume=vol)
    va_high, va_low, poc = sa._compute_volume_profile(sess)
    assert va_high >= va_low
    assert poc > 108.0


def test_midday_lull_aggression_is_low():
    sa = SessionAnalyzer(_cfg())
    close = np.linspace(100.0, 104.0, 160)  # ends near 12:09 ET
    st = sa.update(_bars_for_day("2026-01-10", close))
    assert st is not None
    assert st.phase == SessionPhase.MIDDAY_LULL
    assert st.aggression_scalar <= 0.20


def test_prior_day_levels_are_extracted():
    sa = SessionAnalyzer(_cfg())
    prev = _bars_for_day("2026-01-11", np.linspace(95.0, 105.0, 80))
    curr = _bars_for_day("2026-01-12", np.linspace(100.0, 108.0, 120))
    both = pd.concat([prev, curr], axis=0)

    st = sa.update(both)
    assert st is not None
    assert abs(st.levels.prior_day_high - float(prev["high"].max())) < 1e-9
    assert abs(st.levels.prior_day_low - float(prev["low"].min())) < 1e-9
    assert abs(st.levels.prior_day_close - float(prev["close"].iloc[-1])) < 1e-9

