"""
Session structure analysis for day_skimmer mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class SessionPhase(Enum):
    PRE_OPEN = "pre_open"
    OPENING_DRIVE = "opening_drive"
    INITIAL_BALANCE = "initial_balance"
    RANGE_EXTENSION = "range_extension"
    MIDDAY_LULL = "midday_lull"
    CLOSING_DRIVE = "closing_drive"
    POST_CLOSE = "post_close"


class SessionType(Enum):
    TREND_DAY = "trend_day"
    RANGE_DAY = "range_day"
    REVERSAL_DAY = "reversal_day"
    BREAKOUT_DAY = "breakout_day"
    ROTATIONAL = "rotational"


@dataclass
class SessionLevels:
    open_price: float
    session_high: float
    session_low: float
    opening_range_high: float
    opening_range_low: float
    initial_balance_high: float
    initial_balance_low: float
    vwap: float
    poc: float
    value_area_high: float
    value_area_low: float
    prior_day_high: float
    prior_day_low: float
    prior_day_close: float
    prior_day_vwap: float


@dataclass
class SessionState:
    phase: SessionPhase
    session_type: SessionType
    levels: SessionLevels
    minutes_into_session: int
    range_pct: float
    ib_range_pct: float
    range_extension_up: bool
    range_extension_down: bool
    vwap_slope: float
    volume_profile_skew: float
    relative_volume: float
    trade_allowed: bool
    aggression_scalar: float


class SessionAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._opening_range_minutes = int(getattr(cfg, "SKIMMER_OPENING_RANGE_MIN", 15))
        self._initial_balance_minutes = int(getattr(cfg, "SKIMMER_INITIAL_BALANCE_MIN", 60))
        self._lull_start_hour = int(getattr(cfg, "SKIMMER_LULL_START_HOUR", 11))
        self._lull_start_minute = int(getattr(cfg, "SKIMMER_LULL_START_MIN", 30))
        self._lull_end_hour = int(getattr(cfg, "SKIMMER_LULL_END_HOUR", 14))
        self._closing_drive_min = int(getattr(cfg, "SKIMMER_CLOSING_DRIVE_MIN", 60))
        self._volume_profile_bins = int(getattr(cfg, "SKIMMER_VP_BINS", 24))
        self._min_bars_for_analysis = 10

    def update(self, bars_1m: pd.DataFrame, prior_day_bars: pd.DataFrame | None = None) -> SessionState | None:
        if bars_1m is None or len(bars_1m) < self._min_bars_for_analysis:
            return None
        sess = self._extract_current_session(bars_1m)
        if sess is None or len(sess) < self._min_bars_for_analysis:
            return None

        prior = self._extract_prior_session(bars_1m, prior_day_bars)
        levels = self._compute_levels(sess, prior)
        phase = self._determine_phase(sess)
        minutes = len(sess)
        session_type = self._classify_session_type(sess, levels, minutes)

        current = float(sess["close"].iloc[-1])
        range_ext_up = current > levels.initial_balance_high
        range_ext_down = current < levels.initial_balance_low
        range_pct = (levels.session_high - levels.session_low) / (levels.open_price + 1e-9) * 100.0
        ib_range_pct = (levels.initial_balance_high - levels.initial_balance_low) / (levels.open_price + 1e-9) * 100.0
        vwap_slope = self._compute_vwap_slope(sess)
        vp_skew = self._compute_volume_profile_skew(sess)
        rel_vol = self._compute_relative_volume(sess)
        trade_allowed = phase not in (SessionPhase.PRE_OPEN, SessionPhase.POST_CLOSE, SessionPhase.MIDDAY_LULL)
        aggression = self._compute_aggression(phase, session_type, rel_vol, range_ext_up, range_ext_down)

        return SessionState(
            phase=phase,
            session_type=session_type,
            levels=levels,
            minutes_into_session=minutes,
            range_pct=float(range_pct),
            ib_range_pct=float(ib_range_pct),
            range_extension_up=bool(range_ext_up),
            range_extension_down=bool(range_ext_down),
            vwap_slope=float(vwap_slope),
            volume_profile_skew=float(vp_skew),
            relative_volume=float(rel_vol),
            trade_allowed=bool(trade_allowed),
            aggression_scalar=float(aggression),
        )

    def _extract_current_session(self, bars: pd.DataFrame) -> pd.DataFrame | None:
        if not isinstance(bars.index, pd.DatetimeIndex):
            return bars
        latest_date = bars.index[-1].date()
        sess = bars[bars.index.date == latest_date]
        return sess if not sess.empty else None

    def _extract_prior_session(self, bars: pd.DataFrame, prior: pd.DataFrame | None) -> pd.DataFrame | None:
        if prior is not None and not prior.empty:
            return prior
        if not isinstance(bars.index, pd.DatetimeIndex):
            return None
        dates = sorted(set(bars.index.date))
        if len(dates) < 2:
            return None
        prior_date = dates[-2]
        prev = bars[bars.index.date == prior_date]
        return prev if not prev.empty else None

    def _compute_levels(self, sess: pd.DataFrame, prior: pd.DataFrame | None) -> SessionLevels:
        open_price = float(sess["open"].iloc[0])
        session_high = float(sess["high"].max())
        session_low = float(sess["low"].min())

        or_n = min(self._opening_range_minutes, len(sess))
        or_slice = sess.iloc[:or_n]
        orh = float(or_slice["high"].max())
        orl = float(or_slice["low"].min())

        ib_n = min(self._initial_balance_minutes, len(sess))
        ib_slice = sess.iloc[:ib_n]
        ibh = float(ib_slice["high"].max())
        ibl = float(ib_slice["low"].min())

        tp = (sess["high"] + sess["low"] + sess["close"]) / 3.0
        vol = sess["volume"].fillna(0.0)
        cvol = vol.cumsum()
        vwap = float((tp * vol).cumsum().iloc[-1] / (cvol.iloc[-1] + 1e-9))

        va_high, va_low, poc = self._compute_volume_profile(sess)

        if prior is not None and not prior.empty:
            pd_high = float(prior["high"].max())
            pd_low = float(prior["low"].min())
            pd_close = float(prior["close"].iloc[-1])
            pd_tp = (prior["high"] + prior["low"] + prior["close"]) / 3.0
            pd_vol = prior["volume"].fillna(0.0)
            pd_cvol = pd_vol.cumsum()
            pd_vwap = float((pd_tp * pd_vol).cumsum().iloc[-1] / (pd_cvol.iloc[-1] + 1e-9))
        else:
            pd_high = session_high
            pd_low = session_low
            pd_close = open_price
            pd_vwap = vwap

        return SessionLevels(
            open_price=open_price,
            session_high=session_high,
            session_low=session_low,
            opening_range_high=orh,
            opening_range_low=orl,
            initial_balance_high=ibh,
            initial_balance_low=ibl,
            vwap=vwap,
            poc=poc,
            value_area_high=va_high,
            value_area_low=va_low,
            prior_day_high=pd_high,
            prior_day_low=pd_low,
            prior_day_close=pd_close,
            prior_day_vwap=pd_vwap,
        )

    def _compute_volume_profile(self, sess: pd.DataFrame) -> tuple[float, float, float]:
        prices = (sess["high"] + sess["low"] + sess["close"]) / 3.0
        volumes = sess["volume"].fillna(0.0)
        price_min = float(prices.min())
        price_max = float(prices.max())
        if price_max - price_min < 1e-6:
            p = float(prices.mean()) if len(prices) else 0.0
            return p, p, p

        n_bins = max(4, int(self._volume_profile_bins))
        bins = np.linspace(price_min, price_max, n_bins + 1)
        bin_vol = np.zeros(n_bins, dtype=float)
        pv = prices.values
        vv = volumes.values
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (pv >= bins[i]) & (pv <= bins[i + 1])
            else:
                mask = (pv >= bins[i]) & (pv < bins[i + 1])
            bin_vol[i] = float(vv[mask].sum())

        poc_idx = int(np.argmax(bin_vol))
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0

        total_vol = float(bin_vol.sum())
        target = 0.70 * total_vol
        va_vol = float(bin_vol[poc_idx])
        lo_idx = poc_idx
        hi_idx = poc_idx
        while va_vol < target and (lo_idx > 0 or hi_idx < n_bins - 1):
            expand_lo = float(bin_vol[lo_idx - 1]) if lo_idx > 0 else -1.0
            expand_hi = float(bin_vol[hi_idx + 1]) if hi_idx < n_bins - 1 else -1.0
            if expand_lo >= expand_hi and lo_idx > 0:
                lo_idx -= 1
                va_vol += max(0.0, expand_lo)
            elif hi_idx < n_bins - 1:
                hi_idx += 1
                va_vol += max(0.0, expand_hi)
            elif lo_idx > 0:
                lo_idx -= 1
                va_vol += max(0.0, expand_lo)
            else:
                break

        return float(bins[hi_idx + 1]), float(bins[lo_idx]), float(poc)

    def _determine_phase(self, sess: pd.DataFrame) -> SessionPhase:
        if not isinstance(sess.index, pd.DatetimeIndex):
            return SessionPhase.OPENING_DRIVE
        now = sess.index[-1]
        minutes = len(sess)
        hour, minute = int(now.hour), int(now.minute)

        if hour < 9 or (hour == 9 and minute < 30):
            return SessionPhase.PRE_OPEN
        if hour >= 16:
            return SessionPhase.POST_CLOSE
        if minutes <= self._opening_range_minutes:
            return SessionPhase.OPENING_DRIVE
        if minutes <= self._initial_balance_minutes:
            return SessionPhase.INITIAL_BALANCE

        lull_start = self._lull_start_hour * 60 + self._lull_start_minute
        lull_end = self._lull_end_hour * 60
        cur_min = hour * 60 + minute
        close_min = 16 * 60 - self._closing_drive_min

        if cur_min >= close_min:
            return SessionPhase.CLOSING_DRIVE
        if lull_start <= cur_min < lull_end:
            return SessionPhase.MIDDAY_LULL
        return SessionPhase.RANGE_EXTENSION

    def _classify_session_type(self, sess: pd.DataFrame, levels: SessionLevels, minutes: int) -> SessionType:
        if minutes < self._initial_balance_minutes:
            return SessionType.RANGE_DAY

        current = float(sess["close"].iloc[-1])
        ib_range = levels.initial_balance_high - levels.initial_balance_low
        total_range = levels.session_high - levels.session_low
        open_to_current = current - levels.open_price

        if total_range > 0 and abs(open_to_current) > 0.6 * total_range:
            ext_up = levels.session_high > levels.initial_balance_high
            ext_down = levels.session_low < levels.initial_balance_low
            if (ext_up and not ext_down) or (ext_down and not ext_up):
                return SessionType.TREND_DAY

        if ib_range > 0 and total_range > 2.0 * ib_range:
            return SessionType.BREAKOUT_DAY

        if minutes > 90:
            first_30 = sess.iloc[: min(30, len(sess))]
            early_direction = float(first_30["close"].iloc[-1]) - levels.open_price
            if early_direction != 0 and np.sign(open_to_current) != np.sign(early_direction):
                if abs(open_to_current) > 0.3 * total_range:
                    return SessionType.REVERSAL_DAY

        closes = sess["close"].values
        vwap_val = levels.vwap
        crosses = int(np.sum(np.abs(np.diff(np.sign(closes - vwap_val))) > 0))
        if crosses >= 4:
            return SessionType.ROTATIONAL

        return SessionType.RANGE_DAY

    def _compute_vwap_slope(self, sess: pd.DataFrame, lookback: int = 20) -> float:
        tp = (sess["high"] + sess["low"] + sess["close"]) / 3.0
        vol = sess["volume"].fillna(0.0)
        cvol = vol.cumsum()
        vwap = (tp * vol).cumsum() / (cvol + 1e-9)
        if len(vwap) < lookback:
            lookback = max(3, len(vwap))
        recent = vwap.iloc[-lookback:]
        if len(recent) < 2:
            return 0.0
        return float((recent.iloc[-1] - recent.iloc[0]) / (abs(recent.iloc[0]) + 1e-9))

    def _compute_volume_profile_skew(self, sess: pd.DataFrame) -> float:
        mid = (float(sess["high"].max()) + float(sess["low"].min())) / 2.0
        tp = (sess["high"] + sess["low"] + sess["close"]) / 3.0
        vol = sess["volume"].fillna(0.0)
        above = float(vol[tp >= mid].sum())
        below = float(vol[tp < mid].sum())
        total = above + below + 1e-12
        return float((above - below) / total)

    def _compute_relative_volume(self, sess: pd.DataFrame) -> float:
        vol = sess["volume"].fillna(0.0)
        if len(vol) < 5:
            return 1.0
        recent = float(vol.iloc[-5:].mean())
        overall = float(vol.mean())
        return float(recent / (overall + 1e-9))

    def _compute_aggression(
        self,
        phase: SessionPhase,
        session_type: SessionType,
        rel_vol: float,
        ext_up: bool,
        ext_down: bool,
    ) -> float:
        if phase in (SessionPhase.PRE_OPEN, SessionPhase.POST_CLOSE, SessionPhase.MIDDAY_LULL):
            return 0.0

        if phase == SessionPhase.OPENING_DRIVE:
            base = 0.70
        elif phase == SessionPhase.INITIAL_BALANCE:
            base = 0.50
        elif phase == SessionPhase.RANGE_EXTENSION:
            base = 0.80
        elif phase == SessionPhase.CLOSING_DRIVE:
            base = 0.65
        else:
            base = 0.0

        if session_type == SessionType.TREND_DAY:
            base = min(1.0, base + 0.20)
        elif session_type == SessionType.ROTATIONAL:
            base = max(0.0, base - 0.25)
        elif session_type == SessionType.REVERSAL_DAY:
            base = max(0.0, base - 0.10)

        if ext_up or ext_down:
            base = min(1.0, base + 0.10)
        if rel_vol > 1.5:
            base = min(1.0, base + 0.08)
        elif rel_vol < 0.6:
            base = max(0.0, base - 0.15)

        return float(np.clip(base, 0.0, 1.0))
