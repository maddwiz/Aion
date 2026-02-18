"""
Multi-timeframe bar engine for day_skimmer mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TimeframeData:
    bars: pd.DataFrame
    timeframe: str
    ema_fast: pd.Series | None = None
    ema_slow: pd.Series | None = None
    atr: pd.Series | None = None
    rsi: pd.Series | None = None
    vwap: pd.Series | None = None
    volume_ma: pd.Series | None = None


class BarEngine:
    """
    Maintains synchronized multi-timeframe views from 1-minute bars.
    """

    TIMEFRAMES = ["1m", "5m", "15m", "1H"]
    RESAMPLE_MAP = {"1m": "1min", "5m": "5min", "15m": "15min", "1H": "1h"}

    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_1m: pd.DataFrame = pd.DataFrame()
        self.frames: dict[str, TimeframeData] = {}
        self._ema_fast_span = int(getattr(cfg, "SKIMMER_EMA_FAST", 9))
        self._ema_slow_span = int(getattr(cfg, "SKIMMER_EMA_SLOW", 21))
        self._atr_period = int(getattr(cfg, "SKIMMER_ATR_PERIOD", 14))
        self._rsi_period = int(getattr(cfg, "SKIMMER_RSI_PERIOD", 14))
        self._vol_ma_period = int(getattr(cfg, "SKIMMER_VOL_MA_PERIOD", 20))

    def update(self, bars_1m: pd.DataFrame) -> dict[str, TimeframeData]:
        if bars_1m is None or bars_1m.empty:
            return self.frames

        df1 = self._normalize_bars(bars_1m)
        if df1.empty:
            return self.frames

        self.raw_1m = df1
        for tf in self.TIMEFRAMES:
            if tf == "1m":
                df = self.raw_1m.copy()
            else:
                df = self._resample(self.raw_1m, self.RESAMPLE_MAP[tf])
            if df.empty:
                continue

            tfd = TimeframeData(bars=df, timeframe=tf)
            tfd.ema_fast = df["close"].ewm(span=self._ema_fast_span, adjust=False).mean()
            tfd.ema_slow = df["close"].ewm(span=self._ema_slow_span, adjust=False).mean()
            tfd.atr = self._compute_atr(df, self._atr_period)
            tfd.rsi = self._compute_rsi(df["close"], self._rsi_period)
            tfd.vwap = self._compute_vwap(df)
            tfd.volume_ma = df["volume"].rolling(self._vol_ma_period, min_periods=1).mean()
            self.frames[tf] = tfd

        return self.frames

    def _normalize_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        cols = {str(c).strip().lower(): c for c in df.columns}

        if not isinstance(df.index, pd.DatetimeIndex):
            tcol = None
            for name in ["datetime", "date", "timestamp", "time"]:
                if name in cols:
                    tcol = cols[name]
                    break
            if tcol is None:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            idx = pd.to_datetime(df[tcol], errors="coerce")
            df = df.loc[idx.notna()].copy()
            df.index = pd.DatetimeIndex(idx[idx.notna()])

        rename = {}
        for need in ["open", "high", "low", "close", "volume"]:
            if need in cols:
                rename[cols[need]] = need
            elif need.capitalize() in df.columns:
                rename[need.capitalize()] = need
        df = df.rename(columns=rename)
        miss = [c for c in ["open", "high", "low", "close", "volume"] if c not in df.columns]
        if miss:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        out = df[["open", "high", "low", "close", "volume"]].copy()
        out = out.apply(pd.to_numeric, errors="coerce")
        out = out.dropna(subset=["open", "high", "low", "close"])
        out["volume"] = out["volume"].fillna(0.0)
        out = out.sort_index()
        return out

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        r = (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "close"])
        )
        return r

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        h = df["high"]
        l = df["low"]
        c_prev = df["close"].shift(1)
        tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
        return tr.ewm(span=max(2, int(period)), adjust=False).mean()

    def _compute_rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        p = max(2, int(period))
        avg_gain = gain.ewm(span=p, adjust=False).mean()
        avg_loss = loss.ewm(span=p, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].fillna(0.0)
        dates = df.index.date
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for d in np.unique(dates):
            mask = dates == d
            sess_tp = tp[mask]
            sess_vol = vol[mask]
            cum_vol = sess_vol.cumsum()
            cum_tp_vol = (sess_tp * sess_vol).cumsum()
            out[mask] = cum_tp_vol / (cum_vol + 1e-12)
        return out

    def get(self, tf: str) -> TimeframeData | None:
        return self.frames.get(tf)

    def latest_price(self) -> float | None:
        tf = self.frames.get("1m")
        if tf is None or tf.bars.empty:
            return None
        return float(tf.bars["close"].iloc[-1])

