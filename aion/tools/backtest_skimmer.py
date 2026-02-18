#!/usr/bin/env python3
"""
Minute-bar replay backtester for AION day_skimmer mode.

Reads local 1-minute CSV files:
  aion/data/intraday/SYMBOL_1m_YYYYMMDD.csv

and writes:
  aion/state/skimmer_backtest_trades.csv
  aion/state/skimmer_backtest_summary.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aion import config as cfg
from aion.brain.bar_engine import BarEngine
from aion.brain.intraday_confluence import IntradaySignalBundle, score_intraday_entry
from aion.brain.intraday_patterns import detect_all_intraday_patterns
from aion.brain.intraday_risk import IntradayRiskManager, IntradayRiskParams, compute_position_size
from aion.brain.session_analyzer import SessionAnalyzer

FILE_RE = re.compile(r"^([A-Za-z0-9\.\-_]+)_1m_(\d{8})\.csv$")


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    x = df.copy()
    rename = {}
    for c in x.columns:
        lc = str(c).strip().lower()
        if lc in {"datetime", "date", "timestamp", "time"}:
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


def _discover_files(data_dir: Path, symbols: set[str] | None = None) -> list[tuple[str, Path, dt.date]]:
    out: list[tuple[str, Path, dt.date]] = []
    if not data_dir.exists():
        return out
    for p in sorted(data_dir.glob("*_1m_*.csv")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        sym = str(m.group(1)).upper()
        if symbols and sym not in symbols:
            continue
        try:
            d = dt.datetime.strptime(m.group(2), "%Y%m%d").date()
        except Exception:
            continue
        out.append((sym, p, d))
    return out


def _minutes_to_close(ts: pd.Timestamp | None) -> int:
    if ts is None:
        return 390
    return max(0, (16 * 60) - (int(ts.hour) * 60 + int(ts.minute)))


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = float(equity_curve[0])
    mdd = 0.0
    for eq in equity_curve:
        peak = max(peak, float(eq))
        if peak <= 0:
            continue
        dd = float((eq - peak) / peak)
        mdd = min(mdd, dd)
    return float(mdd)


@dataclass
class BacktestPosition:
    trade_id: int
    symbol: str
    side: str
    entry_price: float
    entry_qty: int
    current_qty: int
    risk_distance: float
    partial_taken: bool
    highest_since_entry: float
    lowest_since_entry: float
    entry_time: pd.Timestamp
    stop_price: float
    r_target_1: float
    entry_equity: float
    realized_pnl: float = 0.0


class SkimmerBacktester:
    def __init__(self, cfg_mod=cfg):
        self.cfg = cfg_mod
        self.equity_start = float(cfg_mod.EQUITY_START)
        self.cash = float(cfg_mod.EQUITY_START)
        self.position: BacktestPosition | None = None
        self.last_price = 0.0
        self.trade_id_seq = 0
        self.trade_logs: list[dict] = []
        self.trade_results: list[dict] = []
        self.equity_curve: list[float] = [float(self.equity_start)]
        self.session_returns: list[float] = []
        self.risk = IntradayRiskManager(
            equity=float(cfg_mod.EQUITY_START),
            params=IntradayRiskParams(
                stop_atr_multiple=float(cfg_mod.SKIMMER_STOP_ATR_MULTIPLE),
                risk_per_trade_pct=float(cfg_mod.SKIMMER_RISK_PER_TRADE),
                max_position_pct=float(cfg_mod.SKIMMER_MAX_POSITION_PCT),
                partial_profit_r=float(cfg_mod.SKIMMER_PARTIAL_PROFIT_R),
                partial_profit_fraction=float(cfg_mod.SKIMMER_PARTIAL_PROFIT_FRAC),
                trailing_stop_atr=float(cfg_mod.SKIMMER_TRAILING_STOP_ATR),
                max_trades_per_session=int(cfg_mod.SKIMMER_MAX_TRADES_SESSION),
                max_daily_loss_pct=float(cfg_mod.SKIMMER_MAX_DAILY_LOSS_PCT),
                max_open_positions=1,
                no_new_entries_after_min=int(cfg_mod.SKIMMER_NO_ENTRY_BEFORE_CLOSE_MIN),
                force_close_all_at_min=int(cfg_mod.SKIMMER_FORCE_CLOSE_BEFORE_MIN),
            ),
        )

    def _equity(self, mark_price: float | None = None) -> float:
        px = float(self.last_price if mark_price is None else mark_price)
        eq = float(self.cash)
        if self.position is not None:
            if self.position.side == "LONG":
                eq += px * self.position.current_qty
            else:
                eq -= px * self.position.current_qty
        return float(eq)

    def _log_event(self, *, ts: pd.Timestamp, symbol: str, action: str, qty: int, price: float, reason: str, pnl: float, score: float):
        self.trade_logs.append(
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": str(symbol).upper(),
                "trade_id": int(self.position.trade_id if self.position is not None else -1),
                "action": str(action),
                "qty": int(max(0, qty)),
                "price": float(price),
                "reason": str(reason),
                "pnl": float(pnl),
                "equity": float(self._equity(price)),
                "confluence_score": float(np.clip(score, 0.0, 1.0)),
            }
        )

    def _close_qty(self, ts: pd.Timestamp, qty: int, reason: str, price: float, score: float = 0.0):
        if self.position is None:
            return
        q = int(max(0, min(int(qty), int(self.position.current_qty))))
        if q <= 0:
            return
        pos = self.position
        if pos.side == "LONG":
            self.cash += float(price) * q
            pnl = (float(price) - float(pos.entry_price)) * q
        else:
            self.cash -= float(price) * q
            pnl = (float(pos.entry_price) - float(price)) * q
        pos.current_qty -= q
        pos.realized_pnl += float(pnl)
        self._log_event(
            ts=ts,
            symbol=pos.symbol,
            action=("PARTIAL_EXIT" if reason == "partial_profit_1R" else "EXIT"),
            qty=q,
            price=float(price),
            reason=str(reason),
            pnl=float(pnl),
            score=float(score),
        )
        if pos.current_qty <= 0:
            risk_unit = float(max(1e-9, pos.risk_distance * max(1, pos.entry_qty)))
            r_multiple = float(pos.realized_pnl / risk_unit)
            self.trade_results.append(
                {
                    "trade_id": int(pos.trade_id),
                    "symbol": str(pos.symbol).upper(),
                    "side": str(pos.side),
                    "entry_time": str(pos.entry_time),
                    "exit_time": str(ts),
                    "pnl": float(pos.realized_pnl),
                    "r_multiple": float(r_multiple),
                    "entry_equity": float(pos.entry_equity),
                }
            )
            self.risk.record_trade_result(float(pos.realized_pnl))
            self.position = None

    def _manage_position(self, ts: pd.Timestamp, frames: dict[str, object], price: float):
        if self.position is None:
            return
        pos = self.position
        tfd5 = frames.get("5m")
        atr_5m = 0.0
        if tfd5 is not None and getattr(tfd5, "atr", None) is not None and len(tfd5.atr) > 0:
            atr_5m = float(max(0.0, _safe_float(tfd5.atr.iloc[-1], 0.0)))

        if pos.side == "LONG":
            pos.highest_since_entry = max(float(pos.highest_since_entry), float(price))
            if float(price) <= float(pos.stop_price):
                self._close_qty(ts, pos.current_qty, "initial_stop", float(price))
                return
            if (not pos.partial_taken) and float(price) >= float(pos.r_target_1):
                close_qty = int(pos.current_qty * float(self.risk.params.partial_profit_fraction))
                if close_qty > 0:
                    self._close_qty(ts, close_qty, "partial_profit_1R", float(price))
                    if self.position is None:
                        return
                    self.position.partial_taken = True
                    self.position.stop_price = max(float(self.position.stop_price), float(self.position.entry_price))
            if self.position and self.position.partial_taken and self.position.current_qty > 0 and atr_5m > 0:
                trail = max(
                    float(self.position.entry_price),
                    float(self.position.highest_since_entry - atr_5m * float(self.risk.params.trailing_stop_atr)),
                )
                if float(price) <= float(trail):
                    self._close_qty(ts, self.position.current_qty, "trailing_stop", float(price))
        else:
            pos.lowest_since_entry = min(float(pos.lowest_since_entry), float(price))
            if float(price) >= float(pos.stop_price):
                self._close_qty(ts, pos.current_qty, "initial_stop", float(price))
                return
            if (not pos.partial_taken) and float(price) <= float(pos.r_target_1):
                close_qty = int(pos.current_qty * float(self.risk.params.partial_profit_fraction))
                if close_qty > 0:
                    self._close_qty(ts, close_qty, "partial_profit_1R", float(price))
                    if self.position is None:
                        return
                    self.position.partial_taken = True
                    self.position.stop_price = min(float(self.position.stop_price), float(self.position.entry_price))
            if self.position and self.position.partial_taken and self.position.current_qty > 0 and atr_5m > 0:
                trail = min(
                    float(self.position.entry_price),
                    float(self.position.lowest_since_entry + atr_5m * float(self.risk.params.trailing_stop_atr)),
                )
                if float(price) >= float(trail):
                    self._close_qty(ts, self.position.current_qty, "trailing_stop", float(price))

    def _evaluate_entry(self, ts: pd.Timestamp, symbol: str, frames: dict[str, object], session_state):
        if self.position is not None:
            return
        mins_to_close = _minutes_to_close(ts)
        can_enter, _ = self.risk.can_enter(mins_to_close, 0)
        if not can_enter:
            return
        tfd1 = frames.get("1m")
        tfd5 = frames.get("5m")
        if (
            tfd1 is None
            or tfd5 is None
            or tfd1.bars is None
            or tfd5.bars is None
            or tfd1.bars.empty
            or tfd5.bars.empty
            or getattr(tfd5, "atr", None) is None
            or len(tfd5.atr) == 0
        ):
            return
        price = float(_safe_float(tfd1.bars["close"].iloc[-1], 0.0))
        atr_5m = float(max(0.0, _safe_float(tfd5.atr.iloc[-1], 0.0)))
        if price <= 0 or atr_5m <= 0:
            return
        patterns = detect_all_intraday_patterns(tfd5.bars, tfd5.atr)

        best_side = None
        best_score = None
        for side in ["LONG", "SHORT"]:
            bundle = IntradaySignalBundle(
                symbol=symbol,
                side=side,
                session=session_state,
                patterns=patterns,
                bars=frames,
                q_overlay_bias=0.0,
                q_overlay_confidence=0.0,
            )
            out = score_intraday_entry(bundle, self.cfg)
            if (best_score is None) or (out.score > best_score.score):
                best_side = side
                best_score = out
        if best_side is None or best_score is None or (not best_score.entry_allowed):
            return

        sizing = compute_position_size(
            side=best_side,
            entry_price=price,
            atr_5m=atr_5m,
            equity=max(1.0, self._equity(price)),
            params=self.risk.params,
        )
        qty = int(max(0, sizing.shares))
        if qty <= 0:
            return
        notional = float(price) * qty
        if best_side == "LONG":
            if self.cash < notional:
                return
            self.cash -= notional
        else:
            self.cash += notional

        self.trade_id_seq += 1
        self.position = BacktestPosition(
            trade_id=int(self.trade_id_seq),
            symbol=str(symbol).upper(),
            side=str(best_side),
            entry_price=float(price),
            entry_qty=int(qty),
            current_qty=int(qty),
            risk_distance=float(sizing.risk_distance),
            partial_taken=False,
            highest_since_entry=float(price),
            lowest_since_entry=float(price),
            entry_time=ts,
            stop_price=float(sizing.stop_price),
            r_target_1=float(sizing.r_target_1),
            entry_equity=float(self._equity(price)),
        )
        self._log_event(
            ts=ts,
            symbol=symbol,
            action=f"ENTRY_{best_side}",
            qty=qty,
            price=float(price),
            reason="confluence_entry",
            pnl=0.0,
            score=float(best_score.score),
        )

    def run_session(self, symbol: str, bars: pd.DataFrame):
        if bars.empty:
            return
        be = BarEngine(self.cfg)
        sa = SessionAnalyzer(self.cfg)
        start_eq = float(self._equity())

        for i in range(len(bars)):
            chunk = bars.iloc[: i + 1]
            frames = be.update(chunk)
            session_state = sa.update(chunk)
            price = float(_safe_float(chunk["close"].iloc[-1], 0.0))
            if price <= 0:
                continue
            ts = chunk.index[-1]
            self.last_price = float(price)
            self._manage_position(ts, frames, price)
            if self.position is None:
                self._evaluate_entry(ts, symbol, frames, session_state)
            if self.risk.should_force_close_all(_minutes_to_close(ts)) and self.position is not None:
                self._close_qty(ts, self.position.current_qty, "session_end", float(price))
            self.equity_curve.append(float(self._equity(price)))

        if self.position is not None:
            ts = bars.index[-1]
            self._close_qty(ts, self.position.current_qty, "session_close", float(bars["close"].iloc[-1]))
        end_eq = float(self._equity())
        if start_eq > 0:
            self.session_returns.append((end_eq - start_eq) / start_eq)

    def _summary(self, session_count: int, file_count: int) -> dict:
        pnls = np.asarray([float(x["pnl"]) for x in self.trade_results], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        hit_rate = float((len(wins) / len(pnls)) if len(pnls) else 0.0)
        avg_win = float(np.mean(wins)) if len(wins) else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) else 0.0
        gross_profit = float(np.sum(wins)) if len(wins) else 0.0
        gross_loss = float(-np.sum(losses)) if len(losses) else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        sess = np.asarray(self.session_returns, dtype=float)
        sharpe = 0.0
        if len(sess) > 1:
            sd = float(np.std(sess, ddof=1))
            if sd > 1e-12:
                sharpe = float((np.mean(sess) / sd) * math.sqrt(252.0))

        total_pnl = float(np.sum(pnls)) if len(pnls) else 0.0
        ending_equity = float(self.equity_start + total_pnl)
        mdd = float(_max_drawdown(self.equity_curve))

        return {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files_processed": int(file_count),
            "sessions_processed": int(session_count),
            "completed_trades": int(len(self.trade_results)),
            "hit_rate": float(hit_rate),
            "sharpe": float(sharpe),
            "total_pnl": float(total_pnl),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "max_drawdown": float(mdd),
            "equity_start": float(self.equity_start),
            "equity_end": float(ending_equity),
        }


def run_backtest(
    *,
    data_dir: Path,
    output_dir: Path,
    symbols: list[str] | None = None,
    max_sessions: int | None = None,
    cfg_mod=cfg,
) -> tuple[Path, Path]:
    sym_set = {s.strip().upper() for s in symbols if s.strip()} if symbols else None
    files = _discover_files(data_dir=data_dir, symbols=sym_set)
    bt = SkimmerBacktester(cfg_mod=cfg_mod)

    session_count = 0
    for sym, path, _d in files:
        bars = _normalize_bars(pd.read_csv(path))
        if bars.empty:
            continue
        bt.run_session(sym, bars)
        session_count += 1
        if max_sessions is not None and session_count >= int(max_sessions):
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "skimmer_backtest_trades.csv"
    summary_path = output_dir / "skimmer_backtest_summary.json"

    with trades_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "symbol",
                "trade_id",
                "action",
                "qty",
                "price",
                "reason",
                "pnl",
                "equity",
                "confluence_score",
            ],
        )
        writer.writeheader()
        for row in bt.trade_logs:
            writer.writerow(row)

    summary = bt._summary(session_count=session_count, file_count=len(files))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return trades_path, summary_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest AION day_skimmer on local 1-minute CSV history.")
    p.add_argument("--data-dir", default=str(Path(cfg.AION_HOME) / "data" / "intraday"))
    p.add_argument("--output-dir", default=str(Path(cfg.STATE_DIR)))
    p.add_argument("--symbols", default="", help="Comma-separated symbols (default: all files found).")
    p.add_argument("--max-sessions", type=int, default=0, help="Optional max sessions to process.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    symbols = [x.strip().upper() for x in str(args.symbols).split(",") if x.strip()]
    max_sessions = int(args.max_sessions) if int(args.max_sessions) > 0 else None
    trades_path, summary_path = run_backtest(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        symbols=symbols,
        max_sessions=max_sessions,
        cfg_mod=cfg,
    )
    print(f"[skimmer-backtest] trades={trades_path}")
    print(f"[skimmer-backtest] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
