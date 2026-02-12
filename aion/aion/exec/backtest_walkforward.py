import datetime as dt
import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as cfg
from ..brain.signals import build_trade_signal, compute_features, opposite_confidence
from ..data.ib_client import disconnect, hist_bars, ib
from ..utils.logging_utils import BACKTEST, log_run, write_json


@dataclass
class Params:
    threshold: float
    stop_mult: float
    target_mult: float


def load_symbols() -> list[str]:
    watchlist = cfg.STATE_DIR / "watchlist.txt"
    if watchlist.exists():
        syms = [s.strip().upper() for s in watchlist.read_text().splitlines() if s.strip()]
        if syms:
            return syms[: cfg.WF_MAX_SYMBOLS]

    symbols = []
    for name in ["dow30.txt", "sp500.txt", "nasdaq100.txt"]:
        path = cfg.UNIVERSE_DIR / name
        if path.exists():
            symbols.extend([s.strip().upper() for s in path.read_text().splitlines() if s.strip()])

    uniq = []
    seen = set()
    for sym in symbols:
        if sym not in seen:
            uniq.append(sym)
            seen.add(sym)
    return uniq[: cfg.WF_MAX_SYMBOLS]


def max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / max(peak, 1e-9)
        worst = max(worst, dd)
    return worst


def _metrics(trade_returns: list[float], curve: list[float]):
    trades = len(trade_returns)
    wins = sum(1 for x in trade_returns if x > 0)
    losses = sum(1 for x in trade_returns if x < 0)
    winrate = wins / trades if trades else 0.0
    expectancy = float(np.mean(trade_returns)) if trades else 0.0
    pnl = float(np.sum(trade_returns)) if trades else 0.0
    std = float(np.std(trade_returns)) if trades else 0.0
    sharpe_like = (expectancy / (std + 1e-9)) * np.sqrt(max(trades, 1))
    mdd = max_drawdown(curve)
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "expectancy": expectancy,
        "pnl": pnl,
        "sharpe_like": float(sharpe_like),
        "max_drawdown": mdd,
        "final_equity": curve[-1] if curve else 1.0,
    }


def simulate_window(df: pd.DataFrame, params: Params):
    if len(df) < cfg.MIN_BARS:
        return _metrics([], [1.0])

    feats = compute_features(df, cfg)
    position = None
    curve = [1.0]
    trade_returns = []

    profile = {
        "entry_threshold_long": params.threshold,
        "entry_threshold_short": params.threshold,
        "opposite_exit_threshold": min(0.90, params.threshold + 0.03),
    }

    for i in range(cfg.MIN_BARS, len(feats)):
        row = feats.iloc[i]
        price = float(row["close"])
        atr = float(row["atr"])
        if not np.isfinite(price) or not np.isfinite(atr) or atr <= 0:
            curve.append(curve[-1])
            continue

        slice_start = max(0, i - cfg.SWING_LOOKBACK)
        high = float(df["high"].iloc[slice_start:i].max())
        low = float(df["low"].iloc[slice_start:i].min())
        signal = build_trade_signal(row, price, high, low, cfg, profile=profile)

        if position is not None:
            position["bars_held"] += 1
            side = position["side"]

            if side == "LONG":
                position["peak"] = max(position["peak"], price)
                position["trail"] = max(position["trail"], position["peak"] - atr * position["trail_mult"])
                stop_hit = price <= max(position["stop"], position["trail"])
                target_hit = price >= position["target"]
            else:
                position["trough"] = min(position["trough"], price)
                position["trail"] = min(position["trail"], position["trough"] + atr * position["trail_mult"])
                stop_hit = price >= min(position["stop"], position["trail"])
                target_hit = price <= position["target"]

            opposite = opposite_confidence(side, signal) >= signal["opposite_exit_threshold"]
            timeout = position["bars_held"] >= cfg.MAX_HOLD_CYCLES

            if stop_hit or target_hit or opposite or timeout:
                if side == "LONG":
                    ret = (price - position["entry"]) / max(1e-9, position["entry"])
                else:
                    ret = (position["entry"] - price) / max(1e-9, position["entry"])
                trade_returns.append(ret)
                curve.append(curve[-1] * (1.0 + ret))
                position = None
                continue

        if position is None and signal["side"] is not None:
            stop_dist = max(atr * params.stop_mult, price * 0.0035)
            if signal["side"] == "LONG":
                position = {
                    "side": "LONG",
                    "entry": price,
                    "stop": price - stop_dist,
                    "target": price + (atr * params.target_mult),
                    "trail": price - stop_dist,
                    "trail_mult": cfg.TRAIL_ATR_MULT,
                    "peak": price,
                    "bars_held": 0,
                }
            else:
                position = {
                    "side": "SHORT",
                    "entry": price,
                    "stop": price + stop_dist,
                    "target": price - (atr * params.target_mult),
                    "trail": price + stop_dist,
                    "trail_mult": cfg.TRAIL_ATR_MULT,
                    "trough": price,
                    "bars_held": 0,
                }

        curve.append(curve[-1])

    if position is not None:
        final_price = float(df["close"].iloc[-1])
        if position["side"] == "LONG":
            ret = (final_price - position["entry"]) / max(1e-9, position["entry"])
        else:
            ret = (position["entry"] - final_price) / max(1e-9, position["entry"])
        trade_returns.append(ret)
        curve.append(curve[-1] * (1.0 + ret))

    return _metrics(trade_returns, curve)


def _score_metrics(m: dict) -> float:
    return (
        m["pnl"]
        + 0.30 * m["sharpe_like"]
        + 0.20 * m["winrate"]
        - 0.75 * m["max_drawdown"]
    )


def _neighbor_param_grid(best: Params):
    ts = sorted(set([max(0.50, best.threshold - 0.02), best.threshold, min(0.80, best.threshold + 0.02)]))
    ss = sorted(set([max(0.8, best.stop_mult - 0.2), best.stop_mult, min(2.5, best.stop_mult + 0.2)]))
    tg = sorted(set([max(1.4, best.target_mult - 0.4), best.target_mult, min(5.0, best.target_mult + 0.4)]))
    return [Params(a, b, c) for a, b, c in itertools.product(ts, ss, tg)]


def stability_test(train_df: pd.DataFrame, best: Params):
    scores = []
    params = _neighbor_param_grid(best)
    for p in params:
        m = simulate_window(train_df, p)
        scores.append(_score_metrics(m))

    if not scores:
        return {
            "stability_score": 0.0,
            "mean_neighbor_score": 0.0,
            "std_neighbor_score": 0.0,
            "passed": False,
        }

    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    best_s = max(scores)
    stability = mean_s / (abs(best_s) + 1e-9)
    penalty = std_s / (abs(mean_s) + 1e-9)
    stability_score = max(0.0, min(1.0, 0.7 * stability + 0.3 * (1.0 - min(1.0, penalty))))

    return {
        "stability_score": float(stability_score),
        "mean_neighbor_score": mean_s,
        "std_neighbor_score": std_s,
        "passed": bool(stability_score >= cfg.WF_STABILITY_MIN_SCORE),
    }


def select_params(train_df: pd.DataFrame) -> tuple[Params, dict, dict]:
    best = None
    best_score = -1e9
    best_metrics = None

    for threshold, stop_mult, target_mult in itertools.product(
        cfg.WF_THRESHOLDS, cfg.WF_STOP_MULTS, cfg.WF_TARGET_MULTS
    ):
        p = Params(threshold=threshold, stop_mult=stop_mult, target_mult=target_mult)
        m = simulate_window(train_df, p)
        score = _score_metrics(m)
        if score > best_score:
            best = p
            best_score = score
            best_metrics = m

    stab = stability_test(train_df, best)
    return best, best_metrics, stab


def walkforward_symbol(df: pd.DataFrame):
    train = cfg.WALKFORWARD_TRAIN_BARS
    test = cfg.WALKFORWARD_TEST_BARS
    step = cfg.WALKFORWARD_STEP_BARS

    out = []
    start = 0
    while start + train + test <= len(df):
        train_df = df.iloc[start : start + train].copy()
        test_df = df.iloc[start + train : start + train + test].copy()

        params, train_metrics, stability = select_params(train_df)
        test_metrics = simulate_window(test_df, params)

        out.append(
            {
                "window_start": int(start),
                "window_end": int(start + train + test),
                "params": {
                    "threshold": params.threshold,
                    "stop_mult": params.stop_mult,
                    "target_mult": params.target_mult,
                },
                "train": train_metrics,
                "test": test_metrics,
                "stability": stability,
            }
        )
        start += step

    return out


def aggregate_test_metrics(windows: list[dict]):
    test_returns = [w["test"]["pnl"] for w in windows]
    stability_scores = [float(w.get("stability", {}).get("stability_score", 0.0)) for w in windows]
    stable_windows = sum(1 for w in windows if w.get("stability", {}).get("passed"))

    if not test_returns:
        return {
            "windows": 0,
            "avg_test_pnl": 0.0,
            "avg_stability": 0.0,
            "stable_windows": 0,
        }

    return {
        "windows": len(test_returns),
        "avg_test_pnl": float(np.mean(test_returns)),
        "median_test_pnl": float(np.median(test_returns)),
        "positive_windows": int(sum(1 for x in test_returns if x > 0)),
        "avg_stability": float(np.mean(stability_scores)) if stability_scores else 0.0,
        "stable_windows": int(stable_windows),
    }


def main() -> int:
    try:
        ib()
    except Exception as exc:
        print(f"[AION] ERROR: Unable to connect to IBKR ({cfg.IB_HOST}:{cfg.IB_PORT}): {exc}")
        return 1

    symbols = load_symbols()
    run_ts = dt.datetime.now().isoformat()
    payload = {
        "run_ts": run_ts,
        "duration": cfg.WALKFORWARD_DURATION,
        "bar_size": cfg.WALKFORWARD_BAR_SIZE,
        "train_bars": cfg.WALKFORWARD_TRAIN_BARS,
        "test_bars": cfg.WALKFORWARD_TEST_BARS,
        "step_bars": cfg.WALKFORWARD_STEP_BARS,
        "symbols": {},
        "summary": {},
    }

    symbol_summaries = []
    stability_summaries = []
    for sym in symbols:
        df = hist_bars(sym, duration=cfg.WALKFORWARD_DURATION, barSize=cfg.WALKFORWARD_BAR_SIZE)
        if df.empty or len(df) < (cfg.WALKFORWARD_TRAIN_BARS + cfg.WALKFORWARD_TEST_BARS):
            continue

        windows = walkforward_symbol(df)
        agg = aggregate_test_metrics(windows)
        payload["symbols"][sym] = {
            "bars": len(df),
            "windows": windows,
            "aggregate": agg,
        }
        symbol_summaries.append(agg.get("avg_test_pnl", 0.0))
        stability_summaries.append(agg.get("avg_stability", 0.0))

    payload["summary"] = {
        "symbols_tested": len(payload["symbols"]),
        "avg_symbol_test_pnl": float(np.mean(symbol_summaries)) if symbol_summaries else 0.0,
        "median_symbol_test_pnl": float(np.median(symbol_summaries)) if symbol_summaries else 0.0,
        "avg_stability": float(np.mean(stability_summaries)) if stability_summaries else 0.0,
    }

    write_json(BACKTEST, payload)
    log_run(f"Walk-forward backtest complete: {BACKTEST}")
    print(json.dumps(payload["summary"], indent=2))

    disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
