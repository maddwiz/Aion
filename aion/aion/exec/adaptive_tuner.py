import datetime as dt
import json
import math
from collections import defaultdict

import pandas as pd

from .. import config as cfg

TRADES = cfg.LOG_DIR / "shadow_trades.csv"
PROFILE = cfg.STATE_DIR / "strategy_profile.json"
REASON_FEEDBACK = cfg.STATE_DIR / "reason_feedback.json"


def _load_trades() -> pd.DataFrame:
    if not TRADES.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(TRADES)
    except Exception:
        return pd.DataFrame()
    if "side" not in df.columns or "symbol" not in df.columns:
        return pd.DataFrame()
    if "timestamp" not in df.columns:
        df["timestamp"] = ""
    return df


def _load_profile() -> dict:
    if PROFILE.exists():
        try:
            data = json.loads(PROFILE.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "trading_enabled": True,
        "entry_threshold_long": cfg.ENTRY_THRESHOLD_LONG,
        "entry_threshold_short": cfg.ENTRY_THRESHOLD_SHORT,
        "opposite_exit_threshold": cfg.OPPOSITE_EXIT_THRESHOLD,
        "max_trades_per_day": cfg.MAX_TRADES_PER_DAY,
    }


def _safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _extract_entry_reasons(reason_text: str) -> list[str]:
    raw = str(reason_text or "")
    if "|" in raw:
        raw = raw.split("|", 1)[1]
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    clean = []
    seen = set()
    for p in parts:
        if p.lower().startswith("adaptive reason factor"):
            continue
        if p not in seen:
            clean.append(p)
            seen.add(p)
    return clean


def _entry_side_to_pos(side: str):
    if side == "ENTRY_BUY":
        return "LONG"
    if side == "ENTRY_SELL":
        return "SHORT"
    return None


def _close_side_to_pos(side: str):
    if side in {"PARTIAL_SELL", "EXIT_SELL"}:
        return "LONG"
    if side in {"PARTIAL_BUY", "EXIT_BUY"}:
        return "SHORT"
    return None


def _reconstruct_closed_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.sort_values(["timestamp"]).reset_index(drop=True)

    open_pos = {}
    closed = []

    for _, row in work.iterrows():
        action = str(row.get("side", ""))
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue

        qty = max(0, _safe_int(row.get("qty"), 0))
        pnl = _safe_float(row.get("pnl"), 0.0)
        ts = row.get("timestamp")

        pos_side = _entry_side_to_pos(action)
        if pos_side:
            open_pos[symbol] = {
                "symbol": symbol,
                "side": pos_side,
                "qty_open": max(1, qty),
                "realized_pnl": 0.0,
                "reasons": _extract_entry_reasons(str(row.get("reason", ""))),
                "regime": str(row.get("regime", "")),
                "entry_confidence": _safe_float(row.get("confidence"), 0.0),
                "opened_at": ts,
                "partials": 0,
            }
            continue

        pos_side = _close_side_to_pos(action)
        if not pos_side:
            continue

        ctx = open_pos.get(symbol)
        if not ctx or ctx.get("side") != pos_side:
            continue

        ctx["realized_pnl"] += pnl
        close_qty = qty if qty > 0 else ctx["qty_open"]
        ctx["qty_open"] = max(0, int(ctx["qty_open"]) - int(close_qty))

        if action.startswith("PARTIAL"):
            ctx["partials"] += 1

        if action.startswith("EXIT") or ctx["qty_open"] <= 0:
            closed.append(
                {
                    "symbol": symbol,
                    "side": ctx["side"],
                    "pnl": float(ctx["realized_pnl"]),
                    "reasons": ctx["reasons"],
                    "regime": ctx["regime"],
                    "entry_confidence": float(ctx["entry_confidence"]),
                    "opened_at": ctx["opened_at"],
                    "closed_at": ts,
                    "partials": int(ctx["partials"]),
                }
            )
            open_pos.pop(symbol, None)

    if not closed:
        return pd.DataFrame()
    return pd.DataFrame(closed)


def _build_reason_feedback(closed: pd.DataFrame) -> dict:
    if closed.empty:
        return {"multipliers": {}, "stats": {}, "selected": []}

    pnl_all = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
    avg_abs = float(pnl_all.abs().mean()) if len(pnl_all) else 1.0
    avg_abs = max(avg_abs, 1e-6)

    per_reason = defaultdict(list)
    for _, row in closed.iterrows():
        pnl = _safe_float(row.get("pnl"), 0.0)
        for reason in row.get("reasons", []):
            per_reason[str(reason)].append(pnl)

    reason_rows = []
    for reason, vals in per_reason.items():
        samples = len(vals)
        if samples <= 0:
            continue
        winrate = float(sum(1 for v in vals if v > 0) / samples)
        expectancy = float(sum(vals) / samples)
        expectancy_norm = expectancy / avg_abs
        score = 0.60 * ((winrate - 0.50) * 2.0) + 0.40 * math.tanh(expectancy_norm)
        raw_mult = 1.0 + 0.18 * score
        strength = min(1.0, samples / max(1.0, cfg.REASON_ADAPT_MIN_TRADES * 2.0))
        smooth_mult = 1.0 + (raw_mult - 1.0) * strength
        mult = _clamp(smooth_mult, cfg.REASON_ADAPT_MIN_MULT, cfg.REASON_ADAPT_MAX_MULT)
        reason_rows.append(
            {
                "reason": reason,
                "samples": samples,
                "winrate": winrate,
                "expectancy": expectancy,
                "score": score,
                "multiplier": mult,
                "impact_rank": abs(expectancy) * math.sqrt(samples),
            }
        )

    if not reason_rows:
        return {"multipliers": {}, "stats": {}, "selected": []}

    ranked = sorted(reason_rows, key=lambda r: r["impact_rank"], reverse=True)
    selected = [
        r for r in ranked
        if r["samples"] >= cfg.REASON_ADAPT_MIN_TRADES
    ][: cfg.REASON_ADAPT_MAX_REASONS]

    multipliers = {r["reason"]: round(float(r["multiplier"]), 4) for r in selected}
    stats = {
        r["reason"]: {
            "samples": int(r["samples"]),
            "winrate": round(float(r["winrate"]), 4),
            "expectancy": round(float(r["expectancy"]), 4),
            "multiplier": round(float(r["multiplier"]), 4),
        }
        for r in selected
    }

    return {
        "multipliers": multipliers,
        "stats": stats,
        "selected": [r["reason"] for r in selected],
        "coverage_reasons": len(per_reason),
        "qualified_reasons": len(selected),
    }


def main() -> int:
    profile = _load_profile()
    trades = _load_trades()
    closed = _reconstruct_closed_trades(trades)

    if closed.empty:
        profile["updated_at"] = dt.datetime.now().isoformat()
        profile["notes"] = "No reconstructable closed trades yet; profile unchanged."
        PROFILE.write_text(json.dumps(profile, indent=2))
        print("No reconstructable closed trades found. Profile unchanged.")
        return 0

    recent = closed.tail(cfg.ADAPTIVE_LOOKBACK_TRADES).copy()
    pnl = pd.to_numeric(recent["pnl"], errors="coerce").fillna(0.0)

    trade_count = len(recent)
    winrate = float((pnl > 0).mean()) if trade_count else 0.0
    expectancy = float(pnl.mean()) if trade_count else 0.0
    net = float(pnl.sum()) if trade_count else 0.0

    long_th = float(profile.get("entry_threshold_long", cfg.ENTRY_THRESHOLD_LONG))
    short_th = float(profile.get("entry_threshold_short", cfg.ENTRY_THRESHOLD_SHORT))
    max_trades = int(profile.get("max_trades_per_day", cfg.MAX_TRADES_PER_DAY))

    weak = (winrate < cfg.ADAPTIVE_WINRATE_FLOOR) or (expectancy < cfg.ADAPTIVE_EXPECTANCY_FLOOR)
    strong = (winrate > (cfg.ADAPTIVE_WINRATE_FLOOR + 0.08)) and (expectancy > cfg.ADAPTIVE_EXPECTANCY_FLOOR)

    if weak:
        long_th += cfg.ADAPTIVE_THRESHOLD_STEP
        short_th += cfg.ADAPTIVE_THRESHOLD_STEP
        max_trades = max(3, max_trades - 1)
        action = "tighten"
    elif strong:
        long_th -= cfg.ADAPTIVE_THRESHOLD_STEP * 0.5
        short_th -= cfg.ADAPTIVE_THRESHOLD_STEP * 0.5
        max_trades = min(cfg.MAX_TRADES_PER_DAY, max_trades + 1)
        action = "relax"
    else:
        action = "hold"

    long_th = _clamp(long_th, cfg.ADAPTIVE_THRESHOLD_MIN, cfg.ADAPTIVE_THRESHOLD_MAX)
    short_th = _clamp(short_th, cfg.ADAPTIVE_THRESHOLD_MIN, cfg.ADAPTIVE_THRESHOLD_MAX)

    severe = trade_count >= 30 and winrate < 0.30 and expectancy < 0 and net < 0
    trading_enabled = not severe

    reason_feedback = {"multipliers": {}, "stats": {}, "selected": []}
    if cfg.REASON_ADAPT_ENABLED:
        reason_feedback = _build_reason_feedback(recent)
        REASON_FEEDBACK.write_text(
            json.dumps(
                {
                    "updated_at": dt.datetime.now().isoformat(),
                    "lookback_trades": int(trade_count),
                    **reason_feedback,
                },
                indent=2,
            )
        )

    profile.update(
        {
            "updated_at": dt.datetime.now().isoformat(),
            "trading_enabled": trading_enabled,
            "entry_threshold_long": round(long_th, 4),
            "entry_threshold_short": round(short_th, 4),
            "opposite_exit_threshold": round(min(0.90, max(long_th, short_th) + 0.03), 4),
            "max_trades_per_day": max_trades,
            "reason_multipliers": reason_feedback.get("multipliers", {}),
            "reason_stats": reason_feedback.get("stats", {}),
            "adaptive_stats": {
                "lookback_trades": trade_count,
                "winrate": round(winrate, 4),
                "expectancy": round(expectancy, 4),
                "net_pnl": round(net, 2),
                "action": action,
                "severe_underperformance": severe,
                "reason_adapt_enabled": bool(cfg.REASON_ADAPT_ENABLED),
                "reason_qualified": int(reason_feedback.get("qualified_reasons", 0)),
            },
        }
    )

    PROFILE.write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile["adaptive_stats"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
