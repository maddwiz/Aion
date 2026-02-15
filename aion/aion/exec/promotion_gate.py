import datetime as dt
import math
import json
from collections import defaultdict

from .. import config as cfg

PERF = cfg.LOG_DIR / "performance_report.json"
WF = cfg.LOG_DIR / "walkforward_results.json"
MON = cfg.LOG_DIR / "runtime_monitor.json"
TRADES = cfg.LOG_DIR / "shadow_trades.csv"
OUT = cfg.STATE_DIR / "live_promotion.json"


def _load_json(path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _slippage_stats(monitor_payload: dict):
    if not isinstance(monitor_payload, dict):
        return {"samples": 0, "avg_bps": None, "p90_bps": None}
    points = monitor_payload.get("slippage_points", [])
    vals = []
    if isinstance(points, list):
        for x in points:
            v = _safe_float(x, None)
            if v is not None:
                vals.append(float(v))
    if not vals:
        return {"samples": 0, "avg_bps": None, "p90_bps": None}
    vals = sorted(vals)
    n = len(vals)
    avg = sum(vals) / max(1, n)
    pidx = int(round(0.90 * (n - 1)))
    p90 = vals[max(0, min(n - 1, pidx))]
    return {"samples": int(n), "avg_bps": float(avg), "p90_bps": float(p90)}


def _closed_trades_per_day(path):
    if not path.exists():
        return None
    by_day = defaultdict(int)
    try:
        rows = path.read_text().splitlines()
    except Exception:
        return None
    if not rows:
        return None
    header = [h.strip().lower() for h in rows[0].split(",")]
    idx_ts = header.index("timestamp") if "timestamp" in header else -1
    idx_side = header.index("side") if "side" in header else -1
    if idx_ts < 0 or idx_side < 0:
        return None
    for ln in rows[1:]:
        parts = ln.split(",")
        if len(parts) <= max(idx_ts, idx_side):
            continue
        side = str(parts[idx_side]).strip().upper()
        if not (side.startswith("EXIT") or side.startswith("PARTIAL")):
            continue
        ts_raw = str(parts[idx_ts]).strip()
        if not ts_raw:
            continue
        try:
            d = dt.datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            try:
                d = dt.datetime.strptime(ts_raw[:19], "%Y-%m-%d %H:%M:%S").date().isoformat()
            except Exception:
                continue
        by_day[d] += 1
    if not by_day:
        return None
    vals = list(by_day.values())
    return float(sum(vals) / max(1, len(vals)))


def main() -> int:
    perf = _load_json(PERF)
    wf = _load_json(WF)
    mon = _load_json(MON)

    t = perf.get("trade_metrics", {})
    e = perf.get("equity_metrics", {})
    s = wf.get("summary", {})

    reasons = []

    trades = int(t.get("closed_trades", 0))
    winrate = float(t.get("winrate", 0.0))
    profit_factor = float(t.get("profit_factor", 0.0))
    max_dd = float(e.get("max_drawdown", 1.0))
    wf_avg = float(s.get("avg_symbol_test_pnl", -1.0))
    slip = _slippage_stats(mon)
    avg_slip = _safe_float(slip.get("avg_bps"), None)
    p90_slip = _safe_float(slip.get("p90_bps"), None)
    slip_samples = int(slip.get("samples", 0))
    closed_per_day = _closed_trades_per_day(TRADES)

    if trades < cfg.PROMOTION_MIN_TRADES:
        reasons.append(f"Need >= {cfg.PROMOTION_MIN_TRADES} closed trades (have {trades}).")
    if winrate < cfg.PROMOTION_MIN_WINRATE:
        reasons.append(f"Winrate below threshold ({winrate:.2%} < {cfg.PROMOTION_MIN_WINRATE:.2%}).")
    if profit_factor < cfg.PROMOTION_MIN_PROFIT_FACTOR:
        reasons.append(
            f"Profit factor below threshold ({profit_factor:.2f} < {cfg.PROMOTION_MIN_PROFIT_FACTOR:.2f})."
        )
    if max_dd > cfg.PROMOTION_MAX_DRAWDOWN:
        reasons.append(f"Max drawdown too high ({max_dd:.2%} > {cfg.PROMOTION_MAX_DRAWDOWN:.2%}).")
    if wf_avg < cfg.PROMOTION_MIN_WF_AVG_PNL:
        reasons.append(
            f"Walk-forward avg pnl below threshold ({wf_avg:.4f} < {cfg.PROMOTION_MIN_WF_AVG_PNL:.4f})."
        )
    if slip_samples < cfg.PROMOTION_MIN_SLIPPAGE_SAMPLES:
        reasons.append(
            f"Need >= {cfg.PROMOTION_MIN_SLIPPAGE_SAMPLES} slippage samples (have {slip_samples})."
        )
    if (avg_slip is not None) and avg_slip > cfg.PROMOTION_MAX_AVG_SLIPPAGE_BPS:
        reasons.append(
            f"Avg slippage too high ({avg_slip:.2f} bps > {cfg.PROMOTION_MAX_AVG_SLIPPAGE_BPS:.2f} bps)."
        )
    if (p90_slip is not None) and p90_slip > cfg.PROMOTION_MAX_P90_SLIPPAGE_BPS:
        reasons.append(
            f"Slippage p90 too high ({p90_slip:.2f} bps > {cfg.PROMOTION_MAX_P90_SLIPPAGE_BPS:.2f} bps)."
        )
    if (closed_per_day is not None) and closed_per_day > cfg.PROMOTION_MAX_CLOSED_TRADES_PER_DAY:
        reasons.append(
            f"Closed-trade cadence too high ({closed_per_day:.2f}/day > {cfg.PROMOTION_MAX_CLOSED_TRADES_PER_DAY:.2f}/day)."
        )

    approved = len(reasons) == 0

    payload = {
        "generated_at": dt.datetime.now().isoformat(),
        "approved": approved,
        "checks": {
            "closed_trades": trades,
            "winrate": winrate,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "wf_avg_symbol_test_pnl": wf_avg,
            "slippage_samples": slip_samples,
            "avg_slippage_bps": avg_slip,
            "p90_slippage_bps": p90_slip,
            "closed_trades_per_day": closed_per_day,
        },
        "thresholds": {
            "min_trades": cfg.PROMOTION_MIN_TRADES,
            "min_winrate": cfg.PROMOTION_MIN_WINRATE,
            "min_profit_factor": cfg.PROMOTION_MIN_PROFIT_FACTOR,
            "max_drawdown": cfg.PROMOTION_MAX_DRAWDOWN,
            "min_wf_avg_pnl": cfg.PROMOTION_MIN_WF_AVG_PNL,
            "min_slippage_samples": cfg.PROMOTION_MIN_SLIPPAGE_SAMPLES,
            "max_avg_slippage_bps": cfg.PROMOTION_MAX_AVG_SLIPPAGE_BPS,
            "max_p90_slippage_bps": cfg.PROMOTION_MAX_P90_SLIPPAGE_BPS,
            "max_closed_trades_per_day": cfg.PROMOTION_MAX_CLOSED_TRADES_PER_DAY,
        },
        "reasons": reasons,
    }

    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
