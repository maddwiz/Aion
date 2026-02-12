import datetime as dt
import json

from .. import config as cfg

PERF = cfg.LOG_DIR / "performance_report.json"
WF = cfg.LOG_DIR / "walkforward_results.json"
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


def main() -> int:
    perf = _load_json(PERF)
    wf = _load_json(WF)

    t = perf.get("trade_metrics", {})
    e = perf.get("equity_metrics", {})
    s = wf.get("summary", {})

    reasons = []

    trades = int(t.get("closed_trades", 0))
    winrate = float(t.get("winrate", 0.0))
    profit_factor = float(t.get("profit_factor", 0.0))
    max_dd = float(e.get("max_drawdown", 1.0))
    wf_avg = float(s.get("avg_symbol_test_pnl", -1.0))

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
        },
        "thresholds": {
            "min_trades": cfg.PROMOTION_MIN_TRADES,
            "min_winrate": cfg.PROMOTION_MIN_WINRATE,
            "min_profit_factor": cfg.PROMOTION_MIN_PROFIT_FACTOR,
            "max_drawdown": cfg.PROMOTION_MAX_DRAWDOWN,
            "min_wf_avg_pnl": cfg.PROMOTION_MIN_WF_AVG_PNL,
        },
        "reasons": reasons,
    }

    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
