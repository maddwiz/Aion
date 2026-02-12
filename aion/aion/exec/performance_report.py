import datetime as dt
import json
from pathlib import Path

import pandas as pd

from .. import config as cfg
from ..utils.logging_utils import PERF, write_json

TRADES = cfg.LOG_DIR / "shadow_trades.csv"
EQUITY = cfg.LOG_DIR / "shadow_equity.csv"
REPORT_MD = cfg.LOG_DIR / "performance_report.md"
PROFILE = cfg.STATE_DIR / "strategy_profile.json"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, 1e-9)
    return float(dd.max())


def _trade_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "closed_trades": 0,
            "winrate": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0

    return {
        "closed_trades": int(len(df)),
        "winrate": float((pnl > 0).mean()) if len(df) else 0.0,
        "expectancy": float(pnl.mean()) if len(df) else 0.0,
        "profit_factor": float(profit_factor) if profit_factor != float("inf") else 999.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": float(pnl.sum()),
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
    }


def _format_md(payload: dict) -> str:
    t = payload["trade_metrics"]
    e = payload["equity_metrics"]
    reasons = payload.get("reason_feedback", {})
    top_lines = []
    for item in reasons.get("top_multipliers", [])[:6]:
        top_lines.append(
            f"- {item['reason']}: mult={item['multiplier']:.2f} | samples={item['samples']} | win={item['winrate']:.2%}"
        )
    if not top_lines:
        top_lines = ["- No qualified reason feedback yet."]
    return "\n".join(
        [
            "# AION Performance Report",
            "",
            f"Generated: {payload['generated_at']}",
            "",
            "## Trades",
            f"- Closed trades: {t['closed_trades']}",
            f"- Win rate: {t['winrate']:.2%}",
            f"- Expectancy/trade: {t['expectancy']:.2f}",
            f"- Profit factor: {t['profit_factor']:.2f}",
            f"- Net PnL: {t['net_pnl']:.2f}",
            f"- Avg win: {t['avg_win']:.2f}",
            f"- Avg loss: {t['avg_loss']:.2f}",
            "",
            "## Equity",
            f"- Start equity: {e['start_equity']:.2f}",
            f"- End equity: {e['end_equity']:.2f}",
            f"- Return: {e['return_pct']:.2%}",
            f"- Max drawdown: {e['max_drawdown']:.2%}",
            "",
            "## Adaptive Reasons",
            f"- Qualified reasons: {reasons.get('qualified_reasons', 0)}",
            *top_lines,
            "",
        ]
    )


def main() -> int:
    trades = _safe_read_csv(TRADES)
    equity = _safe_read_csv(EQUITY)

    if not trades.empty:
        trades["timestamp"] = pd.to_datetime(trades.get("timestamp"), errors="coerce")
        sides = trades.get("side", "").astype(str)
        closed = trades[sides.str.startswith("EXIT") | sides.str.startswith("PARTIAL")].copy()
    else:
        closed = pd.DataFrame()

    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity.get("timestamp"), errors="coerce")
        eq_col = pd.to_numeric(equity.get("equity"), errors="coerce").dropna()
    else:
        eq_col = pd.Series(dtype=float)

    trade_metrics = _trade_metrics(closed)

    reason_feedback = {"qualified_reasons": 0, "top_multipliers": []}
    if PROFILE.exists():
        try:
            profile = json.loads(PROFILE.read_text())
            stats = profile.get("reason_stats", {})
            if isinstance(stats, dict) and stats:
                rows = []
                for reason, st in stats.items():
                    if not isinstance(st, dict):
                        continue
                    rows.append(
                        {
                            "reason": str(reason),
                            "multiplier": float(st.get("multiplier", 1.0)),
                            "samples": int(st.get("samples", 0)),
                            "winrate": float(st.get("winrate", 0.0)),
                        }
                    )
                rows.sort(key=lambda r: abs(r["multiplier"] - 1.0), reverse=True)
                reason_feedback["qualified_reasons"] = len(rows)
                reason_feedback["top_multipliers"] = rows[:10]
        except Exception:
            pass

    if not eq_col.empty:
        start_equity = float(eq_col.iloc[0])
        end_equity = float(eq_col.iloc[-1])
        return_pct = (end_equity - start_equity) / max(start_equity, 1e-9)
        mdd = _max_drawdown(eq_col)
    else:
        start_equity = 0.0
        end_equity = 0.0
        return_pct = 0.0
        mdd = 0.0

    payload = {
        "generated_at": dt.datetime.now().isoformat(),
        "trade_metrics": trade_metrics,
        "equity_metrics": {
            "start_equity": start_equity,
            "end_equity": end_equity,
            "return_pct": float(return_pct),
            "max_drawdown": float(mdd),
        },
        "reason_feedback": reason_feedback,
    }

    write_json(PERF, payload)
    REPORT_MD.write_text(_format_md(payload))
    print(f"Report written: {PERF}")
    print(f"Markdown written: {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
