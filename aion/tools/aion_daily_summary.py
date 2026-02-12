import csv
import datetime as dt
import os
from pathlib import Path


AION_HOME = Path(os.getenv("AION_HOME", Path(__file__).resolve().parents[1]))
LOGDIR = Path(os.getenv("AION_LOG_DIR", AION_HOME / "logs"))
TRADES = LOGDIR / "shadow_trades.csv"
EQUITY = LOGDIR / "shadow_equity.csv"


def today_range_local():
    now = dt.datetime.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + dt.timedelta(days=1)
    return start, end


def parse_csv(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def to_dt(value):
    if not value:
        return None
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    ]
    base = value.split(".")[0]
    for fmt in fmts:
        try:
            return dt.datetime.strptime(base, fmt)
        except Exception:
            pass
    return None


def summarize():
    start, end = today_range_local()

    trades = parse_csv(TRADES)
    tcount = 0
    pnl = 0.0
    winners = 0

    for trade in trades:
        ts = to_dt(trade.get("timestamp", "") or trade.get("ts", ""))
        if ts is None or not (start <= ts < end):
            continue
        tcount += 1
        try:
            p = float(trade.get("pnl", "0") or trade.get("PnL", "0"))
        except Exception:
            p = 0.0
        pnl += p
        if p > 0:
            winners += 1

    winrate = (winners / tcount * 100.0) if tcount else 0.0

    eq_rows = parse_csv(EQUITY)
    eq_today = [
        row
        for row in eq_rows
        if (ts := to_dt(row.get("timestamp", "") or row.get("ts", ""))) is not None and (start <= ts < end)
    ]

    eq_open = None
    eq_close = None
    if eq_today:
        try:
            eq_open = float(eq_today[0].get("equity", "") or eq_today[0].get("Equity", ""))
            eq_close = float(eq_today[-1].get("equity", "") or eq_today[-1].get("Equity", ""))
        except Exception:
            pass
    eq_delta = None if (eq_open is None or eq_close is None) else (eq_close - eq_open)

    return {
        "date": dt.datetime.now().strftime("%Y-%m-%d"),
        "trades": tcount,
        "winners": winners,
        "winrate_pct": round(winrate, 2),
        "pnl": round(pnl, 2),
        "equity_change": (None if eq_delta is None else round(eq_delta, 2)),
    }


def main():
    summary = summarize()
    eq_str = "n/a" if summary["equity_change"] is None else f"${summary['equity_change']:.2f}"
    msg = (
        f"{summary['trades']} trades | {summary['winrate_pct']}% win | "
        f"PnL: ${summary['pnl']:.2f} | Equity delta: {eq_str}"
    )
    print(msg)


if __name__ == "__main__":
    main()
