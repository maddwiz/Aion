import csv
import datetime as dt
import json
from pathlib import Path
from shutil import copy2

from ..config import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

TRADES = LOG_DIR / "shadow_trades.csv"
EQUITY = LOG_DIR / "shadow_equity.csv"
RUNLOG = LOG_DIR / "aion_run.log"
SCANLOG = LOG_DIR / "universe_scan.log"
SIGNALS = LOG_DIR / "signals.csv"
ALERTS = LOG_DIR / "alerts.log"
BACKTEST = LOG_DIR / "walkforward_results.json"
PERF = LOG_DIR / "performance_report.json"


def _ensure_csv_header(path: Path, header: str):
    if not path.exists():
        path.write_text(header + "\n")
        return

    try:
        lines = path.read_text().splitlines()
        first = lines[0] if lines else ""
    except Exception:
        first = ""

    if first == header:
        return

    # Preserve old file before schema migration.
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{ts}")
    try:
        copy2(path, backup)
    except Exception:
        pass
    path.write_text(header + "\n")


def _ensure_headers():
    _ensure_csv_header(
        TRADES,
        "timestamp,symbol,side,qty,entry,exit,pnl,reason,confidence,regime,stop,target,trail,fill_ratio,slippage_bps",
    )
    _ensure_csv_header(EQUITY, "timestamp,equity,cash,open_pnl,closed_pnl")
    if not RUNLOG.exists():
        RUNLOG.write_text("")
    if not SCANLOG.exists():
        SCANLOG.write_text("")
    _ensure_csv_header(
        SIGNALS,
        "timestamp,symbol,regime,long_conf,short_conf,decision,meta_prob,mtf_score,pattern_hits,indicator_hits,reasons",
    )
    if not ALERTS.exists():
        ALERTS.write_text("")


_ensure_headers()


def log_trade(
    ts: str,
    symbol: str,
    side: str,
    qty: int,
    entry: float,
    exit_: float,
    pnl: float,
    reason: str,
    confidence: float = 0.0,
    regime: str = "",
    stop: float = 0.0,
    target: float = 0.0,
    trail: float = 0.0,
    fill_ratio: float = 1.0,
    slippage_bps: float = 0.0,
):
    with TRADES.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                ts,
                symbol,
                side,
                qty,
                f"{entry:.4f}",
                f"{exit_:.4f}",
                f"{pnl:.2f}",
                reason or "",
                f"{confidence:.4f}",
                regime,
                f"{stop:.4f}",
                f"{target:.4f}",
                f"{trail:.4f}",
                f"{fill_ratio:.4f}",
                f"{slippage_bps:.4f}",
            ]
        )


def log_equity(ts: str, equity: float, cash: float, open_pnl: float, closed_pnl: float):
    with EQUITY.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([ts, f"{equity:.2f}", f"{cash:.2f}", f"{open_pnl:.2f}", f"{closed_pnl:.2f}"])


def log_run(msg: str):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with RUNLOG.open("a") as f:
        f.write(f"[{ts}] {msg}\n")


def log_scan(msg: str):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with SCANLOG.open("a") as f:
        f.write(f"[{ts}] {msg}\n")


def log_alert(msg: str):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with ALERTS.open("a") as f:
        f.write(f"[{ts}] {msg}\n")


def log_signal(
    ts: str,
    symbol: str,
    regime: str,
    long_conf: float,
    short_conf: float,
    decision: str,
    reasons: list[str],
    meta_prob: float = 0.0,
    mtf_score: float = 0.0,
    pattern_hits: int = 0,
    indicator_hits: int = 0,
):
    with SIGNALS.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                ts,
                symbol,
                regime,
                f"{long_conf:.4f}",
                f"{short_conf:.4f}",
                decision,
                f"{meta_prob:.4f}",
                f"{mtf_score:.4f}",
                str(int(pattern_hits)),
                str(int(indicator_hits)),
                " | ".join(reasons),
            ]
        )


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, default=str))
