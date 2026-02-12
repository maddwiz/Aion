from ib_insync import *
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


AION_HOME = Path(os.getenv("AION_HOME", Path(__file__).resolve().parents[1]))
UNI = Path(os.getenv("AION_UNIVERSE_DIR", Path(__file__).resolve().parent / "universe"))
STATE = Path(os.getenv("AION_STATE_DIR", AION_HOME / "state"))
LOGS = Path(os.getenv("AION_LOG_DIR", AION_HOME / "logs"))
STATE.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("AION_SCANNER_CLIENT_ID", "4201"))

BAR_SIZE = os.getenv("AION_SCAN_BAR_SIZE", "15 mins")
DURATION = os.getenv("AION_SCAN_DURATION", "3 D")
WHAT_PRIMARY = os.getenv("AION_SCAN_WHAT_PRIMARY", "TRADES")
WHAT_FALLBACK = os.getenv("AION_SCAN_WHAT_FALLBACK", "MIDPOINT")
USE_RTH = os.getenv("AION_SCAN_USE_RTH", "true").lower() in {"1", "true", "yes", "on"}
SCAN_LIMIT = int(os.getenv("AION_SCAN_LIMIT", "300"))
SHORTLIST_SIZE = int(os.getenv("AION_SHORTLIST_SIZE", "30"))
PAUSE = float(os.getenv("AION_SCAN_PAUSE", "0.25"))

PREF_EX = ("NASDAQ", "NYSE", "ARCA", "AMEX")


def now_ts():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_universe():
    files = ["dow30.txt", "sp500.txt", "nasdaq100.txt"]
    symbols = []
    for f in files:
        path = UNI / f
        if path.exists():
            for line in path.read_text().splitlines():
                sym = line.strip().upper()
                if sym:
                    symbols.append(sym)
    seen = set()
    out = []
    for sym in symbols:
        if sym not in seen:
            out.append(sym)
            seen.add(sym)
    return out[:SCAN_LIMIT]


def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def rsi(series, n=14):
    diff = series.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    rs = up.ewm(alpha=1 / n, min_periods=n).mean() / down.ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + rs)


def recent_swings(close, look=40):
    if len(close) < look + 1:
        return None
    sw_hi = close[-look:].max()
    sw_lo = close[-look:].min()
    if sw_hi == sw_lo:
        return None
    diff = sw_hi - sw_lo
    fib382 = sw_hi - 0.382 * diff
    fib618 = sw_hi - 0.618 * diff
    last = close.iloc[-1]
    prox382 = abs(last - fib382) / max(1e-6, last)
    prox618 = abs(last - fib618) / max(1e-6, last)
    return {
        "last": float(last),
        "hi": float(sw_hi),
        "lo": float(sw_lo),
        "fib382": float(fib382),
        "fib618": float(fib618),
        "prox382": float(prox382),
        "prox618": float(prox618),
    }


def score_symbol(df: pd.DataFrame):
    if len(df) < 50:
        return None
    close = df["close"]
    ema_f = ema(close, 12)
    ema_s = ema(close, 26)
    trend = 1.0 if float(ema_f.iloc[-1]) > float(ema_s.iloc[-1]) else 0.0
    r = float(rsi(close, 14).iloc[-1])
    rsi_ok = 1.0 if 40 <= r <= 70 else 0.0
    sw = recent_swings(close, look=40)
    if sw is None:
        return None
    prox = min(sw["prox382"], sw["prox618"])
    prox_score = max(0.0, 1.0 - (prox / 0.01))
    score = 0.45 * trend + 0.35 * prox_score + 0.20 * rsi_ok
    return float(score), {"ind_rsi": r, **sw}


def _split_exchanges(valid_exchanges: str):
    if not valid_exchanges:
        return []
    return [e.strip().upper() for e in valid_exchanges.split(",") if e.strip()]


def qualify_stock(ib: IB, sym: str):
    try:
        cds = ib.reqContractDetails(Stock(sym, "SMART", "USD"))
    except Exception:
        cds = []
    if not cds:
        return None

    exact = [
        cd
        for cd in cds
        if getattr(cd, "contract", None)
        and str(cd.contract.symbol).upper() == sym.upper()
        and getattr(cd.contract, "secType", "") == "STK"
        and getattr(cd.contract, "currency", "") == "USD"
    ]
    if not exact:
        return None

    def rank(cd: ContractDetails):
        c = cd.contract
        pe = (getattr(c, "primaryExchange", "") or "").upper()
        vex = _split_exchanges(getattr(cd, "validExchanges", ""))
        score = 0
        if any(e in pe for e in PREF_EX):
            score += 120
        if any(e in vex for e in PREF_EX):
            score += 80
        if "NASDAQ" in pe or "NASDAQ" in vex:
            score += 30
        if "NYSE" in pe or "NYSE" in vex:
            score += 25
        if "ARCA" in pe or "ARCA" in vex:
            score += 15
        return score

    exact.sort(key=rank, reverse=True)
    best = exact[0]

    c0 = best.contract
    final = Contract()
    final.conId = c0.conId
    final.symbol = c0.symbol
    final.secType = "STK"
    final.currency = "USD"
    final.exchange = "SMART"

    pe = (getattr(c0, "primaryExchange", "") or "").upper()
    vex = _split_exchanges(getattr(best, "validExchanges", ""))
    if pe and any(e == pe for e in PREF_EX):
        final.primaryExchange = pe
    else:
        for e in PREF_EX:
            if e in vex:
                final.primaryExchange = e
                break

    try:
        qualified = ib.qualifyContracts(final)
        return qualified[0] if qualified else None
    except Exception:
        return None


def fetch_bars_one(ib: IB, contract: Contract, what: str):
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow=what,
        useRTH=USE_RTH,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return None
    df = util.df(bars)
    if df is None or df.empty:
        return None
    return df[["date", "open", "high", "low", "close"]].rename(columns={"date": "ts"})


def fetch_bars(ib: IB, sym: str):
    contract = qualify_stock(ib, sym)
    if not contract:
        return None
    for what in (WHAT_PRIMARY, WHAT_FALLBACK):
        try:
            df = fetch_bars_one(ib, contract, what)
            if df is not None:
                return df
        except Exception:
            continue
    return None


def main():
    ib = IB()
    if not ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=20):
        print(f"[{now_ts()}] ERROR: cannot connect IB Gateway on {HOST}:{PORT}")
        return

    ib.reqMarketDataType(3)

    universe = load_universe()
    results = []
    for sym in universe:
        df = fetch_bars(ib, sym)
        time.sleep(PAUSE)
        if df is None or df.empty:
            continue
        scored = score_symbol(df)
        if scored is None:
            continue
        score, info = scored
        results.append({"symbol": sym, "score": score, "info": info})

    results.sort(key=lambda x: x["score"], reverse=True)
    shortlist = [r["symbol"] for r in results[:SHORTLIST_SIZE]]

    (STATE / "watchlist.txt").write_text("\n".join(shortlist) + ("\n" if shortlist else ""))
    (STATE / "watchlist.json").write_text(
        json.dumps(
            {
                "ts": now_ts(),
                "barSize": BAR_SIZE,
                "duration": DURATION,
                "useRTH": USE_RTH,
                "shortlist": results[:SHORTLIST_SIZE],
                "universe": len(universe),
            },
            indent=2,
        )
    )

    print(f"[{now_ts()}] Universe scan complete. Universe={len(universe)} Shortlist={len(shortlist)}")
    ib.disconnect()


if __name__ == "__main__":
    main()
