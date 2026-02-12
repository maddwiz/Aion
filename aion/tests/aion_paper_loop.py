from ib_insync import *
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


AION_HOME = Path(os.getenv("AION_HOME", Path(__file__).resolve().parents[1]))
STATE_DIR = Path(os.getenv("AION_STATE_DIR", AION_HOME / "state"))
LOG_DIR = Path(os.getenv("AION_LOG_DIR", AION_HOME / "logs"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

WATCHLIST = STATE_DIR / "watchlist.txt"
TRADES_CSV = LOG_DIR / "shadow_trades.csv"
EQUITY_CSV = LOG_DIR / "shadow_equity.csv"

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("AION_CLIENT_ID", "3201"))
MARKET_DATA_TYPE = int(os.getenv("AION_MARKET_DATA_TYPE", "3"))

ACCOUNT_EQUITY = float(os.getenv("AION_ACCOUNT_EQUITY", "5000"))
RISK_PCT = float(os.getenv("AION_RISK_PCT", "0.02"))
DAILY_GOAL_USD = float(os.getenv("AION_DAILY_GOAL_USD", "100"))
TRADES_PER_DAY_CAP = int(os.getenv("AION_TRADES_PER_DAY_CAP", "10"))
BAR_SECONDS = int(os.getenv("AION_BAR_SECONDS", "60"))
LOOKBACK_MIN = int(os.getenv("AION_LOOKBACK_MIN", "120"))

DEFAULT_FX_SYMBOLS = [
    s.strip().upper()
    for s in os.getenv("AION_DEFAULT_SYMBOLS", "EURUSD,USDJPY").split(",")
    if s.strip()
]

H_TR = ["ts", "symbol", "action", "qty", "price", "stop", "target", "pnl"]
H_EQ = ["ts", "equity", "todays_pnl", "trades_today"]

ib = IB()


def _load_symbols(default_fx=None):
    if default_fx is None:
        default_fx = DEFAULT_FX_SYMBOLS
    try:
        if WATCHLIST.exists():
            syms = [s.strip().upper() for s in WATCHLIST.read_text().splitlines() if s.strip()]
            if syms:
                return syms
    except Exception:
        pass
    return list(default_fx)


def now_ts():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_row(path, row, cols):
    pd.DataFrame([row], columns=cols).to_csv(path, mode="a", index=False, header=not path.exists())


def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def rsi(series, n=14):
    diff = series.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    rs = up.ewm(alpha=1 / n, min_periods=n).mean() / down.ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n).mean()


def swings(close, look=60):
    sw_hi = close.rolling(look).max().iloc[-1]
    sw_lo = close.rolling(look).min().iloc[-1]
    if pd.isna(sw_hi) or pd.isna(sw_lo) or sw_hi == sw_lo:
        return None
    diff = sw_hi - sw_lo
    return ({"0.382": sw_hi - 0.382 * diff, "0.618": sw_hi - 0.618 * diff}, sw_lo, sw_hi)


def fx_key(contract):
    return f"{contract.symbol}{contract.currency}" if contract.secType == "CASH" else contract.symbol


def safe_mid(ticker):
    bid, ask = ticker.bid, ticker.ask
    if bid is not None and ask is not None and not math.isnan(bid) and not math.isnan(ask):
        return float((bid + ask) / 2.0)
    for val in (ticker.last, ticker.close, ticker.bid):
        if val is not None and (not isinstance(val, float) or not math.isnan(val)):
            return float(val)
    return None


def candle_ok(df):
    row = df.iloc[-1]
    candle_range = row["high"] - row["low"]
    body = abs(row["close"] - row["open"])
    return (candle_range > 0) and (body / candle_range > 0.25)


def signal(df):
    df = df.copy()
    df["ema_f"] = ema(df["close"], 12)
    df["ema_s"] = ema(df["close"], 26)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)

    if len(df) < max(LOOKBACK_MIN, 30) or df[["ema_f", "ema_s", "rsi", "atr"]].iloc[-1].isna().any():
        return "hold", None

    trend = df["ema_f"].iloc[-1] > df["ema_s"].iloc[-1]
    rsi_ok = 45 <= df["rsi"].iloc[-1] <= 68
    sw = swings(df["close"], look=min(60, len(df) - 1))
    if sw is None:
        return "hold", None

    levels, _, swing_hi = sw
    px = df["close"].iloc[-1]
    near382 = abs(px - levels["0.382"]) / px < 0.002
    near618 = abs(px - levels["0.618"]) / px < 0.002

    if (near382 or near618) and trend and rsi_ok and candle_ok(df):
        stop = px - 1.5 * df["atr"].iloc[-1]
        target = swing_hi
        return "buy", {"entry": float(px), "stop": float(stop), "target": float(target)}

    return "hold", None


def main():
    symbols = _load_symbols()
    if not symbols:
        print(f"[{now_ts()}] ERROR: no symbols configured.")
        sys.exit(1)

    if not ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15):
        print(f"[{now_ts()}] ERROR: cannot connect to IB Gateway on {HOST}:{PORT}")
        sys.exit(1)

    ib.reqMarketDataType(MARKET_DATA_TYPE)

    make_contract = lambda s: Forex(s) if (len(s) == 6 and s.isalpha()) else Stock(s, "SMART", "USD")
    contracts = [make_contract(s) for s in symbols]
    ib.qualifyContracts(*contracts)
    ib.reqTickers(*contracts)

    hist = {s: pd.DataFrame(columns=["ts", "open", "high", "low", "close"]) for s in symbols}
    in_pos = {s: None for s in symbols}
    cash = ACCOUNT_EQUITY
    pnl_today = 0.0
    trades_today = 0
    cur_day = datetime.now().date()

    print(
        f"[{now_ts()}] Aion Shadow: LIVE={MARKET_DATA_TYPE == 1} | Symbols={symbols} | "
        f"Risk={int(RISK_PCT * 100)}% | Cap={TRADES_PER_DAY_CAP}/day | Goal=${DAILY_GOAL_USD}"
    )
    print(f"State: {STATE_DIR}")
    print(f"Logs: {TRADES_CSV} , {EQUITY_CSV}")

    last_bar = 0.0
    while True:
        ib.sleep(1.0)
        ticks = ib.reqTickers(*contracts)
        ts = datetime.now()

        for tick in ticks:
            key = fx_key(tick.contract)
            if key not in hist:
                continue
            price = safe_mid(tick)
            if price is None:
                continue

            df = hist[key]
            if len(df) == 0 or (ts.timestamp() - last_bar) >= BAR_SECONDS:
                row = {"ts": ts, "open": price, "high": price, "low": price, "close": price}
                base = df if not df.empty else pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
                hist[key] = pd.concat([base, pd.DataFrame([row])], ignore_index=True).tail(5000)
            else:
                i = -1
                df.iloc[i, df.columns.get_loc("high")] = max(df.iloc[i]["high"], price)
                df.iloc[i, df.columns.get_loc("low")] = min(df.iloc[i]["low"], price)
                df.iloc[i, df.columns.get_loc("close")] = price

            hist[key].reset_index(drop=True, inplace=True)

        if (ts.timestamp() - last_bar) < BAR_SECONDS:
            continue

        last_bar = ts.timestamp()

        if ts.date() != cur_day:
            cur_day = ts.date()
            pnl_today = 0.0
            trades_today = 0

        if pnl_today >= DAILY_GOAL_USD or trades_today >= TRADES_PER_DAY_CAP:
            eq = cash
            for sym, pos in in_pos.items():
                if pos and len(hist[sym]) > 0:
                    eq += (hist[sym]["close"].iloc[-1] - pos["entry"]) * pos["qty"]
            append_row(
                EQUITY_CSV,
                {"ts": now_ts(), "equity": eq, "todays_pnl": pnl_today, "trades_today": trades_today},
                H_EQ,
            )
            continue

        for sym in symbols:
            df = hist[sym]
            if len(df) < LOOKBACK_MIN:
                continue

            px = df["close"].iloc[-1]
            pos = in_pos[sym]

            if pos:
                stop, target, qty = pos["stop"], pos["target"], pos["qty"]
                if px <= stop or px >= target:
                    pnl = (px - pos["entry"]) * qty
                    cash += pnl
                    pnl_today += pnl
                    append_row(
                        TRADES_CSV,
                        {
                            "ts": now_ts(),
                            "symbol": sym,
                            "action": "EXIT",
                            "qty": qty,
                            "price": round(px, 5),
                            "stop": round(stop, 5),
                            "target": round(target, 5),
                            "pnl": round(pnl, 2),
                        },
                        H_TR,
                    )
                    in_pos[sym] = None
                    continue

            if (not pos) and trades_today < TRADES_PER_DAY_CAP and pnl_today < DAILY_GOAL_USD:
                sig, meta = signal(df)
                if sig == "buy" and meta:
                    risk = max(1e-5, meta["entry"] - meta["stop"])
                    dollars = max(10.0, ACCOUNT_EQUITY * RISK_PCT)
                    qty = int(dollars / risk)
                    if qty > 0:
                        in_pos[sym] = {
                            "entry": meta["entry"],
                            "stop": meta["stop"],
                            "target": meta["target"],
                            "qty": qty,
                        }
                        trades_today += 1
                        append_row(
                            TRADES_CSV,
                            {
                                "ts": now_ts(),
                                "symbol": sym,
                                "action": "BUY",
                                "qty": qty,
                                "price": round(meta["entry"], 5),
                                "stop": round(meta["stop"], 5),
                                "target": round(meta["target"], 5),
                                "pnl": 0.0,
                            },
                            H_TR,
                        )

        eq = cash
        for sym, pos in in_pos.items():
            if pos and len(hist[sym]) > 0:
                eq += (hist[sym]["close"].iloc[-1] - pos["entry"]) * pos["qty"]
        append_row(
            EQUITY_CSV,
            {"ts": now_ts(), "equity": eq, "todays_pnl": pnl_today, "trades_today": trades_today},
            H_EQ,
        )


if __name__ == "__main__":
    main()
