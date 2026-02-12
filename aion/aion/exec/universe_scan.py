import datetime as dt
import json

from .. import config as cfg
from ..brain.signals import build_trade_signal, compute_features
from ..data.ib_client import disconnect, hist_bars, ib
from ..utils.logging_utils import log_scan

WATCHLIST_TXT = cfg.STATE_DIR / "watchlist.txt"
WATCHLIST_JSON = cfg.STATE_DIR / "watchlist.json"


def load_universe() -> list[str]:
    symbols = []
    for name in ["dow30.txt", "sp500.txt", "nasdaq100.txt"]:
        path = cfg.UNIVERSE_DIR / name
        if path.exists():
            symbols += [
                s.strip().upper()
                for s in path.read_text().splitlines()
                if s.strip() and not s.startswith("#")
            ]

    seen = set()
    uniq = []
    for sym in symbols:
        if sym not in seen:
            uniq.append(sym)
            seen.add(sym)
    return uniq


def score_symbol(sym: str):
    try:
        df = hist_bars(sym, duration=cfg.HIST_DURATION, barSize=cfg.HIST_BAR_SIZE)
        min_scan_bars = max(45, cfg.MIN_BARS // 2)
        if df.empty or len(df) < min_scan_bars:
            return None

        feats = compute_features(df, cfg)
        row = feats.iloc[-1]
        px = float(row["close"])
        lookback = min(cfg.SWING_LOOKBACK, len(df))
        hi = float(df["high"].iloc[-lookback:].max())
        lo = float(df["low"].iloc[-lookback:].min())

        signal = build_trade_signal(row, px, hi, lo, cfg)

        trend_score = min(1.0, max(0.0, float(row["adx"]) / 40.0))
        atr_penalty = min(1.0, max(0.0, float(row["atr_pct"]) / max(cfg.REGIME_ATR_PCT_HIGH, 1e-6)))
        raw_conf = max(float(signal["long_conf"]), float(signal["short_conf"]))

        conf_l = float(signal.get("confluence_long", 0.0))
        conf_s = float(signal.get("confluence_short", 0.0))
        side_conf = conf_l if signal["side"] == "LONG" else conf_s if signal["side"] == "SHORT" else max(conf_l, conf_s)
        conflict = min(conf_l, conf_s)

        # Rank by confidence, trend quality, and confluence while penalizing noisy volatility/conflict.
        rank = 0.66 * raw_conf + 0.20 * trend_score + 0.20 * side_conf - 0.16 * atr_penalty - 0.10 * conflict
        if signal["regime"] == "high_vol_chop":
            rank -= 0.12

        return {
            "symbol": sym,
            "rank": float(rank),
            "signal": signal["side"] or "HOLD",
            "regime": signal["regime"],
            "long_conf": float(signal["long_conf"]),
            "short_conf": float(signal["short_conf"]),
            "confidence": float(signal["confidence"]),
            "confluence_long": conf_l,
            "confluence_short": conf_s,
            "adx": float(row["adx"]),
            "atr_pct": float(row["atr_pct"]),
        }

    except Exception as exc:
        log_scan(f"{sym}: scan error: {exc}")
        return None


def main() -> int:
    symbols = load_universe()
    log_scan(f"Universe loaded: {len(symbols)} symbols")

    try:
        ib()
    except Exception as exc:
        log_scan(f"IB connection failed: {exc}")
        print(f"[AION] ERROR: Unable to connect to IBKR ({cfg.IB_HOST}:{cfg.IB_PORT}): {exc}")
        return 1

    rows = []
    for sym in symbols:
        scored = score_symbol(sym)
        if scored is not None:
            rows.append(scored)

    rows.sort(key=lambda x: x["rank"], reverse=True)
    shortlist_rows = rows[: cfg.SHORTLIST_CAP]
    shortlist = [r["symbol"] for r in shortlist_rows]

    WATCHLIST_TXT.write_text("\n".join(shortlist) + ("\n" if shortlist else ""))
    WATCHLIST_JSON.write_text(
        json.dumps(
            {
                "date": dt.datetime.now().isoformat(),
                "universe": len(symbols),
                "ranked": len(rows),
                "cap": cfg.SHORTLIST_CAP,
                "shortlist": shortlist_rows,
            },
            indent=2,
        )
    )

    long_count = sum(1 for r in shortlist_rows if r["signal"] == "LONG")
    short_count = sum(1 for r in shortlist_rows if r["signal"] == "SHORT")
    log_scan(
        f"Shortlist built: {len(shortlist)}/{len(symbols)} symbols promoted | LONG={long_count} SHORT={short_count}"
    )

    disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
