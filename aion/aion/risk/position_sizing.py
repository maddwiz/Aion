def risk_qty(
    equity: float,
    risk_per_trade: float,
    atr: float,
    price: float,
    confidence: float = 1.0,
    stop_atr_mult: float = 1.0,
    max_notional_pct: float = 0.25,
) -> int:
    if equity <= 0 or atr <= 0 or price <= 0:
        return 0

    confidence = max(0.1, min(1.0, confidence))
    adjusted_risk = equity * risk_per_trade * (0.6 + 0.8 * confidence)

    stop_distance = max(atr * stop_atr_mult, price * 0.0035)
    qty_by_risk = int(adjusted_risk / stop_distance)

    max_notional = equity * max_notional_pct
    qty_by_notional = int(max_notional / price)

    qty = min(qty_by_risk, qty_by_notional)
    return max(qty, 0)


def gross_leverage_ok(cash: float, open_positions: dict, next_notional: float, max_gross_leverage: float, equity: float) -> bool:
    if equity <= 0:
        return False
    gross = 0.0
    for pos in open_positions.values():
        gross += abs(pos.get("qty", 0) * pos.get("mark_price", pos.get("entry", 0.0)))
    gross += abs(next_notional)
    return (gross / equity) <= max_gross_leverage
