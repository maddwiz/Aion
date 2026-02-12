from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FillResult:
    filled_qty: int
    avg_fill: float
    fill_ratio: float
    est_slippage_bps: float


class ExecutionSimulator:
    def __init__(self, cfg):
        self.cfg = cfg

    def _spread_bps(self, atr_pct: float, confidence: float) -> float:
        conf_boost = 1.0 - 0.25 * max(0.0, min(1.0, confidence))
        return (self.cfg.SPREAD_BPS_BASE + self.cfg.SPREAD_BPS_VOL_MULT * max(0.0, atr_pct)) * conf_boost

    def execute(self, side: str, qty: int, ref_price: float, atr_pct: float, confidence: float, allow_partial: bool = True) -> FillResult:
        if qty <= 0 or ref_price <= 0:
            return FillResult(0, ref_price, 0.0, 0.0)

        spread_bps = self._spread_bps(atr_pct, confidence)
        queue_impact = self.cfg.EXEC_QUEUE_IMPACT_BPS * min(1.0, qty / 500.0)
        latency_impact = 0.8 * (self.cfg.EXEC_LATENCY_MS / 1000.0)
        slippage_bps = self.cfg.SLIPPAGE_BPS + queue_impact + latency_impact + (atr_pct * 10000.0 * 0.08)

        half_spread = ref_price * (spread_bps / 10000.0) * 0.5
        slip = ref_price * (slippage_bps / 10000.0)

        if side.upper() == "BUY":
            px = ref_price + half_spread + slip
        else:
            px = ref_price - half_spread - slip

        if allow_partial:
            raw_ratio = 0.45 + 0.55 * confidence - 0.75 * min(0.25, atr_pct)
            fill_ratio = max(self.cfg.EXEC_PARTIAL_FILL_MIN, min(self.cfg.EXEC_PARTIAL_FILL_MAX, raw_ratio))
        else:
            fill_ratio = 1.0

        filled_qty = max(0, int(math.floor(qty * fill_ratio)))
        if filled_qty == 0 and qty > 0 and fill_ratio > 0:
            filled_qty = 1

        return FillResult(
            filled_qty=filled_qty,
            avg_fill=float(px),
            fill_ratio=float(filled_qty / max(qty, 1)),
            est_slippage_bps=float(slippage_bps),
        )
