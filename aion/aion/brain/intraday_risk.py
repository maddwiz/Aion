"""
Intraday risk management for day_skimmer mode.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntradayRiskParams:
    stop_atr_multiple: float = 1.5
    risk_per_trade_pct: float = 0.005
    max_position_pct: float = 0.03
    partial_profit_r: float = 1.0
    partial_profit_fraction: float = 0.50
    trailing_stop_atr: float = 1.2
    max_trades_per_session: int = 8
    max_daily_loss_pct: float = 0.015
    max_open_positions: int = 3
    max_correlated_positions: int = 2
    no_new_entries_after_min: int = 45
    force_close_all_at_min: int = 10


@dataclass
class PositionSizeResult:
    shares: int
    stop_price: float
    risk_distance: float
    risk_amount: float
    r_target_1: float
    r_target_2: float


def compute_position_size(
    side: str,
    entry_price: float,
    atr_5m: float,
    equity: float,
    params: IntradayRiskParams,
) -> PositionSizeResult:
    s = str(side or "").strip().upper()
    if s not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")

    px = float(max(1e-9, entry_price))
    atr = float(max(1e-9, atr_5m))
    risk_distance = atr * float(max(0.1, params.stop_atr_multiple))

    if s == "LONG":
        stop_price = px - risk_distance
        r1 = px + risk_distance
        r2 = px + 2.0 * risk_distance
    else:
        stop_price = px + risk_distance
        r1 = px - risk_distance
        r2 = px - 2.0 * risk_distance

    risk_budget = float(max(0.0, equity) * max(0.0, params.risk_per_trade_pct))
    shares = int(risk_budget / max(risk_distance, 1e-9))
    max_shares = int((float(max(0.0, equity)) * max(0.0, params.max_position_pct)) / px)
    shares = max(1, min(shares, max_shares))

    return PositionSizeResult(
        shares=int(shares),
        stop_price=float(stop_price),
        risk_distance=float(risk_distance),
        risk_amount=float(shares * risk_distance),
        r_target_1=float(r1),
        r_target_2=float(r2),
    )


@dataclass
class SessionRiskState:
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    open_positions: int = 0
    max_drawdown_today: float = 0.0
    peak_equity_today: float = 0.0
    session_locked: bool = False
    lock_reason: str = ""


class IntradayRiskManager:
    def __init__(self, equity: float, params: IntradayRiskParams):
        self.equity = float(max(0.0, equity))
        self.params = params
        self.state = SessionRiskState()
        self.state.peak_equity_today = self.equity

    def can_enter(self, minutes_to_close: int, current_open: int) -> tuple[bool, str]:
        if self.state.session_locked:
            return False, f"Session locked: {self.state.lock_reason}"
        if self.state.trades_taken >= int(max(0, self.params.max_trades_per_session)):
            return False, f"Max trades reached ({self.params.max_trades_per_session})"
        if int(current_open) >= int(max(0, self.params.max_open_positions)):
            return False, f"Max open positions ({self.params.max_open_positions})"
        if int(minutes_to_close) <= int(max(0, self.params.no_new_entries_after_min)):
            return False, f"Too close to session end ({minutes_to_close} min left)"
        return True, "Entry allowed"

    def record_trade_result(self, pnl: float):
        p = float(pnl)
        self.state.trades_taken += 1
        self.state.daily_pnl += p
        self.state.daily_pnl_pct = self.state.daily_pnl / (self.equity + 1e-9)
        if p > 0:
            self.state.trades_won += 1
        else:
            self.state.trades_lost += 1
        if self.state.daily_pnl_pct <= -float(max(0.0, self.params.max_daily_loss_pct)):
            self.state.session_locked = True
            self.state.lock_reason = f"Daily loss limit hit ({self.state.daily_pnl_pct:.2%})"

    def should_force_close_all(self, minutes_to_close: int) -> bool:
        return int(minutes_to_close) <= int(max(0, self.params.force_close_all_at_min))

    def session_hit_rate(self) -> float:
        total = int(self.state.trades_won + self.state.trades_lost)
        if total <= 0:
            return 0.5
        return float(self.state.trades_won / max(1, total))

