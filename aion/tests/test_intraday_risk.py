from aion.brain.intraday_risk import (
    IntradayRiskManager,
    IntradayRiskParams,
    compute_position_size,
)


def test_position_sizing_known_inputs():
    p = IntradayRiskParams(
        stop_atr_multiple=1.5,
        risk_per_trade_pct=0.005,
        max_position_pct=0.03,
    )
    out = compute_position_size("LONG", entry_price=100.0, atr_5m=1.0, equity=10_000.0, params=p)
    # risk budget = 50, risk_distance = 1.5 => 33 shares before max position cap
    assert out.shares == 3  # max position cap: 3% of equity / price = 3 shares
    assert abs(out.stop_price - 98.5) < 1e-9
    assert abs(out.r_target_1 - 101.5) < 1e-9


def test_session_lock_triggers_at_daily_loss_limit():
    p = IntradayRiskParams(max_daily_loss_pct=0.01)  # 1%
    rm = IntradayRiskManager(equity=10_000.0, params=p)
    rm.record_trade_result(-120.0)  # -1.2%
    assert rm.state.session_locked is True
    assert "Daily loss limit hit" in rm.state.lock_reason


def test_no_new_entries_in_final_45_minutes():
    p = IntradayRiskParams(no_new_entries_after_min=45)
    rm = IntradayRiskManager(equity=10_000.0, params=p)
    ok, _ = rm.can_enter(minutes_to_close=44, current_open=0)
    assert ok is False


def test_force_close_in_final_10_minutes():
    p = IntradayRiskParams(force_close_all_at_min=10)
    rm = IntradayRiskManager(equity=10_000.0, params=p)
    assert rm.should_force_close_all(minutes_to_close=11) is False
    assert rm.should_force_close_all(minutes_to_close=10) is True


def test_max_trades_per_session_enforced():
    p = IntradayRiskParams(max_trades_per_session=2)
    rm = IntradayRiskManager(equity=10_000.0, params=p)
    rm.record_trade_result(50.0)
    rm.record_trade_result(-30.0)
    ok, reason = rm.can_enter(minutes_to_close=120, current_open=0)
    assert ok is False
    assert "Max trades reached" in reason

