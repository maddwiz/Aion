from aion.risk.exposure_gate import check_exposure


def test_exposure_gate_vetoes_when_gross_limit_exceeded():
    res = check_exposure(
        current_positions={"AAPL": 60000.0, "MSFT": 30000.0},
        proposed_symbol="NVDA",
        proposed_value=20000.0,
        net_liquidation=100000.0,
        max_gross_exposure_pct=0.95,
    )
    assert res.allowed is False
    assert "gross_exposure" in res.reason


def test_exposure_gate_vetoes_single_position_limit():
    res = check_exposure(
        current_positions={"AAPL": 15000.0},
        proposed_symbol="AAPL",
        proposed_value=10000.0,
        net_liquidation=100000.0,
        max_gross_exposure_pct=0.95,
        max_single_position_pct=0.20,
    )
    assert res.allowed is False
    assert "single_position" in res.reason


def test_exposure_gate_vetoes_zero_nlv():
    res = check_exposure(
        current_positions={"AAPL": 1000.0},
        proposed_symbol="MSFT",
        proposed_value=500.0,
        net_liquidation=0.0,
    )
    assert res.allowed is False
    assert "net_liquidation" in res.reason


def test_exposure_gate_allows_within_limits():
    res = check_exposure(
        current_positions={"AAPL": 10000.0, "MSFT": 5000.0},
        proposed_symbol="NVDA",
        proposed_value=3000.0,
        net_liquidation=100000.0,
        max_gross_exposure_pct=0.95,
        max_single_position_pct=0.20,
    )
    assert res.allowed is True
    assert res.reason == "passed"
