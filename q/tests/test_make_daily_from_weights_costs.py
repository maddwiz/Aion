import numpy as np

import tools.make_daily_from_weights as mdw


def test_build_costed_daily_returns_applies_turnover_costs():
    # 3 timesteps, 2 assets.
    w = np.array(
        [
            [0.50, 0.50],
            [0.75, 0.25],
            [0.25, 0.75],
        ],
        dtype=float,
    )
    a = np.array(
        [
            [0.01, 0.00],
            [0.01, 0.00],
            [0.00, 0.01],
        ],
        dtype=float,
    )
    net, gross, cost, turnover, eff_bps, cash_carry, cash_frac = mdw.build_costed_daily_returns(
        w,
        a,
        base_bps=10.0,
        vol_scaled_bps=0.0,
        half_turnover=True,
        fixed_daily_fee=0.0,
    )
    assert len(net) == 3
    assert len(gross) == 3
    assert len(cost) == 3
    # Half-turnover: [0.0, 0.25, 0.50] -> costs [0.0, 0.00025, 0.00050]
    assert np.allclose(turnover, np.array([0.0, 0.25, 0.50], dtype=float))
    assert np.allclose(cost, np.array([0.0, 0.00025, 0.00050], dtype=float))
    assert np.allclose(eff_bps, np.array([10.0, 10.0, 10.0], dtype=float))
    assert np.allclose(cash_carry, np.zeros(3, dtype=float))
    assert np.allclose(cash_frac, np.zeros(3, dtype=float))
    assert np.allclose(net, gross - cost)


def test_build_costed_daily_returns_adds_fixed_daily_fee():
    w = np.array([[1.0], [1.0]], dtype=float)
    a = np.array([[0.01], [0.01]], dtype=float)
    net, gross, cost, _turnover, _eff_bps, _cash_carry, _cash_frac = mdw.build_costed_daily_returns(
        w,
        a,
        base_bps=0.0,
        vol_scaled_bps=0.0,
        fixed_daily_fee=0.0002,
    )
    assert np.allclose(gross, np.array([0.01, 0.01]))
    assert np.allclose(cost, np.array([0.0002, 0.0002]))
    assert np.allclose(net, np.array([0.0098, 0.0098]))


def test_build_costed_daily_returns_vol_scaled_bps_reacts_to_vol():
    w = np.array(
        [
            [1.0],
            [1.0],
            [1.0],
            [1.0],
        ],
        dtype=float,
    )
    # Higher realized vol in later rows.
    a = np.array([[0.0], [0.001], [0.010], [-0.010]], dtype=float)
    _net, _gross, _cost, _turn, eff_bps, _cash_carry, _cash_frac = mdw.build_costed_daily_returns(
        w,
        a,
        base_bps=3.0,
        vol_scaled_bps=10.0,
        vol_lookback=2,
        vol_ref_daily=0.001,
        half_turnover=False,
    )
    assert eff_bps[-1] > eff_bps[1]


def test_build_costed_daily_returns_adds_cash_carry():
    w = np.array([[0.5], [0.5]], dtype=float)
    a = np.array([[0.0], [0.0]], dtype=float)
    net, gross, cost, _turnover, _eff_bps, carry, cash_frac = mdw.build_costed_daily_returns(
        w,
        a,
        base_bps=0.0,
        vol_scaled_bps=0.0,
        fixed_daily_fee=0.0,
        cash_yield_annual=0.05,
        cash_exposure_target=1.0,
    )
    expected_carry = np.full(2, 0.5 * 0.05 / 252.0, dtype=float)
    assert np.allclose(carry, expected_carry)
    assert np.allclose(cash_frac, np.full(2, 0.5, dtype=float))
    assert np.allclose(cost, np.zeros(2, dtype=float))
    assert np.allclose(gross, expected_carry)
    assert np.allclose(net, expected_carry)
