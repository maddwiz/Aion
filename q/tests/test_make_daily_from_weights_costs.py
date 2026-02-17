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
    net, gross, cost = mdw.build_costed_daily_returns(w, a, cost_bps=10.0, fixed_daily_fee=0.0)
    assert len(net) == 3
    assert len(gross) == 3
    assert len(cost) == 3
    # Turnover: [0.0, 0.5, 1.0] -> costs [0.0, 0.0005, 0.0010]
    assert np.allclose(cost, np.array([0.0, 0.0005, 0.0010], dtype=float))
    assert np.allclose(net, gross - cost)


def test_build_costed_daily_returns_adds_fixed_daily_fee():
    w = np.array([[1.0], [1.0]], dtype=float)
    a = np.array([[0.01], [0.01]], dtype=float)
    net, gross, cost = mdw.build_costed_daily_returns(w, a, cost_bps=0.0, fixed_daily_fee=0.0002)
    assert np.allclose(gross, np.array([0.01, 0.01]))
    assert np.allclose(cost, np.array([0.0002, 0.0002]))
    assert np.allclose(net, np.array([0.0098, 0.0098]))
