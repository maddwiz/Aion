import numpy as np

import tools.build_final_portfolio as bfp


def test_compute_hit_gate_scalars_monotone_and_bounded():
    hp = np.array([0.40, 0.50, 0.60], dtype=float)
    g = bfp.compute_hit_gate_scalars(
        hp,
        threshold=0.50,
        floor=0.30,
        ceiling=1.10,
        slope=12.0,
    )
    assert len(g) == len(hp)
    assert float(np.min(g)) >= 0.30 - 1e-9
    assert float(np.max(g)) <= 1.10 + 1e-9
    assert g[0] < g[1] < g[2]


def test_apply_signal_deadzone_prunes_more_on_low_hit_proxy():
    w = np.array(
        [
            [0.0020, -0.0011, 0.0007],  # weak expected hit -> larger deadzone
            [0.0020, -0.0011, 0.0007],  # stronger expected hit -> smaller deadzone
        ],
        dtype=float,
    )
    hp = np.array([0.42, 0.62], dtype=float)
    out, info = bfp.apply_signal_deadzone(
        w,
        base_deadzone=0.0010,
        hit_proxy=hp,
        hit_threshold=0.50,
        hit_sensitivity=1.0,
    )
    assert info["enabled"] is True
    assert out.shape == w.shape
    assert int(np.count_nonzero(out[0])) < int(np.count_nonzero(out[1]))
