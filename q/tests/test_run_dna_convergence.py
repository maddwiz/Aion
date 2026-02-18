import numpy as np

import tools.run_dna_convergence as rdc


def test_identical_assets_produce_near_zero_signal():
    t = 220
    base = 0.003 * np.sin(np.linspace(0.0, 20.0, t))
    ret = np.column_stack([base, base.copy(), base.copy()])

    sig, info = rdc.compute_dna_convergence_signal(
        ret,
        window=48,
        topk=12,
        delta_lookback=21,
        smooth_alpha=0.15,
        divergence_threshold=0.05,
    )

    assert sig.shape == ret.shape
    assert bool(info["ok"]) is True
    assert float(np.max(np.abs(sig))) < 1e-8


def test_diverging_asset_produces_nonzero_score():
    t = 260
    x = np.linspace(0.0, 22.0, t)
    a = 0.004 * np.sin(x)
    b = 0.004 * np.sin(x)
    b[t // 2 :] = -0.004 * np.sin(x[t // 2 :])
    c = 0.003 * np.cos(0.7 * x)
    ret = np.column_stack([a, b, c])

    sig, info = rdc.compute_dna_convergence_signal(
        ret,
        window=64,
        topk=16,
        delta_lookback=21,
        smooth_alpha=0.20,
        divergence_threshold=0.04,
    )

    assert sig.shape == ret.shape
    assert bool(info["ok"]) is True
    # Asset 2 diverges from peers in the back half, so average absolute
    # signal should be materially non-zero.
    assert float(np.mean(np.abs(sig[:, 1]))) > 0.01


def test_output_dimensions_match_asset_universe():
    rng = np.random.default_rng(13)
    ret = rng.normal(0.0, 0.01, size=(180, 7))

    sig, info = rdc.compute_dna_convergence_signal(
        ret,
        window=40,
        topk=10,
        delta_lookback=14,
        smooth_alpha=0.10,
        divergence_threshold=0.05,
    )

    assert sig.shape == ret.shape
    assert int(info["rows"]) == 180
    assert int(info["cols"]) == 7
    assert -1.0 <= float(np.min(sig)) <= 1.0
    assert -1.0 <= float(np.max(sig)) <= 1.0
