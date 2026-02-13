import numpy as np

from qmods.quality_governor import (
    blend_quality,
    build_governor_series,
    drawdown_quality,
    hit_quality,
    sharpe_quality,
)


def test_quality_mappers_bounds():
    assert 0.0 <= float(sharpe_quality(-3.0)) <= 1.0
    assert 0.0 <= float(sharpe_quality(3.0)) <= 1.0
    assert float(sharpe_quality(1.0)) > float(sharpe_quality(0.0)) > float(sharpe_quality(-1.0))

    assert 0.0 <= float(hit_quality(0.40)) <= 1.0
    assert 0.0 <= float(hit_quality(0.60)) <= 1.0
    assert float(hit_quality(0.60)) > float(hit_quality(0.50)) > float(hit_quality(0.40))

    assert 0.0 <= float(drawdown_quality(-0.05)) <= 1.0
    assert 0.0 <= float(drawdown_quality(-0.50)) <= 1.0
    assert float(drawdown_quality(-0.05)) > float(drawdown_quality(-0.30))


def test_blend_quality_ignores_missing():
    q, detail = blend_quality(
        {
            "a": (0.8, 0.7),
            "b": (None, 0.2),
            "c": (0.2, 0.1),
        }
    )
    assert 0.0 <= q <= 1.0
    assert "a" in detail["used_components"]
    assert "b" not in detail["used_components"]


def test_governor_series_shape_bounds_and_monotonic_base():
    T = 120
    dg = np.linspace(1.0, 0.4, T)
    gg = np.linspace(1.0, 0.8, T)
    g_low = build_governor_series(T, base_quality=0.25, disagreement_gate=dg, global_governor=gg)
    g_high = build_governor_series(T, base_quality=0.80, disagreement_gate=dg, global_governor=gg)

    assert g_low.shape == (T,)
    assert g_high.shape == (T,)
    assert np.isfinite(g_low).all() and np.isfinite(g_high).all()
    assert float(np.min(g_low)) >= 0.55 - 1e-9
    assert float(np.max(g_high)) <= 1.15 + 1e-9
    assert float(np.mean(g_high)) > float(np.mean(g_low))
