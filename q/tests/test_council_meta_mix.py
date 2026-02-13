import numpy as np

from qmods.council_meta_mix import adaptive_blend_series, rolling_component_quality


def test_rolling_component_quality_rewards_aligned_signal():
    T = 420
    t = np.linspace(0.0, 16.0, T)
    rng = np.random.default_rng(17)
    ret = 0.0028 * np.sin(t) + 0.0010 * np.cos(1.4 * t) + 0.0005 * rng.standard_normal(T)
    y_fwd = ret[1:]

    aligned = np.sin(t - 0.05)
    anti = -aligned
    q_aligned = rolling_component_quality(aligned, y_fwd, gross=0.24)
    q_anti = rolling_component_quality(anti, y_fwd, gross=0.24)

    assert q_aligned.shape == (T,)
    assert q_anti.shape == (T,)
    assert float(np.mean(q_aligned)) > float(np.mean(q_anti))


def test_adaptive_blend_shifts_alpha_to_stronger_component():
    T = 420
    t = np.linspace(0.0, 18.0, T)
    rng = np.random.default_rng(23)
    ret = 0.0032 * np.sin(t) + 0.0008 * np.cos(1.1 * t) + 0.0005 * rng.standard_normal(T)
    y_fwd = ret[1:]

    meta = np.sin(t - 0.08) + 0.15 * rng.standard_normal(T)
    syn = -0.6 * np.sin(t - 0.08) + 0.25 * rng.standard_normal(T)
    mc = np.clip(0.65 + 0.05 * np.sin(t), 0.0, 1.0)
    sc = np.clip(0.45 + 0.05 * np.cos(t), 0.0, 1.0)

    ctx = adaptive_blend_series(
        meta_signal=meta,
        syn_signal=syn,
        meta_conf=mc,
        syn_conf=sc,
        forward_returns=y_fwd,
        base_alpha=0.50,
        base_gross=0.24,
    )

    alpha = np.asarray(ctx["alpha"], float)
    gross = np.asarray(ctx["gross"], float)
    qmix = np.asarray(ctx["quality_mix"], float)
    assert alpha.shape == (T,)
    assert gross.shape == (T,)
    assert qmix.shape == (T,)
    assert float(np.mean(alpha)) > 0.52
    assert float(np.min(gross)) >= 0.12 - 1e-9
    assert float(np.max(gross)) <= 0.45 + 1e-9
