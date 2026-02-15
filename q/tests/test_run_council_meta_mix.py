import numpy as np

import tools.run_council_meta_mix as rcmm


def test_robust_time_split_score_prefers_aligned_signal():
    rng = np.random.default_rng(101)
    t = np.linspace(0.0, 20.0, 900)
    y = 0.0025 * np.sin(t - 0.08) + 0.0008 * np.cos(1.2 * t) + 0.0005 * rng.standard_normal(t.size)
    y_fwd = y[1:]

    pos_good = np.tanh(0.35 * np.sin(t - 0.10))
    pos_bad = -pos_good

    score_good, folds_good = rcmm.robust_time_split_score(pos_good, y_fwd, n_folds=5, min_fold=80)
    score_bad, folds_bad = rcmm.robust_time_split_score(pos_bad, y_fwd, n_folds=5, min_fold=80)

    assert score_good is not None
    assert score_bad is not None
    assert len(folds_good) >= 3
    assert len(folds_bad) >= 3
    assert float(score_good) > float(score_bad)


def test_robust_time_split_score_returns_none_when_too_short():
    pos = np.zeros(50)
    y_fwd = np.zeros(49)
    score, folds = rcmm.robust_time_split_score(pos, y_fwd, n_folds=4, min_fold=40)
    assert score is None
    assert folds == []
