import numpy as np

from qmods.meta_stack_v1 import MetaStackV1


def test_meta_stack_fit_predict_confidence():
    rng = np.random.default_rng(11)
    T = 420
    K = 6
    X = rng.normal(0.0, 1.0, size=(T, K))
    beta = np.array([0.35, -0.2, 0.15, 0.0, 0.1, -0.05], dtype=float)
    y = (X @ beta + rng.normal(0.0, 0.3, size=T)).astype(float)

    m = MetaStackV1()
    m.fit(X, y)
    p = m.predict(X)
    c = m.predict_confidence(X)

    assert p.shape == (T,)
    assert c.shape == (T,)
    assert np.isfinite(p).all()
    assert np.isfinite(c).all()
    assert float(np.min(c)) >= 0.0 - 1e-9
    assert float(np.max(c)) <= 1.0 + 1e-9
    # basic sanity: predictive direction should be non-random on synthetic linear data
    # Model is trained with 1-step lag (X[t-1] -> y[t]), so compare p[:-1] with y[1:].
    corr = float(np.corrcoef(p[:-1], y[1:])[0, 1])
    assert corr > 0.05
