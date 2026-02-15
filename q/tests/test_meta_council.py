import numpy as np
import pandas as pd

from qmods.meta_council import meta_council


def test_meta_council_outputs_finite_series():
    rng = np.random.default_rng(21)
    r = rng.normal(0.0002, 0.01, 420)
    close = pd.Series(100.0 * np.exp(np.cumsum(r)))

    raw = meta_council(close)
    raw_v2 = meta_council(close, include_v2=True, v2_weight=0.2)

    assert raw.shape == close.shape
    assert raw_v2.shape == close.shape
    assert np.isfinite(raw).all()
    assert np.isfinite(raw_v2).all()
