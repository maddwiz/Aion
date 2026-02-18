import numpy as np

import tools.run_hive_conviction_gate as rhcg


def test_lone_dissenter_gets_dampened():
    w = np.array([[1.0, 1.0, -1.0]], dtype=float)
    names = ["AAPL", "MSFT", "NVDA"]
    hives = {"TECH": ["AAPL", "MSFT", "NVDA"]}

    s, _ = rhcg.compute_hive_conviction_scalar(
        w,
        names,
        hives,
        threshold=0.40,
        floor=0.30,
        ceil=1.12,
        high_conviction=0.70,
    )

    assert abs(float(s[0, 2]) - 0.30) < 1e-12


def test_unanimous_hive_gets_boosted():
    w = np.array([[1.0, 1.0, 1.0]], dtype=float)
    names = ["AAPL", "MSFT", "NVDA"]
    hives = {"TECH": ["AAPL", "MSFT", "NVDA"]}

    s, _ = rhcg.compute_hive_conviction_scalar(
        w,
        names,
        hives,
        threshold=0.40,
        floor=0.30,
        ceil=1.12,
        high_conviction=0.70,
    )

    assert np.allclose(s, np.full_like(s, 1.12))


def test_assets_not_in_any_hive_pass_through():
    w = np.array([[0.5, -0.4, 0.2]], dtype=float)
    names = ["AAPL", "MSFT", "TLT"]
    hives = {"TECH": ["AAPL", "MSFT"]}

    s, _ = rhcg.compute_hive_conviction_scalar(
        w,
        names,
        hives,
        threshold=0.40,
        floor=0.30,
        ceil=1.12,
        high_conviction=0.70,
    )

    assert abs(float(s[0, 2]) - 1.0) < 1e-12


def test_output_dimensions_match_weight_matrix():
    w = np.array(
        [
            [1.0, -1.0, 0.0],
            [0.8, -0.7, 0.1],
            [-0.5, 0.6, 0.2],
        ],
        dtype=float,
    )
    names = ["AAPL", "MSFT", "NVDA"]
    hives = {"TECH": ["AAPL", "MSFT", "NVDA"]}

    s, info = rhcg.compute_hive_conviction_scalar(
        w,
        names,
        hives,
        threshold=0.40,
        floor=0.30,
        ceil=1.12,
        high_conviction=0.70,
    )

    assert s.shape == w.shape
    assert int(info["rows"]) == w.shape[0]
    assert int(info["cols"]) == w.shape[1]
