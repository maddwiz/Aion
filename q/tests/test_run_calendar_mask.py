from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_calendar_mask as rcm


def test_turn_of_month_detection_across_year_boundary():
    idx = pd.bdate_range("2025-12-15", "2026-01-12")
    m = rcm._turn_of_month_mask(idx)

    periods = idx.to_period("M")
    for p in periods.unique():
        loc = np.where(periods == p)[0]
        n = len(loc)
        expect = set(loc[: min(3, n)].tolist() + loc[max(0, n - 2) :].tolist())
        got = {i for i in loc.tolist() if bool(m[i])}
        assert expect.issubset(got)


def test_fomc_parser_and_missing_file_fallback(tmp_path: Path):
    p = tmp_path / "fomc_dates.csv"
    p.write_text(
        "date,type\n2026-03-18,announcement\n2026-04-08,minutes\n2026-05-06,announcement\n",
        encoding="utf-8",
    )

    dates = rcm._load_fomc_announcement_dates(p)
    assert pd.Timestamp("2026-03-18") in dates
    assert pd.Timestamp("2026-05-06") in dates
    assert pd.Timestamp("2026-04-08") not in dates

    missing = rcm._load_fomc_announcement_dates(tmp_path / "missing.csv")
    assert missing == set()


def test_calendar_mask_scalars_respect_floor_and_ceil():
    t = 80
    rng = np.random.default_rng(7)
    hit = (rng.random(t) > 0.48).astype(float)
    features = {k: np.zeros(t, dtype=bool) for k in rcm.STATIC_SCALARS.keys()}
    features["turn_of_month"][::7] = True
    features["fomc_announcement_day"][::13] = True

    s, _ = rcm.compute_calendar_mask_scalar(
        hit,
        features,
        beta=0.6,
        floor=0.75,
        ceil=1.15,
        min_feature_days=10,
    )

    assert s.shape[0] == t
    assert float(np.min(s)) >= 0.75 - 1e-12
    assert float(np.max(s)) <= 1.15 + 1e-12


def test_walkforward_calibration_differentiates_feature_quality():
    t = 140
    hit = np.full(t, 0.5, dtype=float)
    features = {k: np.zeros(t, dtype=bool) for k in rcm.STATIC_SCALARS.keys()}

    # Feature A (good): historically high hit rate.
    good_idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    features["turn_of_month"][good_idx] = True
    hit[good_idx] = 1.0

    # Feature B (bad): historically low hit rate.
    bad_idx = np.array([11, 21, 31, 41, 51, 61, 71, 81, 91, 101])
    features["pre_3day_weekend"][bad_idx] = True
    hit[bad_idx] = 0.0

    s, _ = rcm.compute_calendar_mask_scalar(
        hit,
        features,
        beta=0.9,
        floor=0.75,
        ceil=1.15,
        min_feature_days=5,
    )

    # On later samples after enough history, good-feature days should scale higher.
    assert float(s[100]) > float(s[101])
