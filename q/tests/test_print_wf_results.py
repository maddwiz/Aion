from pathlib import Path

import numpy as np

from tools import print_wf_results as pwr


def _write_series(path: Path, values):
    arr = np.asarray(values, float).ravel()
    np.savetxt(path, arr, delimiter=",")


def test_pick_returns_series_auto_prefers_newer_daily(tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    wf = runs / "wf_oos_returns.csv"
    daily = runs / "daily_returns.csv"
    _write_series(wf, [0.1, -0.2, 0.3])
    _write_series(daily, [0.2, 0.1, -0.1])
    wf.touch()
    daily.touch()

    arr, source = pwr.pick_returns_series(runs, source_pref="auto")
    assert source == "daily_returns"
    assert arr is not None
    assert arr.shape[0] == 3


def test_pick_returns_series_auto_uses_wf_when_daily_missing(tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    wf = runs / "wf_oos_returns.csv"
    _write_series(wf, [0.1, -0.2, 0.3])

    arr, source = pwr.pick_returns_series(runs, source_pref="auto")
    assert source == "wf_oos_returns"
    assert arr is not None
    assert arr.shape[0] == 3


def test_pick_returns_series_explicit_pref_uses_requested_file(tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    wf = runs / "wf_oos_returns.csv"
    daily = runs / "daily_returns.csv"
    _write_series(wf, [0.1, -0.2, 0.3])
    _write_series(daily, [0.2, 0.1, -0.1])

    arr, source = pwr.pick_returns_series(runs, source_pref="wf_oos_returns")
    assert source == "wf_oos_returns"
    assert arr is not None
    assert np.allclose(arr, np.array([0.1, -0.2, 0.3]))
