import json
from pathlib import Path

import numpy as np

import tools.build_final_portfolio as bfp


def _turnover_mean(w: np.ndarray) -> float:
    if w.shape[0] < 2:
        return 0.0
    return float(np.mean(np.sum(np.abs(np.diff(w, axis=0)), axis=1)))


def test_auto_turnover_govern_builds_budget_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("Q_ENABLE_AUTO_TURNOVER_GOV", "1")
    monkeypatch.setenv("Q_AUTO_TURNOVER_MAX_STEP", "0.40")
    monkeypatch.setenv("Q_AUTO_TURNOVER_BUDGET_WINDOW", "3")
    monkeypatch.setenv("Q_AUTO_TURNOVER_BUDGET_LIMIT", "0.85")
    monkeypatch.setattr(bfp, "RUNS", tmp_path)

    w = np.array(
        [
            [0.90, -0.90, 0.20],
            [-0.85, 0.80, -0.10],
            [0.95, -0.70, -0.40],
            [-0.70, 0.75, 0.35],
        ],
        float,
    )

    out, tag, scale = bfp._auto_turnover_govern(w)
    assert out is not None
    assert tag == "turnover_budget_auto"
    assert scale is not None and len(scale) == w.shape[0]
    assert out.shape == w.shape
    assert _turnover_mean(out) <= _turnover_mean(w)

    assert (tmp_path / "weights_turnover_budget_governed.csv").exists()
    assert (tmp_path / "turnover_before.csv").exists()
    assert (tmp_path / "turnover_after.csv").exists()
    assert (tmp_path / "turnover_budget_rolling_after.csv").exists()
    info = json.loads((tmp_path / "turnover_governor_auto_info.json").read_text(encoding="utf-8"))
    assert info.get("source") == "build_final_portfolio:auto"
