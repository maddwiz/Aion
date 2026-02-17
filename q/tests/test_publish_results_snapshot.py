import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.publish_results_snapshot as prs


def test_publish_results_snapshot_writes_expected_files(tmp_path: Path, monkeypatch):
    qroot = tmp_path / "q"
    runs = qroot / "runs_plus"
    data = qroot / "data"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    np.savetxt(runs / "wf_oos_returns.csv", np.random.default_rng(1).normal(0.0004, 0.01, 600), delimiter=",")
    np.savetxt(runs / "daily_returns.csv", np.random.default_rng(2).normal(0.0005, 0.01, 600), delimiter=",")
    pd.DataFrame({"DATE": pd.date_range("2024-01-01", periods=650, freq="D"), "Close": np.linspace(100, 120, 650)}).to_csv(
        data / "SPY.csv", index=False
    )
    pd.DataFrame({"DATE": pd.date_range("2024-01-01", periods=650, freq="D"), "Close": np.linspace(80, 140, 650)}).to_csv(
        data / "QQQ.csv", index=False
    )
    pd.DataFrame({"global_governor": [0.9, 0.8, 1.0], "quality_governor": [1.0, 0.95, 0.97]}).to_csv(
        runs / "final_governor_trace.csv", index=False
    )
    (runs / "strict_oos_validation.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (runs / "cost_stress_validation.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (runs / "final_portfolio_info.json").write_text(json.dumps({"runtime_total_scalar_mean": 0.4}), encoding="utf-8")

    monkeypatch.setattr(prs, "ROOT", qroot)
    monkeypatch.setattr(prs, "RUNS", runs)
    monkeypatch.setattr(prs, "DATA", data)
    monkeypatch.setattr(prs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prs, "RESULTS", tmp_path / "results")
    prs.RESULTS.mkdir(parents=True, exist_ok=True)

    rc = prs.main()
    assert rc == 0
    assert (prs.RESULTS / "walkforward_metrics.json").exists()
    assert (prs.RESULTS / "walkforward_equity.csv").exists()
    assert (prs.RESULTS / "benchmarks_metrics.csv").exists()
    assert (prs.RESULTS / "governor_compound_summary.json").exists()
