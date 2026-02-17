import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_external_holdout_validation as ehv


def _write_price_csv(path: Path, n: int = 220, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 16.0, n)
    close = 100.0 + np.cumsum(0.2 * np.sin(x) + 0.4 * rng.standard_normal(n))
    df = pd.DataFrame({"Date": pd.date_range("2022-01-03", periods=n, freq="B"), "Close": close})
    df.to_csv(path, index=False)


def test_external_holdout_validation_writes_metrics(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    holdout = root / "data_holdout"
    holdout.mkdir(parents=True, exist_ok=True)

    np.savetxt(runs / "portfolio_weights_final.csv", np.array([[0.6, 0.4]], dtype=float), delimiter=",")
    (runs / "asset_names.csv").write_text("symbol\nAAA\nBBB\n", encoding="utf-8")
    _write_price_csv(holdout / "AAA.csv", n=220, seed=4)
    _write_price_csv(holdout / "BBB.csv", n=220, seed=7)

    monkeypatch.setattr(ehv, "ROOT", root)
    monkeypatch.setattr(ehv, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_DIR", str(holdout))
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_MIN_ROWS", "100")

    rc = ehv.main()
    assert rc == 0

    out = json.loads((runs / "external_holdout_validation.json").read_text(encoding="utf-8"))
    assert out["ok"] is True
    assert int(out["rows"]) >= 100
    assert int(out["assets"]) == 2
    assert "metrics_external_holdout_net" in out
    assert (runs / "external_holdout_returns.csv").exists()


def test_external_holdout_validation_handles_missing_source(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    np.savetxt(runs / "portfolio_weights_final.csv", np.array([[1.0]], dtype=float), delimiter=",")
    (runs / "asset_names.csv").write_text("symbol\nAAA\n", encoding="utf-8")

    monkeypatch.setattr(ehv, "ROOT", root)
    monkeypatch.setattr(ehv, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_DIR", str(root / "missing_holdout"))

    rc = ehv.main()
    assert rc == 0
    out = json.loads((runs / "external_holdout_validation.json").read_text(encoding="utf-8"))
    assert out["ok"] is False
    assert "reason" in out
