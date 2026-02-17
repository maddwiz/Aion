import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.build_final_portfolio as bfp
import tools.run_asset_class_diversification as acd


def _class_share_mean(w: np.ndarray, classes: list[str], cls: str) -> float:
    arr = np.asarray(w, float)
    idx = [i for i, c in enumerate(classes) if str(c).upper() == str(cls).upper()]
    if not idx:
        return 0.0
    gross = np.sum(np.abs(arr), axis=1)
    cls_gross = np.sum(np.abs(arr[:, idx]), axis=1)
    ratio = cls_gross / np.maximum(gross, 1e-9)
    return float(np.mean(ratio))


def test_run_asset_class_diversification_outputs_and_reduces_eq_weight(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    t = 220
    names = ["SPY", "TLT", "GLD", "EURUSD"]
    rng = np.random.default_rng(42)
    r = np.column_stack(
        [
            rng.normal(0.0004, 0.030, size=t),  # EQ high vol
            rng.normal(0.0001, 0.010, size=t),  # RATES lower vol
            rng.normal(0.0002, 0.012, size=t),  # COMMOD
            rng.normal(0.0001, 0.009, size=t),  # FX
        ]
    )
    w0 = np.tile(np.array([0.90, 0.04, 0.03, 0.03], dtype=float), (t, 1))

    np.savetxt(runs / "asset_returns.csv", r, delimiter=",")
    np.savetxt(runs / "weights_tail_blend.csv", w0, delimiter=",")
    pd.DataFrame({"asset": names}).to_csv(runs / "asset_names.csv", index=False)

    monkeypatch.setattr(acd, "ROOT", tmp_path)
    monkeypatch.setattr(acd, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = acd.main()
    assert rc == 0
    out = np.loadtxt(runs / "weights_asset_class_diversified.csv", delimiter=",")
    cmap = pd.read_csv(runs / "asset_class_map_used.csv")
    info = json.loads((runs / "asset_class_diversification_info.json").read_text(encoding="utf-8"))
    classes = [str(x).upper() for x in cmap["asset_class"].tolist()]

    assert out.shape == w0.shape
    assert int(info.get("assets", 0)) == len(names)
    assert "EQ" in [str(x).upper() for x in info.get("classes", [])]
    # Under this synthetic setup, class diversification should reduce EQ concentration.
    assert _class_share_mean(out, classes, "EQ") < _class_share_mean(w0, classes, "EQ")


def test_build_final_base_candidates_gate_asset_class_source(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    np.savetxt(runs / "weights_asset_class_diversified.csv", np.ones((5, 3)), delimiter=",")
    np.savetxt(runs / "weights_tail_blend.csv", np.zeros((5, 3)), delimiter=",")

    monkeypatch.setattr(bfp, "ROOT", root)
    monkeypatch.delenv("Q_ENABLE_ASSET_CLASS_DIVERSIFICATION", raising=False)
    cands_off = bfp._base_weight_candidates()
    assert "runs_plus/weights_asset_class_diversified.csv" not in cands_off
    w_off, src_off = bfp.first_mat(cands_off)
    assert w_off is not None
    assert src_off == "runs_plus/weights_tail_blend.csv"

    monkeypatch.setenv("Q_ENABLE_ASSET_CLASS_DIVERSIFICATION", "1")
    cands_on = bfp._base_weight_candidates()
    assert "runs_plus/weights_asset_class_diversified.csv" in cands_on
    w_on, src_on = bfp.first_mat(cands_on)
    assert w_on is not None
    assert src_on == "runs_plus/weights_asset_class_diversified.csv"
