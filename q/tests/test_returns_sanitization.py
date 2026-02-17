import numpy as np
from pathlib import Path

import tools.make_returns_and_weights as mrw
import tools.rebuild_asset_matrix as ram


def test_make_returns_sanitize_clips_outliers():
    r = np.array([0.01, -2.0, 1e9, np.nan, np.inf, -np.inf], float)
    out, n_clip = mrw.sanitize_returns(r, clip_abs=0.35)
    assert n_clip >= 2
    assert np.isfinite(out).all()
    assert float(np.min(out)) >= -0.95
    assert float(np.max(out)) <= 0.35


def test_rebuild_asset_matrix_sanitize_clips_outliers():
    r = np.array([-10.0, -0.5, 0.02, 9.0], float)
    out, n_clip = ram._sanitize_returns(r)
    assert n_clip == 2
    assert float(np.min(out)) >= -0.95
    assert float(np.max(out)) <= ram.RET_CLIP_ABS


def test_make_returns_collect_data_files_prefers_data_new(monkeypatch, tmp_path: Path):
    d = tmp_path / "data"
    dn = tmp_path / "data_new"
    d.mkdir(parents=True, exist_ok=True)
    dn.mkdir(parents=True, exist_ok=True)
    (d / "SPY.csv").write_text("Date,Close\n2024-01-01,1\n", encoding="utf-8")
    (dn / "SPY.csv").write_text("Date,Close\n2024-01-01,2\n", encoding="utf-8")
    (dn / "TLT.csv").write_text("Date,Close\n2024-01-01,3\n", encoding="utf-8")
    monkeypatch.setattr(mrw, "DATA_DIRS", [d, dn])
    out = mrw._collect_data_files()
    by_sym = {sym: str(fp) for sym, fp in out}
    assert set(by_sym.keys()) == {"SPY", "TLT"}
    assert by_sym["SPY"].endswith(str(dn / "SPY.csv"))
