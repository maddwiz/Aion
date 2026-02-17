import json
from pathlib import Path

import pandas as pd

import tools.ingest_multi_asset_csv_bundle as ims


def _write_csv(path: Path, n: int = 40):
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2022-01-03", periods=n, freq="B"),
            "Close": [100.0 + i * 0.1 for i in range(n)],
        }
    )
    df.to_csv(path, index=False)


def test_ingest_multi_asset_csv_bundle_copies_files(tmp_path: Path, monkeypatch):
    root = tmp_path
    src = tmp_path / "src_bundle"
    src.mkdir(parents=True, exist_ok=True)
    _write_csv(src / "SPY.csv")
    _write_csv(src / "TLT.csv")
    _write_csv(src / "GLD.csv")

    monkeypatch.setattr(ims, "ROOT", root)
    monkeypatch.setattr(ims, "DATA_NEW", root / "data_new")
    monkeypatch.setattr(ims, "RUNS", root / "runs_plus")
    ims.DATA_NEW.mkdir(parents=True, exist_ok=True)
    ims.RUNS.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("Q_MULTI_ASSET_SOURCE_DIR", str(src))
    monkeypatch.setattr("sys.argv", ["ingest_multi_asset_csv_bundle.py"])

    rc = ims.main()
    assert rc == 0
    report = json.loads((ims.RUNS / "multi_asset_ingest_report.json").read_text(encoding="utf-8"))
    assert int(report["files_copied"]) == 3
    assert (ims.DATA_NEW / "SPY.csv").exists()
    assert (ims.DATA_NEW / "TLT.csv").exists()
    assert (ims.RUNS / "cluster_map.csv").exists()


def test_ingest_multi_asset_csv_bundle_handles_missing_source(tmp_path: Path, monkeypatch):
    root = tmp_path
    monkeypatch.setattr(ims, "ROOT", root)
    monkeypatch.setattr(ims, "DATA_NEW", root / "data_new")
    monkeypatch.setattr(ims, "RUNS", root / "runs_plus")
    ims.DATA_NEW.mkdir(parents=True, exist_ok=True)
    ims.RUNS.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("Q_MULTI_ASSET_SOURCE_DIR", str(root / "does_not_exist"))
    monkeypatch.setattr("sys.argv", ["ingest_multi_asset_csv_bundle.py"])
    rc = ims.main()
    assert rc == 0
    assert not (ims.RUNS / "multi_asset_ingest_report.json").exists()
