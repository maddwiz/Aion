import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd


def _load_tool_module():
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "backtest_skimmer.py"
    spec = importlib.util.spec_from_file_location("backtest_skimmer_tool", tool_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mk_cfg():
    return SimpleNamespace(
        EQUITY_START=5000.0,
        SKIMMER_STOP_ATR_MULTIPLE=1.5,
        SKIMMER_RISK_PER_TRADE=0.005,
        SKIMMER_MAX_POSITION_PCT=0.03,
        SKIMMER_PARTIAL_PROFIT_R=1.0,
        SKIMMER_PARTIAL_PROFIT_FRAC=0.50,
        SKIMMER_TRAILING_STOP_ATR=1.2,
        SKIMMER_MAX_TRADES_SESSION=8,
        SKIMMER_MAX_DAILY_LOSS_PCT=0.015,
        SKIMMER_NO_ENTRY_BEFORE_CLOSE_MIN=45,
        SKIMMER_FORCE_CLOSE_BEFORE_MIN=10,
        SKIMMER_ENTRY_THRESHOLD=0.58,
    )


def _write_intraday_csv(path: Path, rows: int = 150):
    idx = pd.date_range("2026-01-05 09:30:00", periods=rows, freq="1min")
    base = 100.0 + np.arange(rows) * 0.02
    df = pd.DataFrame(
        {
            "datetime": idx,
            "open": base,
            "high": base + 0.05,
            "low": base - 0.05,
            "close": base + 0.01,
            "volume": np.full(rows, 25000.0),
        }
    )
    df.to_csv(path, index=False)


def test_run_backtest_writes_outputs(tmp_path):
    mod = _load_tool_module()
    data_dir = tmp_path / "intraday"
    out_dir = tmp_path / "state"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_intraday_csv(data_dir / "AAPL_1m_20260105.csv")

    trades_path, summary_path = mod.run_backtest(
        data_dir=data_dir,
        output_dir=out_dir,
        symbols=["AAPL"],
        max_sessions=1,
        cfg_mod=_mk_cfg(),
    )

    assert trades_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["files_processed"] == 1
    assert summary["sessions_processed"] == 1
    assert "sharpe" in summary
