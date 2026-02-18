import json
from pathlib import Path

import numpy as np

import tools.run_confirmation_delay as rcd


def test_signal_flip_starts_low_then_ramps_up():
    w = np.array(
        [
            [0.6, 0.2],
            [0.5, 0.2],
            [0.4, 0.2],
            [-0.5, -0.2],
            [-0.6, -0.2],
            [-0.6, -0.1],
            [-0.5, -0.1],
        ],
        dtype=float,
    )

    s, info = rcd.compute_confirmation_delay_scalar(
        w,
        floor=0.20,
        ramp_days=2.0,
        fast_confirm=False,
        lookback=5,
        mag_change=0.20,
        min_exposure=0.01,
        proxy_returns=None,
    )

    assert s.shape[0] == w.shape[0]
    assert np.allclose(s[:3], np.ones(3))
    assert abs(float(s[3]) - 0.20) < 1e-12
    assert float(s[4]) > float(s[3])
    assert float(s[5]) > float(s[4])
    assert float(np.max(s)) <= 1.0 + 1e-12
    assert int(info["signal_reset_days"]) >= 1


def test_fast_confirm_jumps_to_one_on_breakout():
    # Enter long after flat period, then immediate positive breakout.
    w = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.2],
            [0.6, 0.2],
            [0.6, 0.2],
        ],
        dtype=float,
    )
    proxy_ret = np.array([0.0, 0.0, -0.002, 0.040, 0.001], dtype=float)

    s, _ = rcd.compute_confirmation_delay_scalar(
        w,
        floor=0.15,
        ramp_days=4.0,
        fast_confirm=True,
        lookback=2,
        mag_change=0.15,
        min_exposure=0.01,
        proxy_returns=proxy_ret,
    )

    assert abs(float(s[2]) - 0.15) < 1e-12
    assert float(s[3]) == 1.0


def test_flat_signal_stays_at_one():
    w = np.full((12, 3), 0.1, dtype=float)

    s, _ = rcd.compute_confirmation_delay_scalar(
        w,
        floor=0.25,
        ramp_days=3.0,
        fast_confirm=True,
        lookback=5,
        mag_change=0.30,
        min_exposure=0.01,
        proxy_returns=np.zeros(12, dtype=float),
    )

    assert np.allclose(s, np.ones(12))


def test_missing_inputs_emits_neutral_scalars(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    np.savetxt(runs / "asset_returns.csv", np.zeros((6, 3), dtype=float), delimiter=",")

    monkeypatch.setattr(rcd, "ROOT", root)
    monkeypatch.setattr(rcd, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = rcd.main()
    assert rc == 0

    out = np.loadtxt(runs / "confirmation_delay_scalar.csv", delimiter=",").ravel()
    info = json.loads((runs / "confirmation_delay_info.json").read_text(encoding="utf-8"))

    assert out.shape[0] == 6
    assert np.allclose(out, np.ones(6))
    assert info["ok"] is False
    assert info["fallback_reason"] == "missing_weights"


def test_parameter_clipping(monkeypatch):
    monkeypatch.setenv("Q_CONFIRMATION_FLOOR", "0.90")
    monkeypatch.setenv("Q_CONFIRMATION_RAMP_DAYS", "0.10")
    monkeypatch.setenv("Q_CONFIRMATION_FAST_CONFIRM", "0")
    monkeypatch.setenv("Q_CONFIRMATION_LOOKBACK", "99")

    params = rcd._params_from_env()
    assert abs(float(params["floor"]) - 0.50) < 1e-12
    assert abs(float(params["ramp_days"]) - 0.5) < 1e-12
    assert bool(params["fast_confirm"]) is False
    assert int(params["lookback"]) == 30
