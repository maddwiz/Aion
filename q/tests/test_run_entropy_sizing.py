from pathlib import Path

import numpy as np

import tools.run_entropy_sizing as res


def test_run_entropy_sizing_scales_by_vote_entropy(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    votes = np.array(
        [
            [1.0, 0.0, 0.0],  # concentrated -> low entropy -> larger scalar
            [1.0, 1.0, 1.0],  # uniform -> high entropy -> smaller scalar
            [0.8, 0.2, 0.0],
        ],
        dtype=float,
    )
    np.savetxt(runs / "council_votes.csv", votes, delimiter=",")

    monkeypatch.setattr(res, "ROOT", root)
    monkeypatch.setattr(res, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = res.main()
    assert rc == 0

    scalar = np.loadtxt(runs / "entropy_sizing_scalar.csv", delimiter=",").ravel()
    assert len(scalar) == votes.shape[0]
    assert float(scalar[0]) > float(scalar[1])
