from pathlib import Path

import numpy as np

from qmods.dreams import save_dream_png, save_dream_video


def test_dreams_compat_writes_png_and_gif(tmp_path: Path):
    series = 100.0 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.0, 0.01, 180)))
    png = tmp_path / "dream.png"
    save_dream_png(series, png)
    assert png.exists()

    save_dream_video(series, tmp_path, frames=24, step=3, fps=10)
    assert (tmp_path / "dream.gif").exists()
