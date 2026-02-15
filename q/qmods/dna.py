# qmods/dna.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

# headless plotting
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None


@dataclass
class DNAConfig:
    fast: int = 20    # ~1 trading month
    slow: int = 126   # ~6 months
    step: int = 21    # compare to ~1 month ago


def _clean_series(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def fft_topk_dna(x, topk: int = 12) -> dict:
    """
    Compact spectral fingerprint for drift/similarity comparisons.
    """
    a = _clean_series(x)
    if a.size == 0:
        return {"idx": [], "amp": []}
    if a.size < 16:
        a = np.pad(a, (0, 16 - a.size), mode="edge")
    a = a - float(np.mean(a))
    spec = np.abs(np.fft.rfft(a))
    if spec.size == 0:
        return {"idx": [], "amp": []}
    k = int(max(1, min(int(topk), spec.size)))
    idx = np.argpartition(spec, -k)[-k:]
    amp = spec[idx]
    order = np.argsort(idx)
    idx = idx[order].astype(int)
    amp = amp[order].astype(float)
    den = float(np.sum(np.abs(amp))) + 1e-12
    amp = (amp / den).tolist()
    return {"idx": idx.tolist(), "amp": amp}


def _dna_to_dense(dna) -> dict[int, float]:
    if isinstance(dna, dict):
        idx = dna.get("idx", [])
        amp = dna.get("amp", [])
    elif isinstance(dna, (list, tuple)) and len(dna) == 2:
        idx, amp = dna
    else:
        return {}
    try:
        idx_arr = np.asarray(idx, dtype=int).ravel()
        amp_arr = np.asarray(amp, dtype=float).ravel()
    except Exception:
        return {}
    n = min(len(idx_arr), len(amp_arr))
    out = {}
    for i, a in zip(idx_arr[:n], amp_arr[:n]):
        if not np.isfinite(a):
            continue
        out[int(i)] = float(a)
    return out


def dna_distance(a, b) -> float:
    """
    1 - cosine similarity between two DNA fingerprints.
    """
    da = _dna_to_dense(a)
    db = _dna_to_dense(b)
    keys = sorted(set(da.keys()) | set(db.keys()))
    if not keys:
        return 0.0
    va = np.array([da.get(k, 0.0) for k in keys], dtype=float)
    vb = np.array([db.get(k, 0.0) for k in keys], dtype=float)
    den = float(np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12
    cos = float(np.dot(va, vb) / den)
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(1.0 - cos)


def _moments(ret: pd.Series, w: int):
    r = ret.dropna()
    mu = r.rolling(w).mean()
    sd = r.rolling(w).std(ddof=1)
    skew = r.rolling(w).apply(
        lambda x: float(pd.Series(x).skew()) if len(x) >= 3 else np.nan,
        raw=False
    )
    return mu, sd, skew


def _latent_for_symbol(ret: pd.Series, cfg: DNAConfig) -> pd.DataFrame:
    mu_f, sd_f, sk_f = _moments(ret, cfg.fast)
    mu_s, sd_s, _     = _moments(ret, cfg.slow)
    Z = pd.concat([mu_f, mu_s, sd_f, sd_s, sk_f], axis=1)
    Z.columns = ["mu_fast", "mu_slow", "vol_fast", "vol_slow", "skew_fast"]
    return Z


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def _drift_series(Z: pd.DataFrame, step: int) -> pd.Series:
    """1 - cosine(latent_t, latent_{t-step}); 0=no change, higher=more change."""
    vals = []
    idx = Z.index
    for i in range(len(idx)):
        j = i - step
        if j < 0:
            vals.append(np.nan)
            continue
        a = Z.iloc[i].to_numpy(dtype=float)
        b = Z.iloc[j].to_numpy(dtype=float)
        a = np.nan_to_num(a, nan=0.0)
        b = np.nan_to_num(b, nan=0.0)
        vals.append(1.0 - _cosine(a, b))
    return pd.Series(vals, index=idx, name="dna_drift")


def compute_dna(
    prices: pd.DataFrame,
    out_json: str = "runs_plus/dna_drift.json",
    out_png: str = "runs_plus/dna_drift.png"
):
    """
    Writes:
      - runs_plus/dna_drift.json  (per-symbol time series; date keys as YYYY-MM-DD strings)
      - runs_plus/dna_drift.png   (avg drift chart)
    """
    cfg = DNAConfig()
    ret = prices.pct_change()
    drift_map: dict[str, dict[str, float]] = {}
    avg_series = []

    for sym in prices.columns:
        Z = _latent_for_symbol(ret[sym], cfg)
        d = _drift_series(Z, cfg.step)
        # Convert Timestamp keys -> ISO strings for JSON
        d_clean = d.dropna()
        d_str_keys = {ts.strftime("%Y-%m-%d"): float(val) for ts, val in d_clean.items()}
        drift_map[sym] = d_str_keys
        avg_series.append(d)

    # ensure output dir
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    # write JSON (keys are strings now)
    Path(out_json).write_text(json.dumps({"dna_drift": drift_map}, indent=2))

    # average drift plot (smoothed)
    if avg_series:
        df = pd.concat(avg_series, axis=1)
        df.columns = prices.columns
        avg_drift = df.mean(axis=1).rolling(5).mean()
        if plt is not None:
            plt.figure(figsize=(8, 3))
            avg_drift.plot()
            plt.title("DNA Drift (avg across symbols)")
            plt.xlabel("")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
        else:
            Path(out_png).parent.mkdir(parents=True, exist_ok=True)
            Path(out_png).touch(exist_ok=True)

    return out_json, out_png
