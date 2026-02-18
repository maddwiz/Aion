import argparse, json, os, pathlib
import numpy as np
from qmods.io import load_close
from qmods.meta_council import meta_council
from qmods.dna import fft_topk_dna
from qmods.heartbeat import heartbeat_bpm
from qmods.drift import rolling_dna_drift
from qmods.dreams import save_dream_png, save_dream_video
from qmods.log import append_growth_log

def strategy_metrics(close, meta_score, cost_bps: float):
    rets = close.pct_change().fillna(0.0).to_numpy(dtype=float)
    pos = np.tanh(np.asarray(meta_score, float))
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    pos_lag = np.roll(pos, 1)
    pos_lag[0] = 0.0
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    cost = float(cost_bps) / 10000.0
    strat = np.nan_to_num(pos_lag * rets - turnover * cost, nan=0.0, posinf=0.0, neginf=0.0)
    hit = float(np.mean(np.sign(pos_lag) == np.sign(rets))) if len(rets) else 0.0
    s = float(np.std(strat, ddof=1)) if len(strat) > 1 else 0.0
    sh = float((np.mean(strat) / s) * np.sqrt(252.0)) if s > 1e-12 else 0.0
    eq = np.cumprod(1.0 + strat)
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0], dtype=float)
    mdd = float(np.min(eq / np.where(peak == 0.0, 1.0, peak) - 1.0)) if len(eq) else 0.0
    return float(hit), float(sh), mdd

def main():
    seed = int(os.getenv("Q_RANDOM_SEED", "42"))
    np.random.seed(seed)
    print(f"Random seed: {seed}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost_bps", type=float, default=1.0)
    ap.add_argument("--dream_frames", type=int, default=90)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    close = load_close(pathlib.Path(args.data) / args.asset)

    # 1) Meta council signal (core + v2 blend)
    meta = meta_council(close, include_v2=True, v2_weight=0.12)

    # 2) DNA + drift + heartbeat
    dna = fft_topk_dna(close.values)
    drift = rolling_dna_drift(close, 126)
    bpm = heartbeat_bpm(close)

    # 3) Council-strategy metrics after transaction costs
    hit, sh, mdd = strategy_metrics(close, meta, cost_bps=float(args.cost_bps))

    # 4) Dream artifacts
    save_dream_png(close.values, outdir/"dream.png")
    save_dream_video(close.values, outdir, frames=int(max(30, args.dream_frames)), step=4, fps=12)

    # 5) Save artifacts
    result = {
        "asset": args.asset,
        "cost_bps": float(args.cost_bps),
        "hit_rate": hit,
        "sharpe": sh,
        "max_dd": mdd,
        "dna": dna,
        "dna_drift_pct": float(drift.ffill().iloc[-1]) if drift.notna().any() else None,
        "heartbeat_bpm_latest": float(bpm.ffill().iloc[-1]) if hasattr(bpm, "notna") and bpm.notna().any() else None,
    }
    (outdir/"summary.json").write_text(json.dumps(result, indent=2))

    # 6) Growth log append
    append_growth_log(result, pathlib.Path("GROWTH_LOG.md"))

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
