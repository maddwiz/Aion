#!/usr/bin/env python3
# Nonlinear fusion of councils (tiny MLP)
# Reads:  runs_plus/council_votes.csv, returns like run_meta_stack
# Writes: runs_plus/synapses_pred.csv
# Appends a small card.

import numpy as np
from pathlib import Path

import json
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qmods.synapses_small import SynapseSmall

RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def load_first(paths):
    for p in paths:
        f = ROOT/p
        if f.exists():
            try:
                return np.loadtxt(f, delimiter=",")
            except:
                try:
                    return np.loadtxt(f, delimiter=",", skiprows=1)
                except:
                    pass
    return None

def append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html"]:
        p = ROOT/name
        if not p.exists(): continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt+card
        p.write_text(txt, encoding="utf-8")
        print(f"✅ Appended card to {name}")

if __name__ == "__main__":
    V = load_first(["runs_plus/council_votes.csv"])
    y = load_first([
        "runs_plus/target_returns.csv",
        "runs_plus/daily_returns.csv",
        "daily_returns.csv",
        "portfolio_daily_returns.csv"
    ])
    if V is None or y is None:
        print("(!) Missing council_votes or returns. Skipping."); raise SystemExit(0)
    if V.ndim == 1: V = V.reshape(-1,1)
    y = np.asarray(y).ravel()
    T = min(len(y), V.shape[0]); V = V[:T]; y = y[:T]

    nn = SynapseSmall(hidden=12, lr=0.008, reg=2e-3, epochs=400, patience=40, grad_clip=2.0).fit(V, y)
    pred = nn.predict(V)
    conf = nn.predict_confidence(V)
    np.savetxt(RUNS/"synapses_pred.csv", pred, delimiter=",")
    np.savetxt(RUNS/"synapses_confidence.csv", conf, delimiter=",")

    corr = float(np.corrcoef(pred[1:], y[1:])[0,1]) if len(pred) > 2 else 0.0
    summary = {
        "rows": int(T),
        "features": int(V.shape[1]),
        "train_loss": float(nn.last_train_loss_ or 0.0),
        "val_loss": float(nn.last_val_loss_ or 0.0),
        "pred_y_corr": corr,
        "mean_confidence": float(np.mean(conf)) if len(conf) else 0.0,
    }
    (RUNS / "synapses_summary.json").write_text(json.dumps(summary, indent=2))

    html = (
        f"<p>SynapsesSmall trained on {T} rows, K={V.shape[1]}.</p>"
        f"<p>corr(pred,y)={corr:.3f}, mean_conf={summary['mean_confidence']:.3f}</p>"
    )
    append_card("Neural Synapses (Tiny MLP) ✔", html)
    print(f"✅ Wrote {RUNS/'synapses_pred.csv'}")
    print(f"✅ Wrote {RUNS/'synapses_confidence.csv'}")
    print(f"✅ Wrote {RUNS/'synapses_summary.json'}")
