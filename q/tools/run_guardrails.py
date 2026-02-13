#!/usr/bin/env python3
# Computes guardrail summaries + optional DD scaling output.
# Writes:
#  - runs_plus/guardrails_summary.json
#  - runs_plus/weights_dd_scaled.csv (if inputs found)
#  - runs_plus/disagreement_gate.csv (if council_votes.csv found)
# Also appends a small card to report_*.

import json, csv, os
from pathlib import Path
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.guardrails_bundle import (
    apply_turnover_budget_governor,
    apply_turnover_governor,
    disagreement_gate,
    disagreement_gate_series,
    parameter_stability_filter,
    regime_governor_from_returns,
    stability_governor,
    turnover_cost_penalty,
)
from qmods.drawdown_floor import drawdown_floor_series

RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

def _maybe_load_csv(p: Path):
    if p.exists():
        try:
            a = np.loadtxt(p, delimiter=",")
            if a.ndim == 1: a = a.reshape(-1,1)
            return a
        except Exception:
            try:
                a = np.loadtxt(p, delimiter=",", skiprows=1)
                if a.ndim == 1: a = a.reshape(-1,1)
                return a
            except Exception:
                return None
    return None

def _load_first_non_none(paths):
    for p in paths:
        a = _maybe_load_csv(p)
        if a is not None:
            return a
    return None

def _as_series_last_col(a):
    if a is None:
        return None
    x = np.asarray(a, float)
    if x.ndim == 2:
        x = x[:, -1]
    return x.ravel()

def _load_named_numeric_col(path: Path, colname: str):
    if not path.exists():
        return None
    vals = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            rdr = csv.DictReader(f)
            if colname not in (rdr.fieldnames or []):
                return None
            for row in rdr:
                try:
                    vals.append(float(row.get(colname, "")))
                except Exception:
                    continue
    except Exception:
        return None
    return np.asarray(vals, float).ravel() if vals else None

def _append_report_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists(): 
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")
        print(f"✅ Appended guardrails card to {name}")

def main():
    # 1) Parameter stability — runs_plus/params_history.csv (n_windows x n_params)
    params_hist = _maybe_load_csv(RUNS/"params_history.csv")
    if params_hist is not None and params_hist.ndim == 2 and params_hist.shape[0] >= 2:
        stab_res = parameter_stability_filter(params_hist, thresh=0.6)
        stab = {"stability_score": float(stab_res.stability_score),
                "kept_params": int(stab_res.keep_mask.sum()),
                "total_params": int(stab_res.keep_mask.size)}
    else:
        stab = {"note": "params_history.csv not found; skipped"}

    # 2) Turnover cost + turnover governor — portfolio_weights.csv (T x N)
    wts = _load_first_non_none([ROOT/"portfolio_weights.csv", RUNS/"portfolio_weights.csv"])
    wts_for_stability = None
    if wts is not None and wts.shape[0] > 2:
        fee_bps = 5.0
        max_step = float(np.clip(float(os.getenv("TURNOVER_MAX_STEP", "0.35")), 0.0, 10.0))
        budget_window = int(np.clip(int(float(os.getenv("TURNOVER_BUDGET_WINDOW", "5"))), 1, 252))
        budget_limit = float(np.clip(float(os.getenv("TURNOVER_BUDGET_LIMIT", "1.00")), 0.0, 20.0))
        gov = apply_turnover_governor(wts, max_step_turnover=max_step)
        w_governed = gov.weights
        budget_info = {"enabled": False}
        if budget_limit > 0.0:
            bres = apply_turnover_budget_governor(
                gov.weights,
                max_step_turnover=max_step,
                budget_window=budget_window,
                budget_limit=budget_limit,
            )
            w_governed = bres.weights
            np.savetxt(RUNS/"weights_turnover_budget_governed.csv", bres.weights, delimiter=",")
            np.savetxt(RUNS/"turnover_budget_rolling_after.csv", bres.rolling_turnover_after, delimiter=",")
            budget_info = {
                "enabled": True,
                "window": int(budget_window),
                "limit": float(budget_limit),
                "rolling_after_mean": float(np.mean(bres.rolling_turnover_after)) if bres.rolling_turnover_after.size else 0.0,
                "rolling_after_max": float(np.max(bres.rolling_turnover_after)) if bres.rolling_turnover_after.size else 0.0,
            }
        np.savetxt(RUNS/"weights_turnover_governed.csv", w_governed, delimiter=",")
        np.savetxt(RUNS/"turnover_before.csv", gov.turnover_before, delimiter=",")
        np.savetxt(RUNS/"turnover_after.csv", gov.turnover_after, delimiter=",")

        cost_raw = turnover_cost_penalty(wts, fee_bps=fee_bps)
        cost_gov = turnover_cost_penalty(w_governed, fee_bps=fee_bps)
        mean_scale = float(np.mean(gov.scale_applied)) if gov.scale_applied.size else 1.0
        cost = {
            "turnover_cost_sharpe_adj": float(cost_raw),
            "turnover_cost_sharpe_adj_governed": float(cost_gov),
            "turnover_before_mean": float(np.mean(gov.turnover_before)) if gov.turnover_before.size else 0.0,
            "turnover_after_mean": float(np.mean(gov.turnover_after)) if gov.turnover_after.size else 0.0,
            "turnover_scale_mean": mean_scale,
            "turnover_max_step": max_step,
            "turnover_budget": budget_info,
        }
        wts_for_stability = w_governed
    else:
        cost = {"note": "portfolio_weights.csv not found; skipped"}
        if wts is not None:
            wts_for_stability = wts

    # 3) Council disagreement gate — runs_plus/council_votes.csv (T x K)
    votes = _maybe_load_csv(RUNS/"council_votes.csv")
    if votes is not None and votes.shape[0] >= 1:
        shock = _as_series_last_col(_maybe_load_csv(RUNS/"shock_mask.csv"))
        if votes.ndim == 2 and votes.shape[0] >= 8:
            gates = disagreement_gate_series(
                votes,
                clamp=(0.45, 1.0),
                lookback=int(np.clip(int(os.getenv("DISAGREEMENT_LOOKBACK", "63")), 8, 252)),
                smooth=float(np.clip(float(os.getenv("DISAGREEMENT_SMOOTH", "0.85")), 0.0, 0.98)),
                shock_mask=shock,
                shock_alpha=float(np.clip(float(os.getenv("DISAGREEMENT_SHOCK_ALPHA", "0.20")), 0.0, 1.0)),
            )
        else:
            gates = np.array([disagreement_gate(v) for v in votes])
        np.savetxt(RUNS/"disagreement_gate.csv", gates, delimiter=",")
        gate_stats = {"gate_mean": float(np.mean(gates)),
                      "gate_min": float(np.min(gates)),
                      "gate_max": float(np.max(gates))}
    else:
        gate_stats = {"note": "council_votes.csv not found; skipped"}

    # 4) Optional drawdown scaling — runs_plus/cum_pnl.csv + weights
    cum = _maybe_load_csv(RUNS/"cum_pnl.csv")
    if (cum is not None) and (wts is not None) and cum.shape[0] >= wts.shape[0]:
        scale = drawdown_floor_series(cum[:wts.shape[0], 0] if cum.ndim==2 and cum.shape[1]==1 else cum[:wts.shape[0]].ravel(),
                                      floor=-0.12, cut=0.6)
        w_scaled = wts.copy()
        T = min(len(scale), w_scaled.shape[0])
        for t in range(T):
            w_scaled[t] *= scale[t]
        np.savetxt(RUNS/"weights_dd_scaled.csv", w_scaled, delimiter=",")
        dd_out = {"dd_floor": -0.12, "cut": 0.6}
    else:
        dd_out = {"note": "cum_pnl.csv or weights missing; skipped"}

    # 5) Regime and stability governors
    ret_mat = _load_first_non_none([RUNS/"daily_returns.csv", ROOT/"daily_returns.csv", RUNS/"wf_oos_returns.csv"])
    ret = _as_series_last_col(ret_mat)
    if ret is None and cum is not None:
        c = _as_series_last_col(cum)
        if c is not None and len(c) > 1:
            ret = np.diff(c, prepend=c[0])

    dna_state = _load_named_numeric_col(RUNS/"dna_drift.csv", "dna_regime_state")
    if ret is not None and len(ret) >= 20:
        reg_lb = int(np.clip(int(os.getenv("REGIME_LOOKBACK", "63")), 5, 252))
        reg_scale = regime_governor_from_returns(ret, lookback=reg_lb, dna_state=dna_state)
        np.savetxt(RUNS/"regime_governor.csv", reg_scale, delimiter=",")
        regime_stats = {
            "lookback": reg_lb,
            "mean": float(np.mean(reg_scale)),
            "min": float(np.min(reg_scale)),
            "max": float(np.max(reg_scale)),
        }
    else:
        reg_scale = None
        regime_stats = {"note": "daily returns missing; skipped"}

    if wts_for_stability is not None and wts_for_stability.shape[0] >= 5:
        st_lb = int(np.clip(int(os.getenv("STABILITY_LOOKBACK", "21")), 5, 252))
        st_scale = stability_governor(wts_for_stability, votes_t=votes, lookback=st_lb)
        np.savetxt(RUNS/"stability_governor.csv", st_scale, delimiter=",")
        stability_gov = {
            "lookback": st_lb,
            "mean": float(np.mean(st_scale)),
            "min": float(np.min(st_scale)),
            "max": float(np.max(st_scale)),
        }
    else:
        st_scale = None
        stability_gov = {"note": "weights missing; skipped"}

    if reg_scale is not None and st_scale is not None:
        L = min(len(reg_scale), len(st_scale))
        # Blend (not multiply) to avoid over-shrinking exposure in normal regimes.
        g = np.clip(0.55 * reg_scale[:L] + 0.45 * st_scale[:L], 0.45, 1.10)
    elif reg_scale is not None:
        g = np.clip(reg_scale, 0.45, 1.10)
    elif st_scale is not None:
        g = np.clip(st_scale, 0.45, 1.10)
    else:
        g = None

    if g is not None and len(g):
        np.savetxt(RUNS/"global_governor.csv", g, delimiter=",")
        global_gov = {"mean": float(np.mean(g)), "min": float(np.min(g)), "max": float(np.max(g))}
    else:
        global_gov = {"note": "no governors produced"}

    out = {
        "stability": stab,
        "turnover_cost": cost,
        "disagreement_gate": gate_stats,
        "drawdown_scaler": dd_out,
        "regime_governor": regime_stats,
        "stability_governor": stability_gov,
        "global_governor": global_gov,
    }
    (RUNS/"guardrails_summary.json").write_text(json.dumps(out, indent=2))
    print(f"✅ Wrote {RUNS/'guardrails_summary.json'}")

    # 5) Report card
    html_bits = []
    if "stability_score" in stab:
        html_bits.append(f"<p><b>Stability:</b> score {stab['stability_score']:.2f} (kept {stab['kept_params']}/{stab['total_params']})</p>")
    if "turnover_cost_sharpe_adj" in cost:
        html_bits.append(
            f"<p><b>Turnover cost adj:</b> raw {cost['turnover_cost_sharpe_adj']:.4f}, "
            f"governed {cost.get('turnover_cost_sharpe_adj_governed', cost['turnover_cost_sharpe_adj']):.4f} "
            f"(max step {cost.get('turnover_max_step', 0.0):.3f})</p>"
        )
        html_bits.append(
            f"<p><b>Turnover mean:</b> before {cost.get('turnover_before_mean', 0.0):.4f}, "
            f"after {cost.get('turnover_after_mean', 0.0):.4f}, "
            f"scale mean {cost.get('turnover_scale_mean', 1.0):.3f}</p>"
        )
        tb = cost.get("turnover_budget", {}) if isinstance(cost.get("turnover_budget"), dict) else {}
        if tb.get("enabled"):
            html_bits.append(
                f"<p><b>Turnover budget:</b> window {tb.get('window')}, limit {tb.get('limit'):.3f}, "
                f"rolling mean {tb.get('rolling_after_mean', 0.0):.3f}, "
                f"rolling max {tb.get('rolling_after_max', 0.0):.3f}</p>"
            )
    if "gate_mean" in gate_stats:
        html_bits.append(f"<p><b>Council gate:</b> mean {gate_stats['gate_mean']:.3f} (min {gate_stats['gate_min']:.3f}, max {gate_stats['gate_max']:.3f})</p>")
    if "dd_floor" in dd_out:
        html_bits.append(f"<p><b>DD Reallocator:</b> floor {dd_out['dd_floor']}, cut {dd_out['cut']} (weights_dd_scaled.csv)</p>")
    if "mean" in regime_stats:
        html_bits.append(
            f"<p><b>Regime governor:</b> mean {regime_stats['mean']:.3f} "
            f"(min {regime_stats['min']:.3f}, max {regime_stats['max']:.3f})</p>"
        )
    if "mean" in stability_gov:
        html_bits.append(
            f"<p><b>Stability governor:</b> mean {stability_gov['mean']:.3f} "
            f"(min {stability_gov['min']:.3f}, max {stability_gov['max']:.3f})</p>"
        )
    if "mean" in global_gov:
        html_bits.append(
            f"<p><b>Global governor:</b> mean {global_gov['mean']:.3f} "
            f"(min {global_gov['min']:.3f}, max {global_gov['max']:.3f})</p>"
        )
    if html_bits:
        _append_report_card("GUARDRAILS SUMMARY", "\n".join(html_bits))

if __name__ == "__main__":
    main()
