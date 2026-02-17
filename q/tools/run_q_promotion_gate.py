#!/usr/bin/env python3
"""
Promotion gate for Q -> AION overlay promotion.

Reads:
  - runs_plus/strict_oos_validation.json

Writes:
  - runs_plus/q_promotion_gate.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _append_card(title: str, html: str) -> None:
    if str(os.getenv("Q_DISABLE_REPORT_CARDS", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def _select_metrics(val: dict, mode: str) -> tuple[str, dict]:
    net = val.get("metrics_oos_net", {}) if isinstance(val.get("metrics_oos_net"), dict) else {}
    robust = val.get("metrics_oos_robust", {}) if isinstance(val.get("metrics_oos_robust"), dict) else {}
    m = str(mode).strip().lower()
    if m == "robust":
        return "metrics_oos_robust", robust
    if m == "single":
        return "metrics_oos_net", net
    if robust:
        return "metrics_oos_robust", robust
    return "metrics_oos_net", net


def _metric_triplet(obj: dict | None) -> tuple[float, float, float, int]:
    d = obj if isinstance(obj, dict) else {}
    sh = float(d.get("sharpe", 0.0))
    hit = float(d.get("hit_rate", 0.0))
    mdd = float(d.get("max_drawdown", 0.0))
    n = int(d.get("n", 0))
    return sh, hit, mdd, n


def main() -> int:
    src = RUNS / "strict_oos_validation.json"
    val = _load_json(src)
    if not isinstance(val, dict):
        out = {
            "ok": False,
            "source": str(src),
            "reason": "missing_strict_oos_validation",
            "reasons": ["strict_oos_validation_missing"],
        }
        (RUNS / "q_promotion_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"✅ Wrote {RUNS/'q_promotion_gate.json'}")
        print("Promotion gate: FAIL (missing strict OOS validation)")
        return 0

    metric_mode = str(os.getenv("Q_PROMOTION_OOS_MODE", "robust_then_single")).strip().lower()
    metric_source, m = _select_metrics(val, metric_mode)
    sharpe = float(m.get("sharpe", 0.0))
    hit = float(m.get("hit_rate", 0.0))
    mdd = float(m.get("max_drawdown", 0.0))
    n = int(m.get("n", 0))
    latest = val.get("metrics_oos_latest", {}) if isinstance(val.get("metrics_oos_latest"), dict) else {}
    latest_sh, latest_hit, latest_mdd, latest_n = _metric_triplet(latest)

    min_sharpe = float(np.clip(float(os.getenv("Q_PROMOTION_MIN_OOS_SHARPE", "1.00")), -2.0, 10.0))
    min_hit = float(np.clip(float(os.getenv("Q_PROMOTION_MIN_OOS_HIT", "0.49")), 0.0, 1.0))
    max_abs_mdd = float(np.clip(float(os.getenv("Q_PROMOTION_MAX_ABS_MDD", "0.10")), 0.001, 2.0))
    min_n = int(np.clip(int(float(os.getenv("Q_PROMOTION_MIN_OOS_SAMPLES", "252"))), 1, 1000000))
    require_cost_stress = str(os.getenv("Q_PROMOTION_REQUIRE_COST_STRESS", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    require_latest_holdout = str(os.getenv("Q_PROMOTION_REQUIRE_LATEST_HOLDOUT", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    min_latest_sh = float(np.clip(float(os.getenv("Q_PROMOTION_MIN_LATEST_OOS_SHARPE", "0.90")), -2.0, 10.0))
    min_latest_hit = float(np.clip(float(os.getenv("Q_PROMOTION_MIN_LATEST_OOS_HIT", "0.48")), 0.0, 1.0))
    max_latest_abs_mdd = float(np.clip(float(os.getenv("Q_PROMOTION_MAX_LATEST_ABS_MDD", "0.12")), 0.001, 2.0))
    min_latest_n = int(np.clip(int(float(os.getenv("Q_PROMOTION_MIN_LATEST_OOS_SAMPLES", "126"))), 1, 1000000))
    require_external_holdout = str(os.getenv("Q_PROMOTION_REQUIRE_EXTERNAL_HOLDOUT", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    min_external_sh = float(np.clip(float(os.getenv("Q_PROMOTION_MIN_EXTERNAL_HOLDOUT_SHARPE", "0.75")), -2.0, 10.0))
    min_external_hit = float(np.clip(float(os.getenv("Q_PROMOTION_MIN_EXTERNAL_HOLDOUT_HIT", "0.47")), 0.0, 1.0))
    max_external_abs_mdd = float(np.clip(float(os.getenv("Q_PROMOTION_MAX_EXTERNAL_HOLDOUT_ABS_MDD", "0.15")), 0.001, 2.0))
    min_external_n = int(np.clip(int(float(os.getenv("Q_PROMOTION_MIN_EXTERNAL_HOLDOUT_SAMPLES", "126"))), 1, 1000000))

    reasons = []
    if n < min_n:
        reasons.append(f"oos_samples<{min_n} ({n})")
    if sharpe < min_sharpe:
        reasons.append(f"oos_sharpe<{min_sharpe:.2f} ({sharpe:.3f})")
    if hit < min_hit:
        reasons.append(f"oos_hit<{min_hit:.2f} ({hit:.3f})")
    if abs(mdd) > max_abs_mdd:
        reasons.append(f"oos_abs_mdd>{max_abs_mdd:.3f} ({abs(mdd):.3f})")
    if require_latest_holdout:
        if latest_n < min_latest_n:
            reasons.append(f"latest_oos_samples<{min_latest_n} ({latest_n})")
        if latest_sh < min_latest_sh:
            reasons.append(f"latest_oos_sharpe<{min_latest_sh:.2f} ({latest_sh:.3f})")
        if latest_hit < min_latest_hit:
            reasons.append(f"latest_oos_hit<{min_latest_hit:.2f} ({latest_hit:.3f})")
        if abs(latest_mdd) > max_latest_abs_mdd:
            reasons.append(f"latest_oos_abs_mdd>{max_latest_abs_mdd:.3f} ({abs(latest_mdd):.3f})")

    ext_path = RUNS / "external_holdout_validation.json"
    ext = _load_json(ext_path)
    ext_summary = None
    ext_metrics = {}
    if isinstance(ext, dict) and ext:
        ext_metrics = ext.get("metrics_external_holdout_net", {}) if isinstance(ext.get("metrics_external_holdout_net"), dict) else {}
        ext_summary = {
            "ok": bool(ext.get("ok", False)),
            "path": str(ext_path),
            "method": str(ext.get("method", "")),
            "rows": int(ext.get("rows", ext_metrics.get("n", 0))),
            "metrics_external_holdout_net": {
                "sharpe": float(ext_metrics.get("sharpe", 0.0)),
                "hit_rate": float(ext_metrics.get("hit_rate", 0.0)),
                "max_drawdown": float(ext_metrics.get("max_drawdown", 0.0)),
                "n": int(ext_metrics.get("n", 0)),
            },
            "reason": str(ext.get("reason", "")),
        }
    if require_external_holdout:
        if not isinstance(ext, dict) or not ext:
            reasons.append("external_holdout_missing")
        elif not bool(ext.get("ok", False)):
            reasons.append("external_holdout_fail")
        else:
            ext_sh = float(ext_metrics.get("sharpe", 0.0))
            ext_hit = float(ext_metrics.get("hit_rate", 0.0))
            ext_mdd = float(ext_metrics.get("max_drawdown", 0.0))
            ext_n = int(ext_metrics.get("n", ext.get("rows", 0)))
            if ext_n < min_external_n:
                reasons.append(f"external_holdout_samples<{min_external_n} ({ext_n})")
            if ext_sh < min_external_sh:
                reasons.append(f"external_holdout_sharpe<{min_external_sh:.2f} ({ext_sh:.3f})")
            if ext_hit < min_external_hit:
                reasons.append(f"external_holdout_hit<{min_external_hit:.2f} ({ext_hit:.3f})")
            if abs(ext_mdd) > max_external_abs_mdd:
                reasons.append(f"external_holdout_abs_mdd>{max_external_abs_mdd:.3f} ({abs(ext_mdd):.3f})")

    cost_stress_path = RUNS / "cost_stress_validation.json"
    cost_stress = _load_json(cost_stress_path)
    cost_stress_summary = None
    if isinstance(cost_stress, dict) and cost_stress:
        worst = cost_stress.get("worst_case_robust", {})
        cost_stress_summary = {
            "ok": bool(cost_stress.get("ok", False)),
            "path": str(cost_stress_path),
            "worst_case_robust": {
                "sharpe": float((worst or {}).get("sharpe", 0.0)),
                "hit_rate": float((worst or {}).get("hit_rate", 0.0)),
                "max_drawdown": float((worst or {}).get("max_drawdown", 0.0)),
            },
            "thresholds": cost_stress.get("thresholds", {}),
            "reasons": list(cost_stress.get("reasons", [])),
        }
    if require_cost_stress:
        if not isinstance(cost_stress, dict) or not cost_stress:
            reasons.append("cost_stress_missing")
        elif not bool(cost_stress.get("ok", False)):
            reasons.append("cost_stress_fail")

    ok = len(reasons) == 0
    out = {
        "ok": bool(ok),
        "source": str(src),
        "metric_source": metric_source,
        "metric_mode": metric_mode,
        "thresholds": {
            "min_oos_sharpe": float(min_sharpe),
            "min_oos_hit": float(min_hit),
            "max_abs_mdd": float(max_abs_mdd),
            "min_oos_samples": int(min_n),
            "require_latest_holdout": bool(require_latest_holdout),
            "min_latest_oos_sharpe": float(min_latest_sh),
            "min_latest_oos_hit": float(min_latest_hit),
            "max_latest_abs_mdd": float(max_latest_abs_mdd),
            "min_latest_oos_samples": int(min_latest_n),
            "require_external_holdout": bool(require_external_holdout),
            "min_external_holdout_sharpe": float(min_external_sh),
            "min_external_holdout_hit": float(min_external_hit),
            "max_external_holdout_abs_mdd": float(max_external_abs_mdd),
            "min_external_holdout_samples": int(min_external_n),
        },
        "metrics_oos_net": {
            "sharpe": float(sharpe),
            "hit_rate": float(hit),
            "max_drawdown": float(mdd),
            "n": int(n),
        },
        "metrics_oos_latest": {
            "sharpe": float(latest_sh),
            "hit_rate": float(latest_hit),
            "max_drawdown": float(latest_mdd),
            "n": int(latest_n),
        },
        "require_cost_stress": bool(require_cost_stress),
        "cost_stress": cost_stress_summary,
        "external_holdout": ext_summary,
        "reasons": reasons,
    }
    (RUNS / "q_promotion_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    badge = "PASS" if ok else "FAIL"
    color = "#1b8f3a" if ok else "#a91d2b"
    html = (
        f"<p><b style='color:{color}'>Promotion {badge}</b> "
        f"(OOS Sharpe={sharpe:.3f}, Hit={hit:.3f}, MaxDD={mdd:.3f}, N={n}, "
        f"source={metric_source}).</p>"
    )
    html += (
        f"<p>Latest OOS window: Sharpe={latest_sh:.3f}, Hit={latest_hit:.3f}, "
        f"MaxDD={latest_mdd:.3f}, N={latest_n}, required={bool(require_latest_holdout)}.</p>"
    )
    if ext_summary:
        em = ext_summary.get("metrics_external_holdout_net", {})
        html += (
            f"<p>External holdout: ok={bool(ext_summary.get('ok', False))}, "
            f"Sharpe={float(em.get('sharpe', 0.0)):.3f}, "
            f"Hit={float(em.get('hit_rate', 0.0)):.3f}, "
            f"MaxDD={float(em.get('max_drawdown', 0.0)):.3f}, "
            f"N={int(em.get('n', 0))}, required={bool(require_external_holdout)}.</p>"
        )
    if cost_stress_summary:
        wc = cost_stress_summary["worst_case_robust"]
        html += (
            f"<p>Cost stress: ok={bool(cost_stress_summary.get('ok', False))}, "
            f"worst robust Sharpe={float(wc.get('sharpe', 0.0)):.3f}, "
            f"Hit={float(wc.get('hit_rate', 0.0)):.3f}, "
            f"MaxDD={float(wc.get('max_drawdown', 0.0)):.3f}.</p>"
        )
    if reasons:
        html += f"<p>Reasons: {', '.join(reasons)}</p>"
    _append_card("Q Promotion Gate ✔", html)

    print(f"✅ Wrote {RUNS/'q_promotion_gate.json'}")
    print(f"Promotion gate: {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
