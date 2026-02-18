#!/usr/bin/env python3
"""
Governor parameter walk-forward validation (opt-in).

This script evaluates runtime-governor parameter combinations on expanding
walk-forward folds:
  - each fold selects the best candidate by train-window score
  - then records out-of-sample test-window metrics for the selected candidate

Outputs:
  - runs_plus/governor_walkforward_metrics.json
"""

from __future__ import annotations

import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import tools.run_runtime_combo_search as rcs

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


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


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(float(os.getenv(name, str(default))))
    except Exception:
        v = int(default)
    return int(np.clip(v, lo, hi))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.getenv(name, str(default)))
    except Exception:
        v = float(default)
    return float(np.clip(v, lo, hi))


def _sample_grid(grid: list[tuple], max_combos: int) -> list[tuple]:
    if len(grid) <= max_combos:
        return grid
    idx = np.linspace(0, len(grid) - 1, num=max_combos, dtype=int)
    keep: list[tuple] = []
    seen = set()
    for i_raw in idx.tolist():
        i = int(i_raw)
        if i in seen:
            continue
        seen.add(i)
        keep.append(grid[i])
    return keep


def _build_candidate_grid() -> tuple[list[dict], dict]:
    floors = rcs._parse_csv_floats(
        os.getenv("Q_GOVERNOR_WF_FLOORS", os.getenv("Q_RUNTIME_SEARCH_FLOORS", "0.18,0.20,0.22")),
        0.0,
        1.0,
    )
    if not floors:
        floors = [0.18]
    flags = rcs._parse_csv_tokens(
        os.getenv(
            "Q_GOVERNOR_WF_FLAGS",
            os.getenv(
                "Q_RUNTIME_SEARCH_FLAGS",
                "uncertainty_sizing,global_governor,quality_governor,heartbeat_scaler",
            ),
        )
    )
    class_enables = rcs._parse_csv_bools(os.getenv("Q_GOVERNOR_WF_CLASS_ENABLES", ""))
    if not class_enables:
        class_enables = rcs._default_class_enable_grid()
    if not class_enables:
        class_enables = [0]

    macro_strengths = rcs._parse_csv_floats(
        os.getenv("Q_GOVERNOR_WF_MACRO_STRENGTHS", os.getenv("Q_RUNTIME_SEARCH_MACRO_STRENGTHS", "0.0")),
        0.0,
        2.0,
    ) or [0.0]
    capacity_strengths = rcs._parse_csv_floats(
        os.getenv("Q_GOVERNOR_WF_CAPACITY_STRENGTHS", os.getenv("Q_RUNTIME_SEARCH_CAPACITY_STRENGTHS", "0.0")),
        0.0,
        2.0,
    ) or [0.0]
    macro_blends = rcs._parse_csv_floats(
        os.getenv("Q_GOVERNOR_WF_MACRO_BLENDS", os.getenv("Q_RUNTIME_SEARCH_MACRO_BLENDS", "0.0")),
        0.0,
        1.0,
    ) or [0.0]
    max_combos = _env_int(
        "Q_GOVERNOR_WF_MAX_COMBOS",
        _env_int("Q_RUNTIME_SEARCH_MAX_COMBOS", 128, 8, 5000),
        4,
        2000,
    )

    bit_combos = list(itertools.product([0, 1], repeat=len(flags))) if flags else [tuple()]
    full_grid = list(
        itertools.product(
            floors,
            bit_combos,
            class_enables,
            macro_strengths,
            capacity_strengths,
            macro_blends,
        )
    )
    sampled = _sample_grid(full_grid, max_combos=max_combos)

    candidates = []
    for floor, bits, use_asset_class, macro_strength, capacity_strength, macro_blend in sampled:
        disabled = [f for f, b in zip(flags, bits) if int(b) == 1]
        candidates.append(
            {
                "runtime_total_floor": float(floor),
                "disable_governors": [str(x) for x in disabled],
                "search_flag_count": int(len(flags)),
                "enable_asset_class_diversification": bool(int(use_asset_class) == 1),
                "macro_proxy_strength": float(macro_strength),
                "capacity_impact_strength": float(capacity_strength),
                "uncertainty_macro_shock_blend": float(macro_blend),
            }
        )

    grid_info = {
        "floors": [float(x) for x in floors],
        "flags": list(flags),
        "class_enables": [int(x) for x in class_enables],
        "macro_strengths": [float(x) for x in macro_strengths],
        "capacity_strengths": [float(x) for x in capacity_strengths],
        "macro_blends": [float(x) for x in macro_blends],
        "full_grid_total": int(len(full_grid)),
        "sampled_grid_total": int(len(sampled)),
        "max_combos": int(max_combos),
    }
    return candidates, grid_info


def _build_folds(total_rows: int, train_min: int, test_size: int, max_folds: int) -> list[dict]:
    n = int(max(0, total_rows))
    if n <= 2:
        return []
    train_min = int(np.clip(train_min, 2, max(2, n - 1)))
    test_size = int(np.clip(test_size, 1, max(1, n - 1)))

    folds: list[dict] = []
    train_end = train_min
    while train_end + test_size <= n:
        fold = {
            "train_start": 0,
            "train_end": int(train_end),
            "test_start": int(train_end),
            "test_end": int(train_end + test_size),
        }
        folds.append(fold)
        train_end += test_size

    if not folds and train_min < n:
        folds.append(
            {
                "train_start": 0,
                "train_end": int(train_min),
                "test_start": int(train_min),
                "test_end": int(n),
            }
        )

    if max_folds > 0 and len(folds) > max_folds:
        folds = folds[-max_folds:]
    return folds


def _candidate_env(candidate: dict, base_env: dict) -> dict:
    env = dict(base_env)
    env["Q_RUNTIME_TOTAL_FLOOR"] = str(float(candidate["runtime_total_floor"]))
    env["Q_DISABLE_GOVERNORS"] = ",".join(candidate["disable_governors"])
    env["Q_ENABLE_ASSET_CLASS_DIVERSIFICATION"] = "1" if bool(candidate["enable_asset_class_diversification"]) else "0"
    env["Q_MACRO_PROXY_STRENGTH"] = str(float(candidate["macro_proxy_strength"]))
    env["Q_CAPACITY_IMPACT_STRENGTH"] = str(float(candidate["capacity_impact_strength"]))
    env["Q_UNCERTAINTY_MACRO_SHOCK_BLEND"] = str(float(candidate["uncertainty_macro_shock_blend"]))
    return env


def _evaluate_candidate(
    candidate: dict,
    base_env: dict,
    eval_combo_fn=None,
    returns_loader_fn=None,
) -> tuple[dict, np.ndarray | None]:
    fn_eval = eval_combo_fn or rcs._eval_combo
    fn_ret = returns_loader_fn or rcs._load_daily_returns_for_governor_eval
    env = _candidate_env(candidate, base_env)
    out = fn_eval(env)
    ret = fn_ret()
    if ret is None:
        return out, None
    r = np.asarray(ret, float).ravel()
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return out, r


def _fold_train_score(metrics: dict, candidate: dict) -> tuple[float, dict]:
    hit_target = _env_float("Q_GOVERNOR_WF_TARGET_HIT", 0.49, 0.0, 1.0)
    hit_weight = _env_float("Q_GOVERNOR_WF_HIT_WEIGHT", 0.75, 0.0, 10.0)
    mdd_ref = _env_float("Q_GOVERNOR_WF_MDD_REF", 0.04, 0.001, 2.0)
    mdd_penalty = _env_float("Q_GOVERNOR_WF_MDD_PENALTY", 4.0, 0.0, 50.0)
    base_score = (
        float(metrics.get("sharpe", 0.0))
        + hit_weight * (float(metrics.get("hit_rate", 0.0)) - hit_target)
        - mdd_penalty * max(0.0, abs(float(metrics.get("max_drawdown", 0.0))) - mdd_ref)
    )
    _, cdetail = rcs._runtime_complexity_penalty(candidate)
    score = float(base_score - float(cdetail.get("complexity_penalty", 0.0)))
    detail = {"base_score": float(base_score), **cdetail}
    return score, detail


def _rows_ok(eval_out: dict) -> bool:
    rc_ok = all(int(x.get("code", 1)) == 0 for x in (eval_out.get("rc") or []))
    return (
        rc_ok
        and bool(eval_out.get("promotion_ok", False))
        and bool(eval_out.get("cost_stress_ok", False))
        and bool(eval_out.get("health_ok", False))
        and int(eval_out.get("health_alerts_hard", 999)) == 0
    )


def run_governor_walkforward(eval_combo_fn=None, returns_loader_fn=None) -> dict:
    base_ret = rcs._load_daily_returns_for_governor_eval()
    if base_ret is None:
        return {
            "ok": False,
            "reason": "missing_daily_returns",
            "rows_total": 0,
            "folds": [],
            "oos_metrics": {"sharpe": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0, "n": 0},
        }
    base_ret = np.asarray(base_ret, float).ravel()
    base_ret = np.nan_to_num(base_ret, nan=0.0, posinf=0.0, neginf=0.0)
    total_rows = int(base_ret.size)
    if total_rows < 30:
        return {
            "ok": False,
            "reason": "insufficient_rows",
            "rows_total": total_rows,
            "folds": [],
            "oos_metrics": rcs._returns_metrics(base_ret),
        }

    train_min_default = max(126, total_rows // 3)
    test_size_default = max(21, min(126, max(10, total_rows // 8)))
    train_min = _env_int("Q_GOVERNOR_WF_TRAIN_MIN", train_min_default, 20, max(20, total_rows - 1))
    test_size = _env_int("Q_GOVERNOR_WF_TEST_SIZE", test_size_default, 5, max(5, total_rows - 1))
    max_folds = _env_int("Q_GOVERNOR_WF_MAX_FOLDS", 12, 1, 200)

    folds = _build_folds(total_rows=total_rows, train_min=train_min, test_size=test_size, max_folds=max_folds)
    if not folds:
        return {
            "ok": False,
            "reason": "no_valid_folds",
            "rows_total": total_rows,
            "folds": [],
            "oos_metrics": rcs._returns_metrics(base_ret),
        }

    candidates, grid_info = _build_candidate_grid()
    if not candidates:
        return {
            "ok": False,
            "reason": "empty_grid",
            "rows_total": total_rows,
            "grid": grid_info,
            "folds": [],
            "oos_metrics": rcs._returns_metrics(base_ret),
        }

    base_env = rcs._base_runtime_env()
    try:
        rcs._refresh_friction_calibration(base_env)
        base_env = rcs._base_runtime_env()
    except Exception:
        pass

    fold_results = []
    oos_segments: list[np.ndarray] = []
    skipped_folds = 0

    for i, fold in enumerate(folds, start=1):
        best = None
        best_score = -1e12
        evaluated = 0
        valid = 0
        train_end = int(fold["train_end"])
        test_start = int(fold["test_start"])
        test_end = int(fold["test_end"])
        for cand in candidates:
            try:
                eval_out, ret = _evaluate_candidate(
                    cand,
                    base_env=base_env,
                    eval_combo_fn=eval_combo_fn,
                    returns_loader_fn=returns_loader_fn,
                )
            except Exception:
                continue
            if ret is None or int(ret.size) < test_end:
                continue
            train_r = ret[:train_end]
            test_r = ret[test_start:test_end]
            if train_r.size < 10 or test_r.size < 5:
                continue
            train_m = rcs._returns_metrics(train_r)
            test_m = rcs._returns_metrics(test_r)
            score, sdetail = _fold_train_score(train_m, cand)
            evaluated += 1
            is_valid = _rows_ok(eval_out)
            if is_valid:
                valid += 1
            else:
                score -= 1.0
            if score > best_score:
                best_score = float(score)
                best = {
                    "candidate": dict(cand),
                    "train_metrics": train_m,
                    "test_metrics": test_m,
                    "score": float(score),
                    "score_detail": dict(sdetail),
                    "valid_row": bool(is_valid),
                    "test_returns": np.asarray(test_r, float),
                }

        if best is None:
            skipped_folds += 1
            continue

        oos_segments.append(np.asarray(best["test_returns"], float))
        fold_results.append(
            {
                "fold": int(i),
                "train_rows": int(train_end),
                "test_rows": int(test_end - test_start),
                "train_start": int(fold["train_start"]),
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
                "candidates_evaluated": int(evaluated),
                "candidates_valid": int(valid),
                "best_score": float(best["score"]),
                "best_valid_row": bool(best["valid_row"]),
                "best_params": best["candidate"],
                "train_metrics": best["train_metrics"],
                "test_metrics": best["test_metrics"],
                "score_detail": best["score_detail"],
            }
        )
        print(
            f"… fold {i}/{len(folds)} "
            f"train={train_end} test={test_end - test_start} "
            f"score={best['score']:.3f} sharpe_oos={best['test_metrics'].get('sharpe', 0.0):.3f}"
        )

    if oos_segments:
        oos = np.concatenate([np.asarray(x, float).ravel() for x in oos_segments])
        oos_metrics = rcs._returns_metrics(oos)
    else:
        oos = np.zeros(0, dtype=float)
        oos_metrics = {"sharpe": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0, "n": 0}

    return {
        "ok": bool(len(fold_results) > 0),
        "reason": "ok" if fold_results else "no_fold_selected",
        "rows_total": int(total_rows),
        "folds_requested": int(len(folds)),
        "folds_completed": int(len(fold_results)),
        "folds_skipped": int(skipped_folds),
        "grid": grid_info,
        "settings": {
            "train_min": int(train_min),
            "test_size": int(test_size),
            "max_folds": int(max_folds),
            "target_hit": _env_float("Q_GOVERNOR_WF_TARGET_HIT", 0.49, 0.0, 1.0),
            "hit_weight": _env_float("Q_GOVERNOR_WF_HIT_WEIGHT", 0.75, 0.0, 10.0),
            "mdd_ref": _env_float("Q_GOVERNOR_WF_MDD_REF", 0.04, 0.001, 2.0),
            "mdd_penalty": _env_float("Q_GOVERNOR_WF_MDD_PENALTY", 4.0, 0.0, 50.0),
        },
        "folds": fold_results,
        "oos_metrics": oos_metrics,
        "oos_rows": int(oos.size),
    }


def main() -> int:
    enabled = _env_bool("Q_GOVERNOR_WALKFORWARD_ENABLED", False)
    if not enabled:
        print("… skip (disabled): set Q_GOVERNOR_WALKFORWARD_ENABLED=1 to run")
        return 0

    out = run_governor_walkforward()
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["enabled"] = True
    (RUNS / "governor_walkforward_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    om = out.get("oos_metrics", {}) if isinstance(out.get("oos_metrics"), dict) else {}
    html = (
        f"<p>Folds={int(out.get('folds_completed', 0))}/{int(out.get('folds_requested', 0))}, "
        f"OOS Sharpe={float(om.get('sharpe', 0.0)):.3f}, "
        f"Hit={float(om.get('hit_rate', 0.0)):.3f}, "
        f"MaxDD={float(om.get('max_drawdown', 0.0)):.3f}, "
        f"N={int(om.get('n', out.get('oos_rows', 0)))}.</p>"
    )
    _append_card("Governor Walk-Forward ✔", html)
    print(f"✅ Wrote {RUNS/'governor_walkforward_metrics.json'}")
    if not bool(out.get("ok", False)):
        print(f"(!) Governor walk-forward finished without selected folds: {out.get('reason', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
