#!/usr/bin/env python3
# Builds runs_plus/daily_returns.csv by multiplying weights × asset_returns
# Chooses best available weights automatically.

import numpy as np
from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def load_mat(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except:
        a = np.loadtxt(p, delimiter=",", skiprows=1)
    if a.ndim == 1: a = a.reshape(-1,1)
    return a

def first_mat(paths):
    for rel in paths:
        a = load_mat(rel)
        if a is not None: return a, rel
    return None, None


def _infer_asset_class(sym: str) -> str:
    s = str(sym or "").upper().replace("-", "").replace("_", "").replace("/", "")
    if not s:
        return "EQ"
    if s in {"VIX", "VIX9D", "VIX3M", "VIXCLS", "UVXY", "VXX"}:
        return "VOL"
    if s in {"LQD", "HYG", "JNK", "BND", "AGG", "MBB", "HYGTR", "LQDTR"} or s.endswith("_TR"):
        return "CREDIT"
    if s in {"TLT", "IEF", "SHY", "VGSH", "DGS2", "DGS3MO", "DGS5", "DGS10", "DGS30", "TY", "ZN", "ZB"}:
        return "RATES"
    if s in {"GLD", "SLV", "USO", "UNG", "CORN", "WEAT", "CPER", "DBC", "DBA", "XLE", "XOP"}:
        return "COMMOD"
    if len(s) == 6 and s[:3] in {"USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"} and s[3:] in {
        "USD",
        "EUR",
        "JPY",
        "GBP",
        "CHF",
        "CAD",
        "AUD",
        "NZD",
    }:
        return "FX"
    if s.startswith("BTC") or s.startswith("ETH") or s in {"IBIT", "FBTC", "BITO"}:
        return "CRYPTO"
    return "EQ"


def _load_asset_names(n: int) -> list[str]:
    p = RUNS / "asset_names.csv"
    if p.exists():
        try:
            import csv

            with p.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.reader(fh))
            if rows and rows[0]:
                vals = [str(r[0]).strip().upper() for r in rows[1:] if r]
                vals = [x for x in vals if x]
                if len(vals) >= n:
                    return vals[:n]
        except Exception:
            pass
    return [f"A{i:03d}" for i in range(n)]


def _load_asset_classes(names: list[str]) -> list[str]:
    p = RUNS / "asset_class_map_used.csv"
    mapping = {}
    if p.exists():
        try:
            import csv

            with p.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            for r in rows:
                a = str(r.get("asset", "")).strip().upper()
                c = str(r.get("asset_class", "")).strip().upper()
                if a and c:
                    mapping[a] = c
        except Exception:
            mapping = {}
    out = []
    for nm in names:
        out.append(mapping.get(str(nm).strip().upper(), _infer_asset_class(nm)))
    return out


def _class_cost_multipliers(n: int) -> tuple[np.ndarray, dict]:
    enabled = str(os.getenv("Q_COST_CLASS_MODEL_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}
    if (not enabled) or n <= 0:
        return np.ones(max(0, int(n)), dtype=float), {"enabled": False, "classes": []}
    names = _load_asset_names(n)
    classes = _load_asset_classes(names)
    mult_cfg = {
        "EQ": float(np.clip(float(os.getenv("Q_COST_CLASS_EQ_MULT", "1.00")), 0.25, 5.0)),
        "CREDIT": float(np.clip(float(os.getenv("Q_COST_CLASS_CREDIT_MULT", "1.10")), 0.25, 5.0)),
        "RATES": float(np.clip(float(os.getenv("Q_COST_CLASS_RATES_MULT", "0.85")), 0.25, 5.0)),
        "COMMOD": float(np.clip(float(os.getenv("Q_COST_CLASS_COMMOD_MULT", "1.20")), 0.25, 5.0)),
        "FX": float(np.clip(float(os.getenv("Q_COST_CLASS_FX_MULT", "0.90")), 0.25, 5.0)),
        "CRYPTO": float(np.clip(float(os.getenv("Q_COST_CLASS_CRYPTO_MULT", "1.60")), 0.25, 5.0)),
        "VOL": float(np.clip(float(os.getenv("Q_COST_CLASS_VOL_MULT", "1.40")), 0.25, 5.0)),
    }
    mult = np.ones(n, dtype=float)
    for i, c in enumerate(classes):
        mult[i] = float(mult_cfg.get(str(c).upper(), 1.0))
    class_counts = {}
    for c in classes:
        k = str(c).upper()
        class_counts[k] = int(class_counts.get(k, 0) + 1)
    return mult, {
        "enabled": True,
        "classes": sorted(set(str(x).upper() for x in classes)),
        "class_counts": class_counts,
        "class_multipliers": mult_cfg,
    }


def _safe_float(x, default):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def resolve_cost_params(*, runs_dir: Path | None = None) -> dict:
    runs = RUNS if runs_dir is None else Path(runs_dir)
    cal_path = runs / "friction_calibration.json"
    cal = _load_json(cal_path)
    rec = cal.get("recommendation") if isinstance(cal.get("recommendation"), dict) else {}
    cal_ok = bool(cal.get("ok", rec.get("ok", False)))
    rec_base = _safe_float(rec.get("recommended_cost_base_bps"), None)
    rec_vol = _safe_float(rec.get("recommended_cost_vol_scaled_bps"), None)
    baseline = _safe_float(rec.get("baseline_cost_base_bps"), None)

    default_base = 10.0
    default_vol = 0.0
    source_base = "default"
    source_vol = "default"
    if cal_ok and rec_base is not None:
        default_base = float(rec_base)
        source_base = "friction_calibration"
    elif baseline is not None:
        default_base = float(baseline)
        source_base = "friction_baseline"
    if cal_ok and rec_vol is not None:
        default_vol = float(rec_vol)
        source_vol = "friction_calibration"

    legacy_bps = str(os.getenv("Q_COST_BPS", "")).strip()
    env_base = str(os.getenv("Q_COST_BASE_BPS", "")).strip()
    env_vol = str(os.getenv("Q_COST_VOL_SCALED_BPS", "")).strip()
    if legacy_bps and not env_base:
        base_bps = float(np.clip(_safe_float(legacy_bps, default_base), 0.0, 100.0))
        source_base = "env:Q_COST_BPS"
    elif env_base:
        base_bps = float(np.clip(_safe_float(env_base, default_base), 0.0, 100.0))
        source_base = "env:Q_COST_BASE_BPS"
    else:
        base_bps = float(np.clip(default_base, 0.0, 100.0))

    if env_vol:
        vol_scaled_bps = float(np.clip(_safe_float(env_vol, default_vol), 0.0, 100.0))
        source_vol = "env:Q_COST_VOL_SCALED_BPS"
    else:
        vol_scaled_bps = float(np.clip(default_vol, 0.0, 100.0))

    return {
        "base_bps": float(base_bps),
        "vol_scaled_bps": float(vol_scaled_bps),
        "source_base": source_base,
        "source_vol": source_vol,
        "friction_calibration_ok": bool(cal_ok),
        "friction_calibration_path": str(cal_path) if cal_path.exists() else "",
    }


def build_costed_daily_returns(
    W: np.ndarray,
    A: np.ndarray,
    *,
    base_bps: float,
    vol_scaled_bps: float = 0.0,
    vol_lookback: int = 20,
    vol_ref_daily: float = 0.0063,
    half_turnover: bool = True,
    fixed_daily_fee: float = 0.0,
    cash_yield_annual: float = 0.0,
    cash_exposure_target: float = 1.0,
    asset_cost_multipliers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(W, float)
    a = np.asarray(A, float)
    T = min(w.shape[0], a.shape[0])
    w = w[:T]
    a = a[:T]
    trade_gross = np.sum(w * a, axis=1)
    target = float(max(1e-9, cash_exposure_target))
    gross_abs = np.sum(np.abs(w), axis=1)
    cash_frac = np.clip(1.0 - (gross_abs / target), 0.0, 1.0)
    cash_carry = cash_frac * (float(max(0.0, cash_yield_annual)) / 252.0)
    gross = trade_gross + cash_carry
    turnover = np.zeros(T, dtype=float)
    turnover_asset = np.zeros((T, w.shape[1]), dtype=float)
    if T > 1:
        turnover_asset[1:] = np.abs(np.diff(w, axis=0))
        turnover[1:] = np.sum(turnover_asset[1:], axis=1)
    if bool(half_turnover):
        turnover_asset *= 0.5
        turnover = 0.5 * turnover

    # Rolling realized volatility (daily) for volatility-scaled slippage.
    vlook = int(np.clip(int(vol_lookback), 2, max(2, T)))
    vol = np.zeros(T, dtype=float)
    for t in range(T):
        lo = max(0, t - vlook + 1)
        seg = trade_gross[lo : t + 1]
        vol[t] = float(np.std(seg)) if seg.size else 0.0
    ref = float(max(1e-6, vol_ref_daily))
    vol_ratio = np.clip(vol / ref, 0.0, 6.0)
    eff_bps = np.clip(float(base_bps), 0.0, 10_000.0) + np.clip(float(vol_scaled_bps), 0.0, 10_000.0) * vol_ratio

    if asset_cost_multipliers is not None:
        am = np.asarray(asset_cost_multipliers, float).ravel()
        if am.size == w.shape[1]:
            class_eff = (eff_bps.reshape(-1, 1) * am.reshape(1, -1))
            var_cost = np.sum((class_eff / 10000.0) * turnover_asset, axis=1)
        else:
            var_cost = (eff_bps / 10000.0) * turnover
    else:
        var_cost = (eff_bps / 10000.0) * turnover
    fee = np.full(T, float(max(0.0, fixed_daily_fee)), dtype=float)
    cost = var_cost + fee
    net = gross - cost
    return net, gross, cost, turnover, eff_bps, cash_carry, cash_frac

if __name__ == "__main__":
    A = load_mat("runs_plus/asset_returns.csv")
    if A is None:
        print("(!) runs_plus/asset_returns.csv missing. Run tools/rebuild_asset_matrix.py first.")
        raise SystemExit(0)
    clip_abs = float(np.clip(float(os.getenv("Q_ASSET_RET_CLIP", "0.35")), 0.01, 5.0))
    A_clean = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A_clean = np.clip(A_clean, -0.95, clip_abs)
    clip_events = int(np.sum(np.abs(A_clean - A) > 1e-12))
    A = A_clean

    W, src = first_mat([
        "runs_plus/portfolio_weights_final.csv",
        "runs_plus/tune_best_weights.csv",
        "runs_plus/weights_regime.csv",
        "runs_plus/weights_tail_blend.csv",
        "runs_plus/portfolio_weights.csv",
        "portfolio_weights.csv",
    ])
    if W is None:
        print("(!) No weights found."); raise SystemExit(0)

    # Align T and N
    T = min(A.shape[0], W.shape[0])
    if A.shape[1] != W.shape[1]:
        print(f"(!) Col mismatch: asset_returns N={A.shape[1]} vs weights N={W.shape[1]}.")
        raise SystemExit(0)

    cost_cfg = resolve_cost_params(runs_dir=RUNS)
    base_bps = float(cost_cfg["base_bps"])
    vol_scaled_bps = float(cost_cfg["vol_scaled_bps"])
    vol_lookback = int(np.clip(int(float(os.getenv("Q_COST_VOL_LOOKBACK", "20"))), 2, 252))
    vol_ref_daily = float(np.clip(float(os.getenv("Q_COST_VOL_REF_DAILY", "0.0063")), 1e-5, 0.25))
    half_turnover = str(os.getenv("Q_COST_HALF_TURNOVER", "1")).strip().lower() in {"1", "true", "yes", "on"}
    fixed_daily_fee = float(np.clip(float(os.getenv("Q_FIXED_DAILY_FEE", "0.0")), 0.0, 1.0))
    cash_yield_annual = float(np.clip(float(os.getenv("Q_CASH_YIELD_ANNUAL", "0.0")), 0.0, 0.20))
    cash_exposure_target = float(np.clip(float(os.getenv("Q_CASH_EXPOSURE_TARGET", "1.0")), 0.25, 5.0))
    class_cost_mult, class_cost_info = _class_cost_multipliers(W.shape[1])
    net, gross, cost, turnover, eff_bps, cash_carry, cash_frac = build_costed_daily_returns(
        W[:T],
        A[:T],
        base_bps=base_bps,
        vol_scaled_bps=vol_scaled_bps,
        vol_lookback=vol_lookback,
        vol_ref_daily=vol_ref_daily,
        half_turnover=half_turnover,
        fixed_daily_fee=fixed_daily_fee,
        cash_yield_annual=cash_yield_annual,
        cash_exposure_target=cash_exposure_target,
        asset_cost_multipliers=class_cost_mult,
    )
    np.savetxt(RUNS/"daily_returns.csv", net, delimiter=",")
    np.savetxt(RUNS/"daily_returns_gross.csv", gross, delimiter=",")
    np.savetxt(RUNS/"daily_costs.csv", cost, delimiter=",")
    np.savetxt(RUNS/"daily_turnover.csv", turnover, delimiter=",")
    np.savetxt(RUNS/"daily_effective_cost_bps.csv", eff_bps, delimiter=",")
    np.savetxt(RUNS/"daily_cash_carry.csv", cash_carry, delimiter=",")
    np.savetxt(RUNS/"daily_cash_fraction.csv", cash_frac, delimiter=",")
    (RUNS / "daily_costs_info.json").write_text(
        json.dumps(
            {
                "cost_base_bps": float(base_bps),
                "cost_base_source": str(cost_cfg.get("source_base", "default")),
                "cost_vol_scaled_bps": float(vol_scaled_bps),
                "cost_vol_scaled_source": str(cost_cfg.get("source_vol", "default")),
                "cost_vol_lookback": int(vol_lookback),
                "cost_vol_ref_daily": float(vol_ref_daily),
                "cost_half_turnover": bool(half_turnover),
                "fixed_daily_fee": float(fixed_daily_fee),
                "friction_calibration_ok": bool(cost_cfg.get("friction_calibration_ok", False)),
                "friction_calibration_path": str(cost_cfg.get("friction_calibration_path", "")),
                "cash_yield_annual": float(cash_yield_annual),
                "cash_exposure_target": float(cash_exposure_target),
                "class_cost_model_enabled": bool(class_cost_info.get("enabled", False)),
                "class_cost_classes": list(class_cost_info.get("classes", [])),
                "class_cost_counts": dict(class_cost_info.get("class_counts", {})),
                "class_cost_multipliers": dict(class_cost_info.get("class_multipliers", {})),
                "rows": int(T),
                "mean_cost_daily": float(np.mean(cost)) if T else 0.0,
                "ann_cost_estimate": float(np.mean(cost) * 252.0) if T else 0.0,
                "mean_gross_daily": float(np.mean(gross)) if T else 0.0,
                "mean_net_daily": float(np.mean(net)) if T else 0.0,
                "mean_cash_carry_daily": float(np.mean(cash_carry)) if T else 0.0,
                "ann_cash_carry_estimate": float(np.mean(cash_carry) * 252.0) if T else 0.0,
                "mean_cash_fraction": float(np.mean(cash_frac)) if T else 0.0,
                "mean_turnover": float(np.mean(turnover)) if T else 0.0,
                "mean_effective_cost_bps": float(np.mean(eff_bps)) if T else 0.0,
                "max_effective_cost_bps": float(np.max(eff_bps)) if T else 0.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if clip_events > 0:
        print(f"(!) Clipped {clip_events} extreme asset-return values with |r|>{clip_abs:.3f}")
    print(
        f"✅ Wrote runs_plus/daily_returns.csv (T={T}) from weights='{src}' "
        f"[base_bps={base_bps:.2f} ({cost_cfg.get('source_base','default')}), "
        f"vol_scaled_bps={vol_scaled_bps:.2f} ({cost_cfg.get('source_vol','default')}), "
        f"fixed_daily_fee={fixed_daily_fee:.5f}, "
        f"cash_yield_annual={cash_yield_annual:.4f}]"
    )
