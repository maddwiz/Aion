#!/usr/bin/env python3
# Parameter stability, turnover cost, council disagreement

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class StabilityResult:
    keep_mask: np.ndarray
    stability_score: float


@dataclass
class TurnoverGovernedResult:
    weights: np.ndarray
    turnover_before: np.ndarray
    turnover_after: np.ndarray
    scale_applied: np.ndarray


@dataclass
class TurnoverBudgetResult:
    weights: np.ndarray
    turnover_before: np.ndarray
    turnover_after: np.ndarray
    scale_applied: np.ndarray
    rolling_turnover_after: np.ndarray


def parameter_stability_filter(params_history: np.ndarray, thresh: float = 0.6) -> StabilityResult:
    # params_history: [n_windows, n_params]
    if params_history.ndim != 2:
        raise ValueError("params_history must be 2D [n_windows, n_params]")
    stds = np.std(params_history, axis=0)
    med = np.median(params_history, axis=0)
    keep = (stds <= (np.abs(med) + 1e-8) * (1.0 - thresh)) | (stds < 1e-3)
    score = float(np.mean(keep))
    return StabilityResult(keep_mask=keep, stability_score=score)

def turnover_cost_penalty(weights_t: np.ndarray, fee_bps: float = 5.0) -> float:
    # weights_t: [T, N]
    if weights_t.ndim != 2 or weights_t.shape[0] < 2:
        return 0.0
    turns = np.sum(np.abs(np.diff(weights_t, axis=0)), axis=1)  # L1 turnover per step
    avg_turn = float(np.mean(turns)) if len(turns) else 0.0
    cost = - (fee_bps / 10000.0) * avg_turn * 252.0  # negative Sharpe adj
    return cost

def disagreement_gate(votes: np.ndarray, clamp=(0.5, 1.0)) -> float:
    # votes: [K] council outputs in [-1,1]
    v = votes.ravel()
    dispersion = float(np.std(v))
    return max(clamp[0], clamp[1] - dispersion)


def apply_turnover_governor(weights_t: np.ndarray, max_step_turnover: float = 0.35) -> TurnoverGovernedResult:
    """
    Enforce an L1 turnover budget per step:
      sum_i |w_t[i] - w_{t-1}[i]| <= max_step_turnover
    """
    w = np.asarray(weights_t, float)
    if w.ndim != 2 or w.shape[0] < 2:
        empty = np.zeros(max(0, w.shape[0] - 1), dtype=float)
        ones = np.ones_like(empty)
        return TurnoverGovernedResult(weights=w.copy(), turnover_before=empty, turnover_after=empty, scale_applied=ones)

    cap = float(max(0.0, max_step_turnover))
    out = w.copy()
    t_before = np.zeros(w.shape[0] - 1, dtype=float)
    t_after = np.zeros(w.shape[0] - 1, dtype=float)
    scales = np.ones(w.shape[0] - 1, dtype=float)

    for t in range(1, w.shape[0]):
        prev = out[t - 1]
        target = w[t]
        delta = target - prev
        turn = float(np.sum(np.abs(delta)))
        t_before[t - 1] = turn
        if cap > 0.0 and turn > cap:
            alpha = cap / (turn + 1e-12)
            scales[t - 1] = alpha
            out[t] = prev + alpha * delta
        else:
            out[t] = target
        t_after[t - 1] = float(np.sum(np.abs(out[t] - out[t - 1])))

    return TurnoverGovernedResult(
        weights=out,
        turnover_before=t_before,
        turnover_after=t_after,
        scale_applied=scales,
    )


def apply_turnover_budget_governor(
    weights_t: np.ndarray,
    max_step_turnover: float = 0.35,
    budget_window: int = 5,
    budget_limit: float = 1.00,
) -> TurnoverBudgetResult:
    """
    Enforce both:
      1) step cap: sum_i |w_t - w_{t-1}| <= max_step_turnover
      2) rolling budget: sum_{k=t-window+1..t} turnover_k <= budget_limit
    """
    w = np.asarray(weights_t, float)
    if w.ndim != 2 or w.shape[0] < 2:
        empty = np.zeros(max(0, w.shape[0] - 1), dtype=float)
        ones = np.ones_like(empty)
        return TurnoverBudgetResult(
            weights=w.copy(),
            turnover_before=empty,
            turnover_after=empty,
            scale_applied=ones,
            rolling_turnover_after=empty,
        )

    cap = float(max(0.0, max_step_turnover))
    win = int(max(1, budget_window))
    lim = float(max(0.0, budget_limit))

    out = w.copy()
    Tm1 = w.shape[0] - 1
    t_before = np.zeros(Tm1, dtype=float)
    t_after = np.zeros(Tm1, dtype=float)
    scales = np.ones(Tm1, dtype=float)
    roll_after = np.zeros(Tm1, dtype=float)

    for t in range(1, w.shape[0]):
        prev = out[t - 1]
        target = w[t]
        delta = target - prev
        turn = float(np.sum(np.abs(delta)))
        t_before[t - 1] = turn

        # Remaining turnover budget for current window.
        j0 = max(0, (t - 1) - (win - 1))
        used = float(np.sum(t_after[j0 : t - 1])) if (t - 1) > j0 else 0.0
        avail = max(0.0, lim - used)

        eff_cap = cap
        if lim > 0.0:
            eff_cap = min(eff_cap, avail)

        if eff_cap > 0.0 and turn > eff_cap:
            alpha = eff_cap / (turn + 1e-12)
            scales[t - 1] = alpha
            out[t] = prev + alpha * delta
        elif eff_cap <= 0.0:
            scales[t - 1] = 0.0
            out[t] = prev
        else:
            out[t] = target

        t_after[t - 1] = float(np.sum(np.abs(out[t] - out[t - 1])))
        j1 = max(0, (t - 1) - (win - 1))
        roll_after[t - 1] = float(np.sum(t_after[j1 : t]))

    return TurnoverBudgetResult(
        weights=out,
        turnover_before=t_before,
        turnover_after=t_after,
        scale_applied=scales,
        rolling_turnover_after=roll_after,
    )


def regime_governor_from_returns(
    returns_t: np.ndarray,
    lookback: int = 63,
    min_scale: float = 0.45,
    max_scale: float = 1.10,
    dna_state: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a smooth exposure governor from realized volatility, drawdown, and optional DNA regime state.
    dna_state: optional {-1,0,1} series where +1 means stressed regime.
    """
    r = np.asarray(returns_t, float).ravel()
    T = len(r)
    if T == 0:
        return np.asarray([], float)

    lb = int(max(5, lookback))
    vol = np.zeros(T, float)
    for t in range(T):
        j = max(0, t - lb + 1)
        w = r[j : t + 1]
        vol[t] = float(np.nanstd(w, ddof=1)) if len(w) > 1 else 0.0

    v_ok = vol[np.isfinite(vol)]
    thr = float(np.nanquantile(v_ok, 0.75)) if v_ok.size else 0.0
    vs = float(np.nanstd(v_ok) + 1e-12) if v_ok.size else 1.0
    z = (vol - thr) / vs
    highvol_pen = 1.0 / (1.0 + np.exp(-z))  # sigmoid in [0,1]

    eq = np.cumprod(1.0 + np.clip(r, -0.95, 0.95))
    peak = np.maximum.accumulate(eq)
    dd = (eq / (peak + 1e-12) - 1.0).clip(-1.0, 0.0)
    dd_pen = np.clip(np.abs(dd) / 0.12, 0.0, 1.0)

    if dna_state is not None:
        ds = np.asarray(dna_state, float).ravel()
        if len(ds) < T:
            pad = np.zeros(T, float)
            pad[-len(ds) :] = ds if len(ds) else 0.0
            ds = pad
        else:
            ds = ds[-T:]
        dna_pen = np.clip((ds + 1.0) / 2.0, 0.0, 1.0)  # -1 -> 0, +1 -> 1
    else:
        dna_pen = np.zeros(T, float)

    raw = 1.05 - 0.45 * highvol_pen - 0.30 * dd_pen - 0.20 * dna_pen
    raw = np.clip(raw, min_scale, max_scale)

    # Smooth governor to avoid fast leverage jumps.
    alpha = 0.18
    out = np.zeros(T, float)
    out[0] = raw[0]
    for t in range(1, T):
        out[t] = (1.0 - alpha) * out[t - 1] + alpha * raw[t]
    return np.clip(out, min_scale, max_scale)


def stability_governor(
    weights_t: np.ndarray,
    votes_t: np.ndarray | None = None,
    lookback: int = 21,
    min_scale: float = 0.55,
    max_scale: float = 1.05,
) -> np.ndarray:
    """
    Penalize unstable behavior:
    - high step turnover
    - high council disagreement (if votes provided)
    """
    w = np.asarray(weights_t, float)
    if w.ndim != 2 or w.shape[0] == 0:
        return np.asarray([], float)
    T = w.shape[0]

    turn = np.zeros(T, float)
    if T > 1:
        turn[1:] = np.sum(np.abs(np.diff(w, axis=0)), axis=1)
    lb = int(max(5, lookback))
    turn_m = np.zeros(T, float)
    for t in range(T):
        j = max(0, t - lb + 1)
        turn_m[t] = float(np.nanmean(turn[j : t + 1]))
    base = float(np.nanmedian(turn_m) + 1e-12)
    turn_pen = np.tanh(turn_m / (2.0 * base + 1e-12))

    if votes_t is not None:
        v = np.asarray(votes_t, float)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        d = np.std(v, axis=1)
        if len(d) < T:
            pad = np.zeros(T, float)
            pad[-len(d) :] = d if len(d) else 0.0
            d = pad
        else:
            d = d[-T:]
        dis_pen = np.clip(d / 0.75, 0.0, 1.0)
    else:
        dis_pen = np.zeros(T, float)

    raw = 1.02 - 0.50 * turn_pen - 0.30 * dis_pen
    raw = np.clip(raw, min_scale, max_scale)

    alpha = 0.22
    out = np.zeros(T, float)
    out[0] = raw[0]
    for t in range(1, T):
        out[t] = (1.0 - alpha) * out[t - 1] + alpha * raw[t]
    return np.clip(out, min_scale, max_scale)
