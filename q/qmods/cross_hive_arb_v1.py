#!/usr/bin/env python3
import numpy as np

def arb_weights(
    hive_scores: dict,
    alpha=2.0,
    drawdown_penalty: dict | None = None,
    disagreement_penalty: dict | None = None,
    inertia: float = 0.80,
    max_weight: float = 0.70,
    min_weight: float = 0.00,
):
    """
    Softmax allocation over hives based on standardized health scores with penalties,
    plus practical execution constraints:
      - inertia smoothing over time
      - optional min/max per-hive weight clamps
    hive_scores: {name: [T]} base score (higher=better)
    drawdown_penalty: {name: [T]} penalty in [0,1], where 1 is worst
    disagreement_penalty: {name: [T]} penalty in [0,1], where 1 is worst
    Returns: (names, W) where W=[T,H] per-hive weights that sum to 1.
    """
    names = sorted(hive_scores.keys())
    S = np.stack([np.asarray(hive_scores[n], float) for n in names], axis=1)
    mu = S.mean(0, keepdims=True); sd = S.std(0, keepdims=True) + 1e-9
    Z = (S - mu) / sd

    if drawdown_penalty:
        D = np.stack([np.asarray(drawdown_penalty.get(n, np.zeros(S.shape[0])), float) for n in names], axis=1)
        Z = Z - 1.4 * np.clip(D, 0.0, 1.0)
    if disagreement_penalty:
        G = np.stack([np.asarray(disagreement_penalty.get(n, np.zeros(S.shape[0])), float) for n in names], axis=1)
        Z = Z - 1.0 * np.clip(G, 0.0, 1.0)

    W = np.exp(alpha * Z)
    W = W / (W.sum(1, keepdims=True) + 1e-9)

    # Clamp single-hive concentration and ensure small floor for exploration/recovery.
    mn = float(np.clip(min_weight, 0.0, 0.95))
    mx = float(np.clip(max_weight, mn + 1e-6, 1.0))
    if mn > 0.0 or mx < 0.999:
        W = np.clip(W, mn, mx)
        W = W / (W.sum(1, keepdims=True) + 1e-9)

    # Inertia smoothing to lower rebalance churn.
    ib = float(np.clip(inertia, 0.0, 0.98))
    if ib > 0.0 and W.shape[0] > 1:
        for t in range(1, W.shape[0]):
            W[t] = ib * W[t - 1] + (1.0 - ib) * W[t]
            W[t] = W[t] / (W[t].sum() + 1e-9)
    return names, W
