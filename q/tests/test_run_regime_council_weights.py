import numpy as np

import tools.run_regime_council_weights as rrcw


def _synth_labels(t: int) -> np.ndarray:
    out = np.full(t, "choppy", dtype=object)
    out[: t // 3] = "trending"
    out[t // 3 : 2 * t // 3] = "mean_reverting"
    return out


def test_different_regimes_produce_different_weight_vectors():
    rng = np.random.default_rng(7)
    t = 270
    k = 3
    labels = _synth_labels(t)

    votes = rng.normal(0.0, 1.0, size=(t, k))
    votes = np.tanh(votes)
    ret = np.zeros(t, dtype=float)
    for i in range(1, t):
        if labels[i] == "trending":
            ret[i] = 0.010 * np.sign(votes[i - 1, 0])
        elif labels[i] == "mean_reverting":
            ret[i] = 0.010 * np.sign(votes[i - 1, 1])
        else:
            ret[i] = 0.006 * np.sign(votes[i - 1, 2])

    w, info = rrcw.compute_regime_council_weights(
        votes,
        ret,
        labels,
        min_regime_days=20,
        train_min=120,
        test_step=21,
        embargo=5,
        eta=0.8,
    )

    assert w.shape == (t, k)
    tr = w[labels == "trending"].mean(axis=0)
    mr = w[labels == "mean_reverting"].mean(axis=0)
    assert tr[0] > mr[0]
    assert mr[1] > tr[1]
    assert len(info["folds"]) > 0


def test_walk_forward_respects_embargo_gap():
    rng = np.random.default_rng(11)
    t = 220
    k = 4
    votes = np.tanh(rng.normal(size=(t, k)))
    ret = rng.normal(0.0, 0.01, size=t)
    labels = _synth_labels(t)

    _, info = rrcw.compute_regime_council_weights(
        votes,
        ret,
        labels,
        min_regime_days=18,
        train_min=100,
        test_step=20,
        embargo=7,
        eta=0.5,
    )

    assert len(info["folds"]) > 0
    assert all(int(f["embargo_gap"]) >= 7 for f in info["folds"])


def test_rare_regime_falls_back_to_global_weights():
    rng = np.random.default_rng(19)
    t = 240
    k = 3
    votes = np.tanh(rng.normal(size=(t, k)))
    ret = rng.normal(0.0, 0.01, size=t)
    labels = np.full(t, "choppy", dtype=object)
    labels[:90] = "trending"
    labels[90:180] = "mean_reverting"
    labels[180:185] = "squeeze"  # too rare for min_regime_days below

    _, info = rrcw.compute_regime_council_weights(
        votes,
        ret,
        labels,
        min_regime_days=25,
        train_min=120,
        test_step=15,
        embargo=5,
    )

    assert int(info["regime_fallback_counts"]["squeeze"]) > 0


def test_output_dimensions_match_signal_matrix():
    rng = np.random.default_rng(23)
    t = 180
    k = 5
    votes = np.tanh(rng.normal(size=(t, k)))
    ret = rng.normal(0.0, 0.01, size=t)
    labels = _synth_labels(t)

    w, info = rrcw.compute_regime_council_weights(
        votes,
        ret,
        labels,
        min_regime_days=15,
        train_min=90,
        test_step=18,
        embargo=5,
    )

    assert w.shape == votes.shape
    assert int(info["rows"]) == t
    assert int(info["cols"]) == k


def test_dynamic_regime_classifier_emits_fold_thresholds():
    rng = np.random.default_rng(31)
    t = 260
    k = 4
    votes = np.tanh(rng.normal(size=(t, k)))
    # Deliberate volatility regime shift so fold thresholds adapt.
    ret = np.concatenate(
        [
            rng.normal(0.0, 0.004, size=t // 2),
            rng.normal(0.0, 0.030, size=t - (t // 2)),
        ]
    )

    w, info = rrcw.compute_regime_council_weights(
        votes,
        ret,
        labels=None,
        min_regime_days=15,
        train_min=100,
        test_step=20,
        embargo=5,
        eta=0.6,
        dynamic_regime_classifier=True,
    )

    assert w.shape == (t, k)
    assert bool(info.get("dynamic_regime_classifier", False))
    folds = info.get("folds", [])
    assert len(folds) > 1
    assert all("classifier_thresholds" in f for f in folds)

    sqv = [float(f["classifier_thresholds"]["squeeze_vol_z_max"]) for f in folds]
    assert max(sqv) - min(sqv) > 1e-6


def test_missing_labels_can_fallback_without_dynamic_classifier():
    rng = np.random.default_rng(41)
    t = 190
    k = 3
    votes = np.tanh(rng.normal(size=(t, k)))
    ret = rng.normal(0.0, 0.01, size=t)

    w, info = rrcw.compute_regime_council_weights(
        votes,
        ret,
        labels=None,
        min_regime_days=15,
        train_min=90,
        test_step=18,
        embargo=5,
        dynamic_regime_classifier=False,
    )

    assert w.shape == (t, k)
    assert not bool(info.get("dynamic_regime_classifier", True))
