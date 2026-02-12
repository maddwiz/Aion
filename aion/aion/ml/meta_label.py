from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


class MetaLabelModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = Path(cfg.STATE_DIR) / "meta_model.json"
        self.weights = np.array([0.0, 0.9, 0.6, 0.4, -0.35, 0.25, 0.15], dtype=float)
        self.feature_names = [
            "bias",
            "confidence",
            "signal_margin",
            "adx_norm",
            "atr_pct",
            "is_trending",
            "is_squeeze",
        ]
        self.load()

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            w = data.get("weights", [])
            if isinstance(w, list) and len(w) == len(self.weights):
                self.weights = np.array([float(v) for v in w], dtype=float)
        except Exception:
            return

    def save(self, samples: int = 0):
        payload = {
            "weights": [float(v) for v in self.weights],
            "samples": int(samples),
        }
        self.path.write_text(json.dumps(payload, indent=2))

    def feature_vector(self, confidence: float, long_conf: float, short_conf: float, adx: float, atr_pct: float, regime: str):
        margin = abs(long_conf - short_conf)
        adx_norm = max(0.0, min(1.0, adx / 45.0))
        is_trending = 1.0 if regime == "trending" else 0.0
        is_squeeze = 1.0 if regime == "squeeze" else 0.0
        return np.array([1.0, confidence, margin, adx_norm, atr_pct, is_trending, is_squeeze], dtype=float)

    def predict_proba(self, confidence: float, long_conf: float, short_conf: float, adx: float, atr_pct: float, regime: str):
        x = self.feature_vector(confidence, long_conf, short_conf, adx, atr_pct, regime)
        return float(self._sigmoid(float(np.dot(self.weights, x))))

    def fit_from_trades(self, trades_csv: Path):
        if not trades_csv.exists():
            return 0
        try:
            df = pd.read_csv(trades_csv)
        except Exception:
            return 0

        if df.empty or "side" not in df.columns or "pnl" not in df.columns:
            return 0

        exits = df[df["side"].astype(str).str.startswith("EXIT")].copy()
        if exits.empty:
            return 0

        for col, default in [
            ("confidence", 0.5),
            ("regime", "mixed"),
            ("pnl", 0.0),
        ]:
            if col not in exits.columns:
                exits[col] = default

        exits["confidence"] = pd.to_numeric(exits["confidence"], errors="coerce").fillna(0.5)
        exits["pnl"] = pd.to_numeric(exits["pnl"], errors="coerce").fillna(0.0)
        exits["regime"] = exits["regime"].fillna("mixed").astype(str)

        # Approximate missing features from available trade logs.
        exits["long_conf"] = np.where(exits["side"].str.contains("SELL"), exits["confidence"], 1 - exits["confidence"])
        exits["short_conf"] = 1 - exits["long_conf"]
        exits["adx"] = 24.0
        exits["atr_pct"] = 0.018

        if len(exits) < self.cfg.META_LABEL_TRAIN_MIN_SAMPLES:
            return 0

        X = []
        y = []
        for _, r in exits.iterrows():
            x = self.feature_vector(
                confidence=float(r["confidence"]),
                long_conf=float(r["long_conf"]),
                short_conf=float(r["short_conf"]),
                adx=float(r["adx"]),
                atr_pct=float(r["atr_pct"]),
                regime=str(r["regime"]),
            )
            X.append(x)
            y.append(1.0 if float(r["pnl"]) > 0 else 0.0)

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        w = self.weights.copy()
        lr = self.cfg.META_LABEL_LR
        for _ in range(self.cfg.META_LABEL_EPOCHS):
            p = self._sigmoid(X @ w)
            grad = (X.T @ (p - y)) / max(len(y), 1)
            w -= lr * grad

        # simple L2 shrink for stability
        w *= 0.995

        self.weights = w
        self.save(samples=len(y))
        return int(len(y))


def safe_logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))
