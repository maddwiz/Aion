#!/usr/bin/env python3
import numpy as np


def _safe_std(x, axis=0, eps=1e-8):
    s = np.nanstd(x, axis=axis)
    return np.where(np.isfinite(s), np.maximum(s, eps), 1.0)


class SynapseSmall:
    """
    Tiny 1-hidden-layer MLP to fuse K council signals nonlinearly.
    Uses tanh + L2 reg; 1-step lag fit to reduce leakage.
    """
    def __init__(self, hidden=8, lr=0.01, reg=1e-3, epochs=300, seed=13, patience=30, grad_clip=2.0):
        self.hidden = hidden; self.lr = lr; self.reg = reg; self.epochs = epochs
        self.patience = int(max(5, patience))
        self.grad_clip = float(max(0.1, grad_clip))
        self.rng = np.random.default_rng(seed)
        self.W1 = None; self.b1 = None; self.W2 = None; self.b2 = None
        self.x_mu = None; self.x_sd = None
        self.resid_std_ = 1.0
        self.last_train_loss_ = None
        self.last_val_loss_ = None

    def _init(self, K):
        self.W1 = self.rng.standard_normal((K, self.hidden)) / np.sqrt(K)
        self.b1 = np.zeros(self.hidden)
        self.W2 = self.rng.standard_normal((self.hidden, 1)) / np.sqrt(self.hidden)
        self.b2 = np.zeros(1)

    def _scale_fit(self, X):
        self.x_mu = np.nanmean(X, axis=0)
        self.x_sd = _safe_std(X, axis=0)
        return (X - self.x_mu) / self.x_sd

    def _scale_apply(self, X):
        if self.x_mu is None or self.x_sd is None:
            return X
        return (X - self.x_mu) / self.x_sd

    def _forward(self, X):
        Z1 = np.tanh(X @ self.W1 + self.b1)
        yhat = np.tanh(Z1 @ self.W2 + self.b2)  # bounded output
        return Z1, yhat

    def _loss(self, yhat, y):
        err = yhat - y
        mse = float(np.mean(err**2))
        reg = float(self.reg * ((self.W1**2).mean() + (self.W2**2).mean()))
        return mse + reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).reshape(-1,1)
        if X.ndim != 2 or X.shape[0] < 8:
            self._init(X.shape[1] if X.ndim == 2 else 1)
            return self
        # lag
        X = X[:-1]; y = y[1:]
        T, K = X.shape
        X = self._scale_fit(X)
        self._init(K)

        split = max(10, int(0.80 * T))
        Xtr, ytr = X[:split], y[:split]
        Xva, yva = X[split:], y[split:]

        best = None
        best_val = np.inf
        bad = 0

        for _ in range(self.epochs):
            Z1, yhat = self._forward(Xtr)
            err = yhat - ytr
            dW2 = Z1.T @ err / max(1, len(Xtr)) + self.reg * self.W2
            db2 = err.mean(0)
            dZ1 = (err @ self.W2.T) * (1 - Z1**2)
            dW1 = Xtr.T @ dZ1 / max(1, len(Xtr)) + self.reg * self.W1
            db1 = dZ1.mean(0)

            # grad clipping for stability
            for g in (dW1, db1, dW2, db2):
                np.clip(g, -self.grad_clip, self.grad_clip, out=g)

            self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

            self.last_train_loss_ = self._loss(yhat, ytr)
            if len(Xva) > 0:
                _, yv = self._forward(Xva)
                vloss = self._loss(yv, yva)
            else:
                vloss = self.last_train_loss_
            self.last_val_loss_ = vloss

            if vloss < best_val - 1e-7:
                best_val = vloss
                best = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best is not None:
            self.W1, self.b1, self.W2, self.b2 = best

        # residual sigma on train slice
        _, yfit = self._forward(Xtr)
        self.resid_std_ = float(np.nanstd((yfit - ytr).ravel(), ddof=1))
        if not np.isfinite(self.resid_std_) or self.resid_std_ <= 1e-8:
            self.resid_std_ = 1.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = self._scale_apply(X)
        _, yhat = self._forward(X)
        return yhat.ravel()

    def predict_confidence(self, X):
        p = self.predict(X)
        # confidence from margin vs residual noise
        z = np.abs(p) / (self.resid_std_ + 1e-8)
        return np.tanh(z)
