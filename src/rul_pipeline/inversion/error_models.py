"""Gaussian likelihoods with explicit error models and an explicit amplitude gain.

    y = g F(a_i) + e,   e ~ N(0, diag(v)),   v = v_meas / n_avg + delta^2

* ``v_meas``: acquisition (measurement) variance - an assumed scalar now, or per-sample from
  repeat scans (divided by n_avg when y is a mean of n_avg independent acquisitions).
* ``delta^2``: model discrepancy (FE vs real specimen). Never estimated from repeats; it is an
  assumed scalar (or reported as a diagnostic from the residual).
* ``g``: instrument-to-simulation amplitude gain, unknown. ``"profiled"`` uses the weighted
  least-squares g for every candidate (flat prior, no Occam penalty); a float fixes it.

Residuals are formed directly (no ||y||^2 - 2 y.F + ||F||^2 expansion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LikelihoodResult:
    log_likelihood: np.ndarray   # (N,)
    gain: np.ndarray             # (N,)
    ssr: np.ndarray              # (N,) variance-weighted sum of squared residuals
    residual_rms: np.ndarray     # (N,) unweighted rms of y - g F
    n_samples: int


def total_variance(var_meas, *, n_avg: int = 1, discrepancy: float = 0.0):
    """v = var_meas / n_avg + discrepancy^2 (scalar or array)."""
    if n_avg < 1:
        raise ValueError("n_avg must be >= 1.")
    return np.asarray(var_meas, dtype=float) / n_avg + float(discrepancy) ** 2


def gaussian_loglik(y: np.ndarray, F: np.ndarray, var, gain="profiled") -> LikelihoodResult:
    """log p(y | a_i) for F of shape (N, *y.shape); ``var`` scalar or broadcastable to y."""
    y = np.asarray(y, dtype=float)
    v = np.broadcast_to(np.asarray(var, dtype=float), y.shape)
    if np.any(v <= 0) or not np.isfinite(v).all():
        raise ValueError("Variance must be positive and finite everywhere.")
    const = -0.5 * float(np.sum(np.log(2 * np.pi * v)))
    n = F.shape[0]
    ll, g, ssr, rms = (np.empty(n) for _ in range(4))
    for i in range(n):
        f = F[i]
        gi = float(np.sum(y * f / v) / np.sum(f * f / v)) if isinstance(gain, str) else float(gain)
        r = y - gi * f
        ssr[i] = float(np.sum(r * r / v))
        rms[i] = float(np.sqrt(np.mean(r * r)))
        g[i] = gi
        ll[i] = -0.5 * ssr[i] + const
    return LikelihoodResult(ll, g, ssr, rms, y.size)
