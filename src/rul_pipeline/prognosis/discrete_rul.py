"""Exact weighted discrete RUL for a discrete crack-size posterior (no sampling, no jitter).

The a = 0 (notch-only) candidate is excluded and its mass reported separately; the remaining
weights are renormalised, so every statistic here is CONDITIONAL on a crack being present and on
the assumed growth law and loading (``ParisModel``) and critical size ``a_c``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..crack_growth import ParisModel


@dataclass(frozen=True)
class DiscreteRUL:
    p_notch_only: float
    a: np.ndarray            # positive candidate sizes used for growth [m] (incl. any offset)
    weights: np.ndarray      # conditional on a crack being present; sums to 1 (empty if no crack mass)
    rul: np.ndarray          # cycles
    mean: float
    quantiles: dict
    p_fail_by: dict
    a_c: float

    def summary(self) -> str:
        if self.a.size == 0:
            return "no crack candidates"
        q = ", ".join(f"P{int(k * 100)}={v:.4g}" for k, v in self.quantiles.items())
        return f"P(notch only)={self.p_notch_only:.4g}; RUL | crack: mean {self.mean:.4g}, {q} cycles"


def weighted_discrete_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Smallest value v with weighted CDF(v) >= q."""
    o = np.argsort(values)
    cdf = np.cumsum(weights[o])
    cdf /= cdf[-1]
    return float(values[o][np.searchsorted(cdf, q - 1e-12, side="left")])


def conditional_discrete_rul(crack_sizes, posterior, model: ParisModel, a_c: float, *,
                             size_offset: float = 0.0, quantiles=(0.05, 0.5, 0.95), horizons=()) -> DiscreteRUL:
    a_all = np.asarray(crack_sizes, dtype=float)
    p = np.asarray(posterior, dtype=float)
    pos = a_all > 0
    p_notch = float(p[~pos].sum())
    if p[pos].sum() <= 0:
        return DiscreteRUL(p_notch, np.empty(0), np.empty(0), np.empty(0), float("nan"), {}, {}, a_c)
    a = a_all[pos] + size_offset
    w = p[pos] / p[pos].sum()
    rul = np.atleast_1d(model.rul(a, a_c, on_exceed="zero"))
    return DiscreteRUL(
        p_notch, a, w, rul, float(np.sum(w * rul)),
        {q: weighted_discrete_quantile(rul, w, q) for q in quantiles},
        {float(h): float(w[rul <= h].sum()) for h in horizons}, float(a_c),
    )
