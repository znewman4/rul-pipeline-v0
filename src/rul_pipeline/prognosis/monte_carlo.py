"""Monte Carlo propagation of p(a | y) to p(RUL | y).

The posterior is never collapsed to a point estimate. Each posterior sample is
pushed through the deterministic structural and growth maps:

    a^(i)        ~ p(a | y)                                (inversion.sample_posterior)
    DK^(i)       = G(a^(i)) = Y Delta_sigma sqrt(pi a^(i))  (structural.mode_i)
    RUL^(i)      = H(a^(i); C, m, Delta_sigma, Y, a_c)      (crack_growth.paris)

so {RUL^(i)} are samples from p(RUL | y) *conditional on* the assumed C, m,
loading and a_c. Because RUL(a) is convex in a, E[RUL] >= RUL(E[a]) (Jensen):
plugging the MAP or mean crack size into Paris would bias the answer.

Summaries are plain sample statistics -- no kernel density estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..crack_growth.paris import ParisModel
from ..structural.mode_i import stress_intensity_mode_i
from ..utils.validation import check_finite, check_positive, check_probability


@dataclass(frozen=True)
class RULResult:
    """Samples from p(RUL | y) with their inputs and summaries. RUL in cycles."""

    a_samples: np.ndarray        # a^(i) [m]
    delta_K_samples: np.ndarray  # Delta K^(i) [Pa sqrt(m)] at the current crack size
    rul_samples: np.ndarray      # RUL^(i) [cycles]; 0 if a^(i) >= a_c
    a_c: float                   # critical crack size used [m]
    mean: float
    median: float
    std: float
    credible_interval: tuple[float, float]
    credible_level: float
    fraction_already_failed: float   # fraction of samples with a^(i) >= a_c

    def summary(self) -> str:
        lo, hi = self.credible_interval
        return (
            f"p(RUL | y): median = {self.median:.4g}, mean = {self.mean:.4g}, std = {self.std:.3g} cycles, "
            f"{100 * self.credible_level:.0f}% CI = [{lo:.4g}, {hi:.4g}] cycles, "
            f"P(a >= a_c now) = {self.fraction_already_failed:.3g}"
        )


def propagate_rul(
    a_samples: ArrayLike,
    model: ParisModel,
    a_c: float,
    *,
    credible_level: float = 0.90,
) -> RULResult:
    """Push posterior crack-size samples through Mode-I + Paris to RUL samples."""
    a = check_positive(a_samples, "a_samples (m)").reshape(-1)
    a_c = float(check_positive(a_c, "a_c (m)"))
    level = check_probability(credible_level, "credible_level")
    dK = stress_intensity_mode_i(a, model.delta_sigma, model.Y)
    rul = model.rul(a, a_c, on_exceed="zero")
    lo, hi = np.quantile(rul, [(1 - level) / 2, (1 + level) / 2])
    return RULResult(
        a_samples=a, delta_K_samples=np.atleast_1d(dK), rul_samples=np.atleast_1d(rul), a_c=a_c,
        mean=float(np.mean(rul)), median=float(np.median(rul)), std=float(np.std(rul, ddof=1)) if a.size > 1 else 0.0,
        credible_interval=(float(lo), float(hi)), credible_level=level,
        fraction_already_failed=float(np.mean(a >= a_c)),
    )


def rul_quantiles(rul_samples: ArrayLike, q: ArrayLike) -> np.ndarray:
    """Empirical quantiles of RUL samples (e.g. q = [0.05, 0.5, 0.95])."""
    return np.quantile(check_finite(rul_samples, "rul_samples"), q)


def probability_of_failure(rul_samples: ArrayLike, horizon: ArrayLike) -> np.ndarray:
    """P(RUL <= horizon | y), the empirical CDF of RUL at each horizon [cycles]."""
    r = np.sort(check_finite(rul_samples, "rul_samples").reshape(-1))
    h = np.asarray(horizon, dtype=float)
    p = np.searchsorted(r, h, side="right") / r.size
    return p if p.ndim else p[()]


def mc_standard_error(p: ArrayLike, n_samples: int) -> np.ndarray:
    """Binomial Monte Carlo standard error sqrt(p (1 - p) / n) of an estimated probability."""
    p = np.asarray(p, dtype=float)
    return np.sqrt(p * (1 - p) / int(n_samples))
