"""Bayesian inversion for crack size a on a 1-D grid of forward-library entries.

For each candidate a_i with prediction F(a_i) and measurement y (both flattened
to length M), with independent Gaussian noise Sigma = sigma^2 I:

    r_i              = y - F(a_i)
    log p(y | a_i)   = -||r_i||^2 / (2 sigma^2) - (M/2) log(2 pi sigma^2)
    log p(a_i | y)   = log p(y | a_i) + log p(a_i) - log Z
    log Z            = logsumexp_i [log p(y | a_i) + log p(a_i)]

Everything is computed in log space: for FMC data M ~ 10^5-10^6, so
log-likelihoods are of order -10^5 and exp() would underflow to zero.

The prior p(a_i) is a *discrete* probability mass on the grid. "Uniform" means
equal mass per grid point; on a non-uniform grid that is not a uniform density
in a. All crack sizes are in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..io.bristolfe import ForwardLibrary, flatten_response
from ..utils.validation import (
    check_increasing_grid,
    check_nonnegative,
    check_positive,
    check_probability,
    normalise_log_weights,
)


@dataclass(frozen=True)
class GridPosterior:
    """Discrete posterior p(a | y) on the crack-size grid, with summaries (all in m)."""

    crack_sizes: np.ndarray        # (N,) grid a_i
    log_likelihood: np.ndarray     # (N,) log p(y | a_i), including the Gaussian constant
    prior: np.ndarray              # (N,) p(a_i), sums to 1
    posterior: np.ndarray          # (N,) p(a_i | y), sums to 1
    log_evidence: float            # log p(y) = log sum_i p(y | a_i) p(a_i)
    map_index: int                 # argmax_i p(a_i | y)
    map: float                     # a_MAP
    mean: float                    # E[a | y]
    std: float                     # sqrt(Var[a | y])
    credible_interval: tuple[float, float]
    credible_level: float

    def summary(self, scale: float = 1e3, unit: str = "mm") -> str:
        lo, hi = self.credible_interval
        return (
            f"p(a | y): MAP = {self.map * scale:.4g} {unit}, mean = {self.mean * scale:.4g} {unit}, "
            f"std = {self.std * scale:.3g} {unit}, {100 * self.credible_level:.0f}% CI = "
            f"[{lo * scale:.4g}, {hi * scale:.4g}] {unit}"
        )


def gaussian_log_likelihood(
    y: ArrayLike, library: ForwardLibrary, sigma: float, *, chunk: int = 8
) -> np.ndarray:
    """log p(y | a_i) for every library entry under y = F(a_i) + e, e ~ N(0, sigma^2 I).

    ``y`` must have shape ``library.measurement_shape``. Residuals are formed
    directly (not via ||y||^2 - 2 y.F + ||F||^2, which cancels catastrophically
    when the fit is good), ``chunk`` candidates at a time to bound memory.
    """
    sigma = float(check_positive(sigma, "sigma"))
    y_flat = flatten_response(y, library)
    F = library.flat_responses()
    ssr = np.empty(library.n_candidates)
    for start in range(0, library.n_candidates, chunk):
        r = y_flat[None, :] - F[start : start + chunk]
        ssr[start : start + chunk] = np.einsum("ij,ij->i", r, r)
    m = y_flat.size
    return -0.5 * ssr / sigma**2 - 0.5 * m * np.log(2 * np.pi * sigma**2)


def estimate_sigma_profile(y: ArrayLike, library: ForwardLibrary) -> float:
    """Plug-in noise estimate sigma_hat = sqrt(min_i ||y - F(a_i)||^2 / M).

    This is the maximum-likelihood sigma at the best-fitting grid point. It
    absorbs any model discrepancy into "noise", so treat it as a diagnostic,
    not as a calibrated noise level.
    """
    y_flat = flatten_response(y, library)
    ssr = ((library.flat_responses() - y_flat[None, :]) ** 2).sum(axis=1)
    return float(np.sqrt(ssr.min() / y_flat.size))


def make_prior(crack_sizes: ArrayLike, prior: ArrayLike | None = None) -> np.ndarray:
    """Return a normalised discrete prior: uniform mass if ``prior`` is None."""
    a = check_increasing_grid(crack_sizes, "crack_sizes")
    if prior is None:
        return np.full(a.size, 1.0 / a.size)
    p = check_nonnegative(prior, "prior")
    if p.shape != a.shape:
        raise ValueError(f"prior has shape {p.shape}; expected {a.shape} (one mass per grid point).")
    if p.sum() <= 0:
        raise ValueError("prior has zero total mass.")
    return p / p.sum()


def discrete_quantile(crack_sizes: np.ndarray, pmf: np.ndarray, q: ArrayLike) -> np.ndarray:
    """Smallest grid value a_i with CDF(a_i) >= q (the standard discrete quantile)."""
    cdf = np.cumsum(pmf)
    cdf /= cdf[-1]
    idx = np.searchsorted(cdf, np.asarray(q, dtype=float) - 1e-12, side="left")
    return crack_sizes[np.clip(idx, 0, crack_sizes.size - 1)]


def grid_posterior(
    crack_sizes: ArrayLike,
    log_likelihood: ArrayLike,
    prior: ArrayLike | None = None,
    *,
    credible_level: float = 0.95,
) -> GridPosterior:
    """Combine a log-likelihood and a discrete prior into a :class:`GridPosterior`.

    The credible interval is equal-tailed: [Q((1-level)/2), Q((1+level)/2)]
    using :func:`discrete_quantile`, so its endpoints are grid points and its
    coverage is at least ``credible_level``.
    """
    a = check_increasing_grid(crack_sizes, "crack_sizes")
    loglik = np.asarray(log_likelihood, dtype=float)
    if loglik.shape != a.shape:
        raise ValueError(f"log_likelihood shape {loglik.shape} does not match crack_sizes {a.shape}.")
    level = check_probability(credible_level, "credible_level")
    p0 = make_prior(a, prior)
    with np.errstate(divide="ignore"):
        log_joint = loglik + np.log(p0)            # zero prior mass -> -inf, allowed
    post, log_z = normalise_log_weights(log_joint)
    mean = float(np.sum(a * post))
    std = float(np.sqrt(max(np.sum((a - mean) ** 2 * post), 0.0)))
    lo, hi = discrete_quantile(a, post, [(1 - level) / 2, (1 + level) / 2])
    i_map = int(np.argmax(post))
    return GridPosterior(
        crack_sizes=a, log_likelihood=loglik, prior=p0, posterior=post, log_evidence=log_z,
        map_index=i_map, map=float(a[i_map]), mean=mean, std=std,
        credible_interval=(float(lo), float(hi)), credible_level=level,
    )


def invert_grid(
    y: ArrayLike,
    library: ForwardLibrary,
    sigma: float,
    *,
    prior: ArrayLike | None = None,
    credible_level: float = 0.95,
) -> GridPosterior:
    """Full V0 inversion: measurement y -> p(a | y) on the library's crack-size grid."""
    loglik = gaussian_log_likelihood(y, library, sigma)
    return grid_posterior(library.crack_sizes, loglik, prior, credible_level=credible_level)


def sample_posterior(
    result: GridPosterior,
    n_samples: int,
    *,
    rng: np.random.Generator | None = None,
    jitter: bool = False,
) -> np.ndarray:
    """Draw a^(i) ~ p(a | y) from the discrete grid posterior.

    ``jitter=False`` returns grid values exactly (the posterior *is* discrete).
    ``jitter=True`` interprets each mass p_i as spread uniformly over its grid
    cell [midpoint to the left, midpoint to the right] (end cells extend half a
    spacing outward), giving continuous samples without inventing any shape
    beyond the grid resolution. Use it when downstream maps (Paris RUL) would
    otherwise produce visibly spiky histograms.
    """
    rng = np.random.default_rng() if rng is None else rng
    if int(n_samples) < 1:
        raise ValueError("n_samples must be >= 1.")
    a = result.crack_sizes
    idx = rng.choice(a.size, size=int(n_samples), p=result.posterior)
    if not jitter:
        return a[idx]
    mid = 0.5 * (a[1:] + a[:-1])
    lower = np.concatenate([[a[0] - (mid[0] - a[0])], mid])
    upper = np.concatenate([mid, [a[-1] + (a[-1] - mid[-1])]])
    lower = np.maximum(lower, 0.5 * a[0])  # keep a > 0
    return rng.uniform(lower[idx], upper[idx])
