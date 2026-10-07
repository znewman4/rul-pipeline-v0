"""Small input-validation and probability helpers shared by every stage.

These functions raise early, with messages that name the offending argument,
so that a mistake (e.g. a crack size in mm instead of m, or a NaN in the
forward library) fails loudly at the stage where it was introduced instead
of producing a silently wrong RUL three stages later.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp


def check_finite(x: ArrayLike, name: str) -> np.ndarray:
    """Return ``x`` as a float array, raising if any entry is NaN or inf."""
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
        raise ValueError(f"{name} contains {n_bad} non-finite value(s) (NaN or inf).")
    return arr


def check_positive(x: ArrayLike, name: str) -> np.ndarray:
    """Return ``x`` as a finite float array, raising unless every entry is > 0."""
    arr = check_finite(x, name)
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive; got min = {arr.min():.6g}.")
    return arr


def check_nonnegative(x: ArrayLike, name: str) -> np.ndarray:
    """Return ``x`` as a finite float array, raising if any entry is < 0."""
    arr = check_finite(x, name)
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative; got min = {arr.min():.6g}.")
    return arr


def check_increasing_grid(x: ArrayLike, name: str, *, min_points: int = 2) -> np.ndarray:
    """Return ``x`` as a 1-D finite float array that is strictly increasing."""
    arr = check_finite(x, name)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}.")
    if arr.size < min_points:
        raise ValueError(f"{name} must contain at least {min_points} point(s); got {arr.size}.")
    if np.any(np.diff(arr) <= 0):
        raise ValueError(f"{name} must be strictly increasing (sorted, no duplicates).")
    return arr


def check_probability(p: float, name: str, *, open_interval: bool = True) -> float:
    """Validate a scalar probability in (0, 1) (or [0, 1] if not ``open_interval``)."""
    p = float(p)
    ok = 0.0 < p < 1.0 if open_interval else 0.0 <= p <= 1.0
    if not (np.isfinite(p) and ok):
        bounds = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must lie in {bounds}; got {p!r}.")
    return p


def normalise_log_weights(log_w: ArrayLike) -> tuple[np.ndarray, float]:
    """Exponentiate and normalise unnormalised log-weights stably.

    Returns ``(w, log_Z)`` with ``w = exp(log_w - log_Z)`` and
    ``log_Z = log(sum(exp(log_w)))`` computed via log-sum-exp, so that
    log-likelihoods of order -1e6 (common for FMC data) do not underflow.
    Entries equal to ``-inf`` (zero prior mass) are allowed.
    """
    log_w = np.asarray(log_w, dtype=float)
    if np.any(np.isnan(log_w)) or np.any(log_w == np.inf):
        raise ValueError("log-weights contain NaN or +inf.")
    log_z = float(logsumexp(log_w))
    if not np.isfinite(log_z):
        raise ValueError("All log-weights are -inf: the distribution has zero total mass.")
    return np.exp(log_w - log_z), log_z
