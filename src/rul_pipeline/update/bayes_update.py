"""Scaffold for sequential Bayesian updating on the crack-size grid.

One inspection cycle k -> k+1 is a predict / update pair:

    predict:  p(a_{k+1} | y_1:k)   = sum_i p(a_{k+1} | a_k = a_i) p(a_k = a_i | y_1:k)
    update:   p(a_{k+1} | y_1:k+1) ∝ p(y_{k+1} | a_{k+1}) p(a_{k+1} | y_1:k)

V0 transition: deterministic growth a_i -> g(a_i) over Delta N cycles (Paris
closed form), optionally blurred by a Gaussian "growth model error" kernel.
The mass at g(a_i), which generally falls between grid nodes, is split
linearly between the two bracketing nodes (conserves mass and mean). Mass
pushed above the largest grid size is reported separately (``p_beyond``):
the library cannot represent it, and for prognosis it means "at least a_N".

This is a grid (point-mass) filter, deliberately generic -- any callable
``propagate`` can replace Paris. It is not a DBN or particle filter yet.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from ..crack_growth.paris import ParisModel
from ..inversion.grid_bayes import GridPosterior, grid_posterior, make_prior
from ..utils.validation import check_increasing_grid, check_nonnegative


def predict_grid(
    crack_sizes: ArrayLike,
    pmf: ArrayLike,
    propagate: Callable[[np.ndarray], np.ndarray],
    *,
    growth_noise_std: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Push a grid pmf through a deterministic growth map ``propagate(a) -> a_new``.

    Returns ``(pmf_pred, p_beyond)`` where ``pmf_pred`` lives on the same grid
    and ``pmf_pred.sum() + p_beyond == 1``. ``growth_noise_std`` [m] adds an
    optional Gaussian transition kernel (column-normalised on the grid).
    """
    a = check_increasing_grid(crack_sizes, "crack_sizes")
    p = make_prior(a, pmf)
    a_new = np.asarray(propagate(a), dtype=float)
    if a_new.shape != a.shape or np.any(np.isnan(a_new)):
        raise ValueError("propagate(a) must return an array of the same shape without NaN.")

    beyond = a_new > a[-1]
    a_in = np.clip(a_new, a[0], a[-1])
    j = np.clip(np.searchsorted(a, a_in, side="right") - 1, 0, a.size - 2)   # left node
    w_right = (a_in - a[j]) / (a[j + 1] - a[j])
    pred = np.zeros_like(p)
    np.add.at(pred, j, np.where(beyond, 0.0, p * (1 - w_right)))
    np.add.at(pred, j + 1, np.where(beyond, 0.0, p * w_right))
    p_beyond = float(p[beyond].sum())

    s = float(check_nonnegative(growth_noise_std, "growth_noise_std"))
    if s > 0:
        kernel = np.exp(-0.5 * ((a[:, None] - a[None, :]) / s) ** 2)      # kernel[to, from]
        pred = kernel @ (pred / kernel.sum(axis=0))
    return pred, p_beyond


def paris_propagator(model: ParisModel, delta_N: float) -> Callable[[np.ndarray], np.ndarray]:
    """``a -> a(delta_N)`` under ``model`` (closed form; runaway -> inf -> counted as beyond)."""
    return lambda a: model.crack_size_after(a, delta_N)


def update_grid(
    crack_sizes: ArrayLike,
    predicted_prior: ArrayLike,
    log_likelihood_next: ArrayLike,
    *,
    credible_level: float = 0.95,
) -> GridPosterior:
    """Bayes' rule on the grid with the predicted prior: returns p(a_{k+1} | y_1:k+1)."""
    return grid_posterior(crack_sizes, log_likelihood_next, predicted_prior, credible_level=credible_level)


def sequential_update(
    posterior_k: GridPosterior,
    model: ParisModel,
    delta_N: float,
    log_likelihood_next: ArrayLike,
    *,
    growth_noise_std: float = 0.0,
) -> tuple[GridPosterior, float]:
    """One predict + update step. Returns (posterior_{k+1}, p_beyond from the prediction)."""
    pred, p_beyond = predict_grid(posterior_k.crack_sizes, posterior_k.posterior,
                                  paris_propagator(model, delta_N), growth_noise_std=growth_noise_std)
    if pred.sum() <= 0:
        raise ValueError("All predicted mass left the grid: extend the forward library to larger cracks.")
    return update_grid(posterior_k.crack_sizes, pred, log_likelihood_next,
                       credible_level=posterior_k.credible_level), p_beyond
