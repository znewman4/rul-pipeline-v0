"""Sequential grid update scaffold: predict (growth) + update (new measurement)."""

import numpy as np
import pytest

from rul_pipeline.crack_growth import ParisModel
from rul_pipeline.inversion import grid_posterior
from rul_pipeline.update import paris_propagator, predict_grid, sequential_update

GRID = np.linspace(0.5e-3, 5e-3, 91)
MODEL = ParisModel(delta_sigma=100e6)


def _post(mean, std):
    return grid_posterior(GRID, -0.5 * ((GRID - mean) / std) ** 2)


def test_zero_growth_prediction_is_identity():
    p = _post(2e-3, 0.3e-3).posterior
    pred, beyond = predict_grid(GRID, p, lambda a: a)
    np.testing.assert_allclose(pred, p, atol=1e-15)
    assert beyond == 0.0


def test_prediction_conserves_mass_and_shifts_mean():
    post = _post(2e-3, 0.2e-3)
    shift = 0.37e-3
    pred, beyond = predict_grid(GRID, post.posterior, lambda a: a + shift)
    assert pred.sum() + beyond == pytest.approx(1.0)
    assert np.sum(GRID * pred) == pytest.approx(post.mean + shift, rel=1e-9)   # linear split keeps the mean


def test_mass_pushed_off_grid_is_reported():
    pred, beyond = predict_grid(GRID, _post(4.8e-3, 0.2e-3).posterior, lambda a: a + 0.5e-3)
    assert beyond > 0.5 and pred.sum() + beyond == pytest.approx(1.0)


def test_paris_prediction_moves_mass_to_larger_cracks():
    post = _post(2e-3, 0.2e-3)
    pred, _ = predict_grid(GRID, post.posterior, paris_propagator(MODEL, 5e4))
    assert np.sum(GRID * pred) > post.mean


def test_second_measurement_sharpens_posterior():
    a_true = 2.0e-3
    loglik = -0.5 * ((GRID - a_true) / 0.3e-3) ** 2                 # same-quality measurement twice
    post1 = grid_posterior(GRID, loglik)
    post2, _ = sequential_update(post1, MODEL, delta_N=0.0, log_likelihood_next=loglik)
    assert post2.std == pytest.approx(post1.std / np.sqrt(2), rel=0.02)
    assert post2.posterior.sum() == pytest.approx(1.0)


def test_update_tracks_growth_between_inspections():
    dN = 8e4
    a1 = 2.0e-3
    a2 = MODEL.crack_size_after(a1, dN)
    post1 = _post(a1, 0.15e-3)
    post2, _ = sequential_update(post1, MODEL, dN, -0.5 * ((GRID - a2) / 0.15e-3) ** 2)
    assert post2.mean == pytest.approx(a2, abs=0.05e-3)
    assert post2.std < post1.std
