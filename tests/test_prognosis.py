"""Monte Carlo propagation p(a | y) -> p(RUL | y)."""

import numpy as np
import pytest

from rul_pipeline.crack_growth import ParisModel
from rul_pipeline.inversion import grid_posterior, sample_posterior
from rul_pipeline.prognosis import mc_standard_error, probability_of_failure, propagate_rul, rul_quantiles

MODEL = ParisModel(delta_sigma=100e6)       # BS 7910 steel-in-air C, m = 3, Y = 1.12
A_C = 10e-3
GRID = np.linspace(0.5e-3, 5e-3, 46)


def _gaussian_posterior(mean, std):
    return grid_posterior(GRID, -0.5 * ((GRID - mean) / std) ** 2)


def _samples(mean=2e-3, std=0.3e-3, n=50_000, seed=0):
    return sample_posterior(_gaussian_posterior(mean, std), n, rng=np.random.default_rng(seed), jitter=True)


def test_rul_samples_behave_sensibly():
    a = _samples()
    res = propagate_rul(a, MODEL, A_C)
    assert res.rul_samples.shape == a.shape and np.all(res.rul_samples > 0)
    assert res.credible_interval[0] < res.median < res.credible_interval[1]
    # Every RUL lies between the deterministic RULs of the extreme crack sizes.
    assert MODEL.rul(a.max(), A_C) <= res.rul_samples.min() and res.rul_samples.max() <= MODEL.rul(a.min(), A_C)
    # Each sample is the deterministic map of its own crack size (no collapse to a point estimate).
    np.testing.assert_allclose(res.rul_samples, MODEL.rul(a, A_C))
    np.testing.assert_allclose(res.delta_K_samples, 1.12 * 100e6 * np.sqrt(np.pi * a))


def test_monte_carlo_mean_matches_exact_grid_expectation():
    post = _gaussian_posterior(2e-3, 0.3e-3)
    a = sample_posterior(post, 200_000, rng=np.random.default_rng(1))
    res = propagate_rul(a, MODEL, A_C)
    exact = np.sum(post.posterior * MODEL.rul(GRID, A_C))
    assert res.mean == pytest.approx(exact, abs=4 * res.std / np.sqrt(a.size))


def test_point_estimate_plug_in_is_biased_jensen():
    res = propagate_rul(_samples(std=0.5e-3), MODEL, A_C)
    assert res.mean > MODEL.rul(res.a_samples.mean(), A_C)       # RUL(a) convex => E[RUL] > RUL(E[a])


def test_rul_spread_grows_with_posterior_spread_and_shrinks_with_crack_size():
    narrow, wide = propagate_rul(_samples(std=0.1e-3), MODEL, A_C), propagate_rul(_samples(std=0.5e-3), MODEL, A_C)
    assert wide.std > narrow.std
    small, large = propagate_rul(_samples(mean=1.5e-3), MODEL, A_C), propagate_rul(_samples(mean=3.5e-3), MODEL, A_C)
    assert large.median < small.median


def test_delta_posterior_gives_deterministic_rul():
    res = propagate_rul(np.full(100, 2e-3), MODEL, A_C)
    assert res.std == pytest.approx(0.0, abs=1e-9) and res.median == pytest.approx(MODEL.rul(2e-3, A_C))


def test_failure_probability_increases_with_horizon():
    res = propagate_rul(_samples(), MODEL, A_C)
    horizons = np.linspace(0, 2 * res.rul_samples.max(), 200)
    pf = probability_of_failure(res.rul_samples, horizons)
    assert np.all(np.diff(pf) >= 0)
    assert pf[0] == 0.0 and pf[-1] == 1.0
    assert probability_of_failure(res.rul_samples, res.median) == pytest.approx(0.5, abs=1e-3)
    assert np.all(mc_standard_error(pf, res.rul_samples.size) <= 0.5 / np.sqrt(res.rul_samples.size) + 1e-15)


def test_samples_beyond_critical_size_count_as_failed():
    a = np.array([2e-3, 9e-3, 11e-3, 12e-3])
    res = propagate_rul(a, MODEL, A_C)
    assert res.fraction_already_failed == 0.5
    assert probability_of_failure(res.rul_samples, 0.0) == 0.5


def test_rul_quantiles():
    np.testing.assert_allclose(rul_quantiles(np.arange(101.0), [0.05, 0.5, 0.95]), [5, 50, 95])
