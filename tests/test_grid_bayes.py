"""Bayesian grid inversion: y -> p(a | y)."""

import numpy as np
import pytest

from rul_pipeline.io import ForwardLibrary
from rul_pipeline.inversion import (
    estimate_sigma_profile,
    gaussian_log_likelihood,
    grid_posterior,
    invert_grid,
    sample_posterior,
)
from rul_pipeline.synthetic import add_measurement_noise, toy_fmc_library, toy_fmc_truth

GRID = np.round(np.arange(0.5, 5.0001, 0.1), 6) * 1e-3   # m


@pytest.fixture(scope="module")
def lib():
    return toy_fmc_library(GRID, n_elements=8)


def _invert(lib, a_true, sigma, seed=0, **kw):
    truth = toy_fmc_truth(a_true, lib)
    y, s = add_measurement_noise(truth, sigma=sigma, rng=np.random.default_rng(seed))
    return invert_grid(y, lib, s, **kw)


def test_posterior_peaks_at_true_crack_size_for_low_noise(lib):
    a_true = GRID[15]                               # 2.0 mm, on the grid
    post = _invert(lib, a_true, sigma=1e-3)
    assert post.map == pytest.approx(a_true)
    assert post.posterior[post.map_index] > 0.99
    assert post.mean == pytest.approx(a_true, abs=1e-6)


def test_posterior_sums_to_one_and_is_nonnegative(lib):
    for sigma in (1e-3, 0.1, 1.0):
        post = _invert(lib, 2.03e-3, sigma)
        assert post.posterior.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(post.posterior >= 0)
        assert post.prior.sum() == pytest.approx(1.0)


def test_posterior_broadens_with_increasing_noise(lib):
    # Peak |F| ~ 1, so sigma ~ 1-3 is the regime where the posterior spans several grid cells.
    sigmas = [0.8, 1.2, 1.8, 2.7]
    stds = [np.mean([_invert(lib, 2.03e-3, s, seed=k).std for k in range(10)]) for s in sigmas]
    assert np.all(np.diff(stds) > 0), stds
    assert stds[-1] > 0.3e-3                          # visibly broad (> 3 grid cells)
    widths = [np.mean([np.diff(_invert(lib, 2.03e-3, s, seed=k).credible_interval)[0] for k in range(10)])
              for s in (0.4, 2.7)]
    assert widths[0] == 0.0 and widths[1] > 0.5e-3


def test_map_is_sensible_for_off_grid_truth_and_moderate_noise(lib):
    a_true = 2.03e-3                                 # between grid points 2.0 and 2.1 mm
    post = _invert(lib, a_true, sigma=0.05)
    assert abs(post.map - a_true) <= 0.1e-3          # within one grid step
    lo, hi = post.credible_interval
    assert lo <= post.map <= hi and lo <= post.mean <= hi


def test_matches_analytic_linear_gaussian_posterior():
    """F(a) = a g: with a flat prior the posterior is N(g.y / g.g, sigma^2 / g.g)."""
    rng = np.random.default_rng(4)
    a = np.linspace(0.5e-3, 5e-3, 2001)
    g = rng.normal(size=50) * 1e3
    sigma, a_true = 0.5, 2.5e-3
    lib_lin = ForwardLibrary(crack_sizes=a, responses=a[:, None] * g[None, :], dims=("crack", "feature"))
    y = a_true * g + sigma * rng.normal(size=g.size)
    post = invert_grid(y, lib_lin, sigma)
    assert post.mean == pytest.approx(g @ y / (g @ g), rel=1e-4)
    assert post.std == pytest.approx(sigma / np.linalg.norm(g), rel=1e-2)


def test_log_space_is_stable_for_huge_log_likelihoods(lib):
    post = _invert(lib, 2.0e-3, sigma=1e-6)          # log-likelihood differences ~ 1e12
    assert np.all(np.isfinite(post.posterior)) and post.posterior.max() == pytest.approx(1.0)
    assert np.all(np.isfinite(post.log_likelihood))  # log-*densities*: may be > 0 for tiny sigma


def test_user_prior_is_respected(lib):
    prior = np.ones(GRID.size)
    prior[GRID > 1.5e-3] = 0.0                       # rule out cracks larger than 1.5 mm
    post = _invert(lib, 2.0e-3, sigma=0.05, prior=prior)
    assert post.posterior[GRID > 1.5e-3].sum() == 0.0
    assert post.map <= 1.5e-3
    with pytest.raises(ValueError, match="shape"):
        grid_posterior(GRID, np.zeros(GRID.size), prior=np.ones(3))
    with pytest.raises(ValueError, match="non-negative"):
        grid_posterior(GRID, np.zeros(GRID.size), prior=-np.ones(GRID.size))


def test_gaussian_log_likelihood_includes_normalising_constant():
    lib_small = ForwardLibrary(crack_sizes=np.array([1e-3, 2e-3]), responses=np.array([[0.0, 0.0], [1.0, 1.0]]),
                               dims=("crack", "feature"))
    ll = gaussian_log_likelihood(np.array([0.0, 0.0]), lib_small, sigma=1.0)
    np.testing.assert_allclose(ll, [-np.log(2 * np.pi), -1.0 - np.log(2 * np.pi)])


def test_sampling_reproduces_posterior_frequencies(lib):
    post = _invert(lib, 2.03e-3, sigma=0.3, seed=1)
    samples = sample_posterior(post, 200_000, rng=np.random.default_rng(0))
    assert set(np.unique(samples)) <= set(GRID)
    freq = np.array([np.mean(samples == a) for a in GRID])
    np.testing.assert_allclose(freq, post.posterior, atol=4e-3)
    assert samples.mean() == pytest.approx(post.mean, rel=1e-2)


def test_jittered_samples_stay_within_grid_cells(lib):
    post = _invert(lib, 2.03e-3, sigma=1e-3)        # essentially all mass at one grid point
    s = sample_posterior(post, 10_000, rng=np.random.default_rng(0), jitter=True)
    assert np.all(np.abs(s - post.map) <= 0.05e-3 + 1e-12)
    assert np.unique(s).size > 9000


def test_profile_sigma_recovers_noise_level(lib):
    truth = lib.responses[15]
    y, s = add_measurement_noise(truth, sigma=0.2, rng=np.random.default_rng(2))
    assert estimate_sigma_profile(y, lib) == pytest.approx(0.2, rel=0.02)
