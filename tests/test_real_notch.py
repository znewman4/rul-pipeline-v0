import numpy as np
import pytest

from rul_pipeline.crack_growth import ParisModel
from rul_pipeline.inversion.error_models import gaussian_loglik, total_variance
from rul_pipeline.inversion.observation import Preprocess, apply, estimate_time_shift, fractional_shift
from rul_pipeline.io.repeat_scans import RepeatGroup, regularised_variance, summarise, validate_group
from rul_pipeline.io.real_fmc import RealFMC
from rul_pipeline.prognosis.discrete_rul import conditional_discrete_rul, weighted_discrete_quantile

T = np.arange(1000) * 40e-9


def _burst(t0, f=5e6):
    t = T - t0
    return np.exp(-((t / 0.4e-6) ** 2)) * np.sin(2 * np.pi * f * t)


def test_fractional_shift_moves_signal_and_preserves_energy():
    x = _burst(10e-6)
    y = fractional_shift(x, 0.5e-6, 40e-9)          # y(t) = x(t + 0.5us): burst appears 0.5 us earlier
    assert np.argmax(np.abs(y)) * 40e-9 == pytest.approx(9.5e-6, abs=60e-9)
    assert np.sum(y**2) == pytest.approx(np.sum(x**2), rel=1e-3)


def test_time_shift_estimator_recovers_known_shift():
    rng = np.random.default_rng(0)
    cand = np.stack([_burst(t0) for t0 in rng.uniform(9e-6, 22e-6, 30)])
    real = fractional_shift(cand, -0.4e-6, 40e-9) * 7.0       # real arrives 0.4 us later; arbitrary gain
    pre = Preprocess((5.0, 30.0), (2.5, 7.5), 0.0)
    best, _, _ = estimate_time_shift(real, cand, T, pre)
    assert best == pytest.approx(0.4, abs=0.05)


def test_likelihood_gain_profiled_ignores_units_and_finds_truth():
    rng = np.random.default_rng(1)
    F = rng.normal(size=(4, 50, 40)) * 1e-12
    F = F / 1e-12 * 1e-12
    y = 3.7e12 * F[2] + rng.normal(scale=0.01, size=F[2].shape)
    r = gaussian_loglik(y, F, 0.01**2)
    assert np.argmax(r.log_likelihood) == 2
    assert r.gain[2] == pytest.approx(3.7e12, rel=0.02)


def test_total_variance_averaging_and_discrepancy():
    assert total_variance(4.0, n_avg=4, discrepancy=0.5) == pytest.approx(1.25)
    with pytest.raises(ValueError):
        total_variance(1.0, n_avg=0)


def test_regularised_variance_nonsingular_and_needs_repeats():
    raw = np.zeros((5, 7))
    raw[0, 0] = 1.0
    v = regularised_variance(raw, 4)
    assert np.all(v > 0)
    with pytest.raises(ValueError):
        regularised_variance(raw, 1)


def _scan(seed, shift_t=0.0):
    rng = np.random.default_rng(seed)
    pairs = np.array([(0, 0), (0, 1), (1, 1)])
    return RealFMC(data=rng.normal(size=(3, 20)), pairs=pairs, time=np.arange(20) * 40e-9 + shift_t,
                   element_x=np.array([0.0, 6.3e-4]), centre_freq=5e6, source=f"s{seed}.mat")


def test_repeat_group_validation_rejects_inconsistent_axes():
    validate_group(RepeatGroup("fixed_probe", (_scan(0), _scan(1))))
    with pytest.raises(ValueError, match="time axis"):
        validate_group(RepeatGroup("fixed_probe", (_scan(0), _scan(1, shift_t=1e-9))))


def test_summarise_variance_estimates_noise_level():
    rng = np.random.default_rng(2)
    proc = rng.normal(scale=0.3, size=(8, 30, 40))
    S = summarise(proc)
    assert S["sd_raw"] == pytest.approx(0.3, rel=0.05)
    assert S["n"] == 8


def test_discrete_rul_excludes_notch_and_is_exact():
    model = ParisModel()
    a = np.array([0.0, 1e-3, 2e-3, 3e-3])
    p = np.array([0.5, 0.1, 0.3, 0.1])
    r = conditional_discrete_rul(a, p, model, 10e-3)
    assert r.p_notch_only == pytest.approx(0.5)
    np.testing.assert_allclose(r.weights, [0.2, 0.6, 0.2])
    np.testing.assert_allclose(r.rul, model.rul(a[1:], 10e-3))
    assert r.mean == pytest.approx(float(np.sum(r.weights * r.rul)))
    assert set(np.round(r.rul, 6)) >= {round(float(v), 6) for v in r.quantiles.values()}   # quantiles are atoms
    assert weighted_discrete_quantile(np.array([1.0, 2.0]), np.array([0.5, 0.5]), 0.5) == 1.0


def test_all_notch_posterior_gives_no_rul():
    r = conditional_discrete_rul(np.array([0.0, 1e-3]), np.array([1.0, 0.0]), ParisModel(), 10e-3)
    assert r.a.size == 0 and r.p_notch_only == 1.0


def test_apply_windows_and_does_not_normalise_traces():
    x = np.stack([_burst(12e-6), 5.0 * _burst(12e-6)])
    y = apply(x, T, Preprocess((8.0, 25.0), None, 0.0), is_real=False)
    assert y.shape[-1] < x.shape[-1]
    assert np.abs(y[1]).max() == pytest.approx(5.0 * np.abs(y[0]).max())
