"""Risk-based next-inspection interval."""

import numpy as np
import pytest

from rul_pipeline.crack_growth import ParisModel
from rul_pipeline.inspection import exceedance_risk, failure_time_risk, select_inspection_interval

MODEL = ParisModel(delta_sigma=100e6)
A_LIM = 10e-3


def _samples(mean=2e-3, std=0.3e-3, n=10_000, seed=0):
    a = np.random.default_rng(seed).normal(mean, std, n)
    return a[a > 0.1e-3]


def test_interval_decreases_as_risk_tolerance_becomes_stricter():
    a = _samples()
    intervals = [select_inspection_interval(a, MODEL, A_LIM, alpha).interval for alpha in (0.2, 0.1, 1e-2, 1e-3)]
    assert np.all(np.diff(intervals) < 0), intervals


def test_interval_decreases_for_larger_initial_cracks():
    grid = np.linspace(0, 5e5, 501)
    intervals = [select_inspection_interval(_samples(mean=m), MODEL, A_LIM, 1e-3, intervals=grid).interval
                 for m in (1.5e-3, 2.5e-3, 3.5e-3, 5e-3)]
    assert np.all(np.diff(intervals) < 0), intervals


def test_interval_decreases_for_heavier_assumed_loading():
    a, grid = _samples(), np.linspace(0, 5e5, 501)
    light = select_inspection_interval(a, ParisModel(delta_sigma=80e6), A_LIM, 1e-2, intervals=grid)
    heavy = select_inspection_interval(a, ParisModel(delta_sigma=120e6), A_LIM, 1e-2, intervals=grid)
    assert heavy.interval < light.interval and heavy.delta_sigma == 120e6


def test_selected_interval_is_largest_safe_candidate():
    a, grid = _samples(), np.linspace(0, 5e5, 1001)
    s = select_inspection_interval(a, MODEL, A_LIM, 0.05, intervals=grid)
    k = np.searchsorted(grid, s.interval)
    assert s.risk[k] <= 0.05 < s.risk[k + 1]
    assert np.all(np.diff(s.risk) >= 0) and not s.capped


def test_exceedance_and_failure_time_criteria_agree():
    a, grid = _samples(), np.linspace(0, 4e5, 401)
    np.testing.assert_allclose(exceedance_risk(a, MODEL, A_LIM, grid), failure_time_risk(a, MODEL, A_LIM, grid),
                               atol=1.0 / a.size)
    s1 = select_inspection_interval(a, MODEL, A_LIM, 0.01, intervals=grid, criterion="exceedance")
    s2 = select_inspection_interval(a, MODEL, A_LIM, 0.01, intervals=grid, criterion="failure_time")
    assert s1.interval == s2.interval


def test_immediate_action_when_current_risk_already_exceeds_alpha():
    a = np.r_[np.full(90, 2e-3), np.full(10, 11e-3)]          # 10 % of mass already beyond a_lim
    s = select_inspection_interval(a, MODEL, A_LIM, 0.05)
    assert s.interval == 0.0 and s.risk[0] == pytest.approx(0.1)


def test_capped_when_no_candidate_violates():
    s = select_inspection_interval(_samples(), MODEL, A_LIM, 0.5, intervals=[0, 10, 100])
    assert s.capped and s.interval == 100


def test_invalid_inputs():
    with pytest.raises(ValueError, match="alpha"):
        select_inspection_interval(_samples(), MODEL, A_LIM, 1.5)
    with pytest.raises(ValueError, match="criterion"):
        select_inspection_interval(_samples(), MODEL, A_LIM, 0.1, criterion="other")
