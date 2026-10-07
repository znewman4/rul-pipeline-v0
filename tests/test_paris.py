"""Paris-law crack growth: closed form vs numerical, monotonicity, invalid input."""

import numpy as np
import pytest

from rul_pipeline.crack_growth import (
    C_BS7910_AIR_SI,
    N_MM_POW_MINUS_3_2,
    ParisModel,
    crack_size_after,
    grow_crack_numerical,
    paris_C_to_SI,
    rul_analytical,
    rul_numerical,
)

C, M, DS, Y = C_BS7910_AIR_SI, 3.0, 100e6, 1.12
A0, AC = 2e-3, 10e-3


def test_bs7910_constant_unit_conversion():
    # 5.21e-13 mm/cycle per (N mm^-3/2)^3  ->  1.6475e-29 m/cycle per (Pa sqrt(m))^3
    assert C_BS7910_AIR_SI == pytest.approx(1.6475e-29, rel=1e-4)
    # same law quoted in MPa sqrt(m) and m/cycle: 1.6475e-11
    assert paris_C_to_SI(1.6475e-11, 3, da_unit_m=1.0, dK_unit_Pa_sqrt_m=1e6) == pytest.approx(C_BS7910_AIR_SI, rel=1e-4)
    assert N_MM_POW_MINUS_3_2 == pytest.approx(31622.7766)


def test_closed_form_matches_hand_calculation():
    # B = C (Y dS sqrt(pi))^3; N = (a0^-1/2 - ac^-1/2) / (0.5 B)
    B = C * (Y * DS * np.sqrt(np.pi)) ** 3
    expected = (A0**-0.5 - AC**-0.5) / (0.5 * B)
    assert rul_analytical(A0, AC, C, M, DS, Y) == pytest.approx(expected, rel=1e-12)
    assert 1.5e5 < expected < 2.5e5                                   # ~1.9e5 cycles


@pytest.mark.parametrize("m", [1.5, 2.0, 2.0 + 1e-9, 2.5, 3.0, 4.0])
def test_analytical_and_numerical_rul_agree(m):
    C_m = 1e-11 * (1e6) ** (-m)                                       # 1e-11 m/cycle at dK = 1 MPa sqrt(m)
    a0 = np.array([0.5e-3, 2e-3, 6e-3])
    exact = rul_analytical(a0, AC, C_m, m, DS, Y)
    numeric = rul_numerical(a0, AC, C_m, m, DS, Y)
    np.testing.assert_allclose(numeric, exact, rtol=1e-7)


def test_m_equal_2_branch_is_continuous():
    C2 = 1e-11 * 1e-12
    n2 = rul_analytical(A0, AC, C2, 2.0, DS, Y)
    assert n2 == pytest.approx(np.log(AC / A0) / (C2 * (Y * DS * np.sqrt(np.pi)) ** 2))
    assert rul_analytical(A0, AC, C2, 2.0 + 1e-7, DS, Y) == pytest.approx(n2, rel=1e-5)


def test_numerical_with_callable_Y_equals_constant_Y():
    assert rul_numerical(A0, AC, C, M, DS, lambda a: Y) == pytest.approx(rul_analytical(A0, AC, C, M, DS, Y), rel=1e-8)


def test_rul_decreases_as_initial_crack_grows():
    N = rul_analytical(np.linspace(0.5e-3, 9e-3, 30), AC, C, M, DS, Y)
    assert np.all(np.diff(N) < 0)


def test_rul_decreases_as_stress_range_increases():
    N = np.array([rul_analytical(A0, AC, C, M, ds, Y) for ds in (50e6, 100e6, 150e6, 200e6)])
    assert np.all(np.diff(N) < 0)
    assert N[0] / N[1] == pytest.approx(2.0**M)                       # N proportional to dS^-m


def test_crack_size_after_inverts_rul():
    model = ParisModel(C, M, DS, Y)
    N_f = model.rul(A0, AC)
    assert model.crack_size_after(A0, N_f) == pytest.approx(AC, rel=1e-10)
    assert model.crack_size_after(A0, 0.0) == pytest.approx(A0)
    a = model.crack_size_after(A0, np.array([0.5, 0.99, 1.0, 5.0]) * N_f)
    assert np.all(np.diff(a[:3]) > 0) and np.isfinite(a[2])
    a_runaway = crack_size_after(A0, 3.0 * N_f / (1 - (A0 / AC) ** 0.5), C, M, DS, Y)
    assert np.isinf(a_runaway)                                        # beyond finite-time blow-up (m > 2)


def test_ode_stepping_matches_closed_form():
    n = np.linspace(0, 1.5e5, 7)
    np.testing.assert_allclose(grow_crack_numerical(A0, n, C, M, DS, Y), crack_size_after(A0, n, C, M, DS, Y),
                               rtol=1e-6)
    stopped = grow_crack_numerical(A0, [0, 1e5, 1e7], C, M, DS, Y, a_stop=AC)
    assert np.isinf(stopped[-1]) and np.isfinite(stopped[1])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"a0": 0.0}, "strictly positive"),
        ({"a0": -1e-3}, "strictly positive"),
        ({"a0": AC}, "already reached"),
        ({"a0": 2 * AC}, "already reached"),
        ({"C": 0.0}, "Paris coefficient"),
        ({"C": -1e-30}, "Paris coefficient"),
        ({"m": 0.0}, "plausible range"),
        ({"m": np.inf}, "non-finite"),
        ({"m": 20.0}, "plausible range"),
        ({"delta_sigma": 0.0}, "stress range"),
        ({"a0": np.nan}, "non-finite"),
        ({"a_c": np.inf}, "non-finite"),
    ],
)
def test_invalid_inputs_raise(kwargs, message):
    args = dict(a0=A0, a_c=AC, C=C, m=M, delta_sigma=DS, Y=Y) | kwargs
    for fn in (rul_analytical, rul_numerical):
        with pytest.raises(ValueError, match=message):
            fn(**args)


def test_on_exceed_zero_returns_zero_rul_for_failed_samples():
    N = rul_analytical(np.array([2e-3, 10e-3, 12e-3]), AC, C, M, DS, Y, on_exceed="zero")
    assert N[0] > 0 and N[1] == 0 and N[2] == 0
