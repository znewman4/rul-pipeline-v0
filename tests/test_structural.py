"""Mode-I structural map K_I = Y sigma sqrt(pi a)."""

import numpy as np
import pytest

from rul_pipeline.structural import Y_EDGE_CRACK, critical_crack_size, stress_intensity_mode_i


def test_matches_hand_calculation():
    # a = 2 mm, sigma = 100 MPa, Y = 1.12:  K = 1.12 * 100e6 * sqrt(pi * 0.002) = 8.878 MPa sqrt(m)
    K = stress_intensity_mode_i(2e-3, 100e6, 1.12)
    assert K == pytest.approx(1.12 * 100e6 * np.sqrt(np.pi * 0.002))
    assert K / 1e6 == pytest.approx(8.878, abs=1e-3)
    assert np.ndim(K) == 0


def test_vector_input_and_sqrt_scaling():
    a = np.array([1e-3, 4e-3, 9e-3])
    K = stress_intensity_mode_i(a, 50e6)
    assert K.shape == (3,)
    np.testing.assert_allclose(K / K[0], [1.0, 2.0, 3.0])            # K proportional to sqrt(a)
    np.testing.assert_allclose(K, Y_EDGE_CRACK * 50e6 * np.sqrt(np.pi * a))


def test_broadcasts_sampled_Y_and_sigma():
    a = np.full(4, 2e-3)
    Y = np.array([1.0, 1.12, 1.2, 1.3])
    sigma = np.array([[80e6], [120e6]])                              # (2, 1) x (4,) -> (2, 4)
    K = stress_intensity_mode_i(a, sigma, Y)
    assert K.shape == (2, 4)
    assert K[1, 3] == pytest.approx(1.3 * 120e6 * np.sqrt(np.pi * 2e-3))


@pytest.mark.parametrize("a, sigma, Y", [(-1e-3, 1e8, 1.12), (0.0, 1e8, 1.12), (1e-3, -1e8, 1.12),
                                         (1e-3, 1e8, 0.0), (np.nan, 1e8, 1.12)])
def test_invalid_inputs_raise(a, sigma, Y):
    with pytest.raises(ValueError):
        stress_intensity_mode_i(a, sigma, Y)


def test_critical_crack_size_inverts_K():
    K_Ic, s_max = 60e6, 200e6
    a_c = critical_crack_size(K_Ic, s_max)
    assert stress_intensity_mode_i(a_c, s_max) == pytest.approx(K_Ic)
    assert a_c == pytest.approx((60 / (1.12 * 200)) ** 2 / np.pi)
