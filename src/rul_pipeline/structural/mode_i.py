"""Version 0 structural map: crack size -> Mode-I stress intensity factor.

    K_I = Y * sigma * sqrt(pi * a)

Units (SI, enforced by convention, not by a units library):
    a      crack size (depth of a surface/edge crack)    [m]
    sigma  remote (far-field) normal stress               [Pa]
    Y      geometry factor                                [-]
    K_I    stress intensity factor                        [Pa sqrt(m)]
           (1 MPa sqrt(m) = 1e6 Pa sqrt(m))

This is a PLACEHOLDER for an FE-derived structural interpretation. A constant
Y = 1.12 is the classical value for a shallow edge crack in a semi-infinite
plate under remote tension (Tada, Paris & Irwin). For a finite plate Y depends
on a/W and on the load type; for irregular geometry it should come from FE
(J-integral / displacement extrapolation) as a function Y(a).

``a``, ``sigma`` and ``Y`` broadcast against each other, so posterior samples
a^(i) can be combined with scalar loads now and with sampled Y^(i), sigma^(i)
later, without changing this code.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..utils.validation import check_positive

#: Edge crack in a semi-infinite plate under remote tension (1.1215 to 4 s.f.).
Y_EDGE_CRACK: float = 1.12


def stress_intensity_mode_i(a: ArrayLike, sigma: ArrayLike, Y: ArrayLike = Y_EDGE_CRACK) -> np.ndarray:
    """K_I = Y sigma sqrt(pi a) [Pa sqrt(m)] for a [m], sigma [Pa]; broadcasts.

    Negative ``sigma`` (compression) is rejected: closed cracks have no Mode-I
    singularity and V0 does not model closure. Pass the stress *range* to get
    Delta K for crack growth.
    """
    a = check_positive(a, "a (crack size, m)")
    sigma = check_positive(sigma, "sigma (stress, Pa)")
    Y = check_positive(Y, "Y (geometry factor)")
    K = Y * sigma * np.sqrt(np.pi * a)
    return K if K.ndim else K[()]


def critical_crack_size(K_Ic: ArrayLike, sigma_max: ArrayLike, Y: ArrayLike = Y_EDGE_CRACK) -> np.ndarray:
    """Crack size at which K_I(a, sigma_max) = K_Ic:  a_c = (K_Ic / (Y sigma_max))^2 / pi  [m].

    Only valid for constant Y (otherwise solve K_I(a) = K_Ic numerically). In
    practice a_c is the minimum of this fracture limit and any net-section /
    plastic-collapse or wall-thickness limit (BS 7910 failure assessment).
    """
    K_Ic = check_positive(K_Ic, "K_Ic (fracture toughness, Pa sqrt(m))")
    sigma_max = check_positive(sigma_max, "sigma_max (Pa)")
    Y = check_positive(Y, "Y (geometry factor)")
    a_c = (K_Ic / (Y * sigma_max)) ** 2 / np.pi
    return a_c if a_c.ndim else a_c[()]
