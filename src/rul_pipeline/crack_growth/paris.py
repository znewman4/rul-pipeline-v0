"""Paris-law fatigue crack growth under constant-amplitude Mode-I loading.

    da/dN   = C (Delta K)^m
    Delta K = Y Delta_sigma sqrt(pi a)          (structural.mode_i)

Units: a [m], Delta_sigma [Pa], Delta K [Pa sqrt(m)], N [cycles],
C [m/cycle / (Pa sqrt(m))^m]. Note C's numerical value depends on the units
of Delta K *and* on m -- use :func:`paris_C_to_SI` when importing published
constants (usually quoted with Delta K in MPa sqrt(m) or N mm^-3/2).

Closed form (constant Y): write B = C (Y Delta_sigma sqrt(pi))^m, so that
da/dN = B a^(m/2), and eps = m/2 - 1. Then

    N(a0 -> a_c) = (a0^-eps - a_c^-eps) / (eps B)        m != 2
                 = ln(a_c / a0) / B                      m == 2

and, inverting for the crack size after N cycles,

    a(N) = a0 (1 - eps B N a0^eps)^(-1/eps)              m != 2
         = a0 exp(B N)                                   m == 2

Both are evaluated with expm1/log1p so that they are accurate as m -> 2.
For m > 2 the crack size reaches infinity in finite N ("runaway"): a(N) = inf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad, solve_ivp

from ..structural.mode_i import Y_EDGE_CRACK, stress_intensity_mode_i
from ..utils.validation import check_finite, check_positive

#: Pa sqrt(m) per unit of Delta K in the two common published unit systems.
MPA_SQRT_M: float = 1e6
N_MM_POW_MINUS_3_2: float = 1e6 * np.sqrt(1e-3)       # 1 N mm^-3/2 = 31 622.8 Pa sqrt(m)


def paris_C_to_SI(C: float, m: float, *, da_unit_m: float, dK_unit_Pa_sqrt_m: float) -> float:
    """Convert a Paris coefficient to SI (da/dN in m/cycle, Delta K in Pa sqrt(m)).

    If da/dN = C dK^m with da in units of ``da_unit_m`` metres and dK in units
    of ``dK_unit_Pa_sqrt_m`` Pa sqrt(m), then C_SI = C da_unit_m / dK_unit^m.

    Example (BS 7910 simplified law, steel in air, mm/cycle and N mm^-3/2)::

        paris_C_to_SI(5.21e-13, 3, da_unit_m=1e-3, dK_unit_Pa_sqrt_m=N_MM_POW_MINUS_3_2)
        # -> 1.6475e-29
    """
    return float(C) * da_unit_m / dK_unit_Pa_sqrt_m ** float(m)


#: BS 7910 simplified Paris law for steels in air (R < 0.5): A = 5.21e-13, m = 3
#: (da/dN in mm/cycle, Delta K in N mm^-3/2). This is a mean + 2 SD design curve.
M_BS7910_AIR: float = 3.0
C_BS7910_AIR_SI: float = paris_C_to_SI(5.21e-13, M_BS7910_AIR, da_unit_m=1e-3,
                                       dK_unit_Pa_sqrt_m=N_MM_POW_MINUS_3_2)


@dataclass(frozen=True)
class ParisModel:
    """Parameter bundle for V0 constant-amplitude Paris growth (all SI).

    C [m/cycle/(Pa sqrt(m))^m], m [-], delta_sigma [Pa], Y [-] (constant).
    The loading (delta_sigma) is part of the model: every RUL computed with it
    is conditional on that assumed future loading.
    """

    C: float = C_BS7910_AIR_SI
    m: float = M_BS7910_AIR
    delta_sigma: float = 100e6
    Y: float = Y_EDGE_CRACK

    def __post_init__(self) -> None:
        _check_params(self.C, self.m, self.delta_sigma, self.Y)

    @property
    def B(self) -> float:
        """da/dN = B a^(m/2), B = C (Y Delta_sigma sqrt(pi))^m."""
        return self.C * (self.Y * self.delta_sigma * np.sqrt(np.pi)) ** self.m

    def growth_rate(self, a: ArrayLike) -> np.ndarray:
        return crack_growth_rate(a, self.C, self.m, self.delta_sigma, self.Y)

    def rul(self, a0: ArrayLike, a_c: float, *, on_exceed: str = "raise") -> np.ndarray:
        return rul_analytical(a0, a_c, self.C, self.m, self.delta_sigma, self.Y, on_exceed=on_exceed)

    def crack_size_after(self, a0: ArrayLike, n_cycles: ArrayLike) -> np.ndarray:
        return crack_size_after(a0, n_cycles, self.C, self.m, self.delta_sigma, self.Y)


def _check_params(C: float, m: float, delta_sigma: float, Y) -> None:
    check_positive(C, "C (Paris coefficient)")
    m = float(check_finite(m, "m (Paris exponent)"))
    if not 0 < m <= 10:
        raise ValueError(f"Paris exponent m = {m} is outside the physically plausible range (0, 10].")
    check_positive(delta_sigma, "delta_sigma (stress range, Pa)")
    if not callable(Y):
        check_positive(Y, "Y (geometry factor)")


def crack_growth_rate(a: ArrayLike, C: float, m: float, delta_sigma: float, Y: float = Y_EDGE_CRACK) -> np.ndarray:
    """da/dN = C (Y Delta_sigma sqrt(pi a))^m  [m/cycle]."""
    _check_params(C, m, delta_sigma, Y)
    return C * stress_intensity_mode_i(a, delta_sigma, Y) ** m


def _check_a0_ac(a0: ArrayLike, a_c: float, on_exceed: str) -> tuple[np.ndarray, float, np.ndarray]:
    a0 = check_positive(a0, "a0 (initial crack size, m)")
    a_c = float(check_positive(a_c, "a_c (critical crack size, m)"))
    exceeded = a0 >= a_c
    if on_exceed not in ("raise", "zero"):
        raise ValueError("on_exceed must be 'raise' or 'zero'.")
    if on_exceed == "raise" and np.any(exceeded):
        raise ValueError(
            f"a0 >= a_c for {int(np.sum(exceeded))} value(s) (max a0 = {a0.max():.4g} m, a_c = {a_c:.4g} m): "
            "the crack has already reached the critical size. Use on_exceed='zero' to return RUL = 0."
        )
    return a0, a_c, exceeded


def rul_analytical(
    a0: ArrayLike,
    a_c: float,
    C: float,
    m: float,
    delta_sigma: float,
    Y: float = Y_EDGE_CRACK,
    *,
    on_exceed: str = "raise",
) -> np.ndarray:
    """Closed-form cycles from a0 to a_c for constant Y, Delta_sigma, C, m (vectorised in a0).

    ``on_exceed='zero'`` returns 0 for a0 >= a_c instead of raising (useful for
    posterior samples that already exceed the critical size).
    """
    _check_params(C, m, delta_sigma, Y)
    a0, a_c, exceeded = _check_a0_ac(a0, a_c, on_exceed)
    B = C * (Y * delta_sigma * np.sqrt(np.pi)) ** m
    eps = m / 2.0 - 1.0
    a0_eff = np.minimum(a0, a_c)
    if eps == 0.0:
        N = np.log(a_c / a0_eff) / B
    else:
        # (a0^-eps - a_c^-eps) / eps  ==  (expm1(-eps ln a0) - expm1(-eps ln a_c)) / eps, stable as eps -> 0
        N = (np.expm1(-eps * np.log(a0_eff)) - np.expm1(-eps * np.log(a_c))) / (eps * B)
    N = np.where(exceeded, 0.0, N)
    return N if N.ndim else N[()]


def crack_size_after(
    a0: ArrayLike,
    n_cycles: ArrayLike,
    C: float,
    m: float,
    delta_sigma: float,
    Y: float = Y_EDGE_CRACK,
) -> np.ndarray:
    """Closed-form crack size a(N) [m] after ``n_cycles`` from a0 (broadcasts); inf after runaway."""
    _check_params(C, m, delta_sigma, Y)
    a0 = check_positive(a0, "a0 (initial crack size, m)")
    n = check_finite(n_cycles, "n_cycles")
    if np.any(n < 0):
        raise ValueError("n_cycles must be non-negative.")
    B = C * (Y * delta_sigma * np.sqrt(np.pi)) ** m
    eps = m / 2.0 - 1.0
    if eps == 0.0:
        log_a = np.log(a0) + B * n
    else:
        arg = -eps * B * n * a0**eps                    # ln a = ln a0 - log1p(arg) / eps
        with np.errstate(invalid="ignore", divide="ignore"):
            log_a = np.where(arg > -1.0, np.log(a0) - np.log1p(np.maximum(arg, -1.0)) / eps, np.inf)
    with np.errstate(over="ignore"):
        a = np.exp(log_a)
    return a if a.ndim else a[()]


def rul_numerical(
    a0: ArrayLike,
    a_c: float,
    C: float,
    m: float,
    delta_sigma: float,
    Y: float | Callable[[float], float] = Y_EDGE_CRACK,
    *,
    on_exceed: str = "raise",
    rtol: float = 1e-10,
) -> np.ndarray:
    """Cycles from a0 to a_c by quadrature, N = integral_{a0}^{a_c} da / (C Delta K(a)^m).

    ``Y`` may be a callable Y(a) (e.g. an FE-derived geometry factor or a
    finite-width correction), which the closed form cannot handle. Integration
    is in ln(a) (da = a dln a) because the integrand varies over decades.
    """
    _check_params(C, m, delta_sigma, Y)
    a0, a_c, exceeded = _check_a0_ac(a0, a_c, on_exceed)
    Y_fn = Y if callable(Y) else (lambda a, _Y=float(Y): _Y)

    def integrand(log_a: float) -> float:
        a = np.exp(log_a)
        return a / (C * (Y_fn(a) * delta_sigma * np.sqrt(np.pi * a)) ** m)

    flat = a0.reshape(-1)
    N = np.array([0.0 if ai >= a_c else quad(integrand, np.log(ai), np.log(a_c), epsrel=rtol, limit=200)[0]
                  for ai in flat]).reshape(a0.shape)
    return N if N.ndim else N[()]


def grow_crack_numerical(
    a0: float,
    n_cycles: ArrayLike,
    C: float,
    m: float,
    delta_sigma: float,
    Y: float | Callable[[float], float] = Y_EDGE_CRACK,
    *,
    a_stop: float | None = None,
    rtol: float = 1e-9,
) -> np.ndarray:
    """Integrate da/dN = C (Y(a) Delta_sigma sqrt(pi a))^m forward (RK45) and return a at ``n_cycles``.

    Cycle-stepping counterpart to :func:`crack_size_after` for general Y(a).
    If ``a_stop`` is given, integration stops when a reaches it; later
    requested cycle counts return ``inf`` (failed).
    """
    _check_params(C, m, delta_sigma, Y)
    a0 = float(check_positive(a0, "a0 (initial crack size, m)"))
    n = np.atleast_1d(check_finite(n_cycles, "n_cycles"))
    if np.any(n < 0) or np.any(np.diff(n) < 0):
        raise ValueError("n_cycles must be non-negative and non-decreasing.")
    Y_fn = Y if callable(Y) else (lambda a, _Y=float(Y): _Y)

    def rhs(_N, y):  # integrate ln a: d ln a / dN = (da/dN) / a, smooth over decades of a
        a = np.exp(y[0])
        return [C * (Y_fn(a) * delta_sigma * np.sqrt(np.pi * a)) ** m / a]

    events = None
    if a_stop is not None:
        def hit(_N, y):
            return y[0] - np.log(a_stop)
        hit.terminal, hit.direction = True, 1
        events = [hit]
    sol = solve_ivp(rhs, (0.0, float(n[-1])), [np.log(a0)], t_eval=n, events=events, rtol=rtol, atol=1e-12)
    out = np.full(n.shape, np.inf)
    out[: sol.y.shape[1]] = np.exp(sol.y[0])
    return out
