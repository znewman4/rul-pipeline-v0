"""Toy ray-based FMC forward model, used as a stand-in for a BristolFE library.

This is NOT a physical scattering model. It exists only so the whole pipeline
can be run and tested before the BristolFE export exists. It produces FMC data
with the right *structure* (64-element 5 MHz linear array, tx x rx x time,
arrival times set by real geometry) and a smooth, monotone dependence on crack
size, which is all the inversion machinery needs.

Geometry (2-D, matching BristolFE's plane-strain models)::

      x ->     array on top surface z = 0, element centres x_e
     z  ====================================================
     |
     v                         | crack tip  (x_c, T - a)     <- tip diffraction
                               |
     ==========================|=========================  backwall z = T
                               corner     (x_c, T)         <- corner trap

Each scatterer s contributes, for transmitter e and receiver r,

    A_s(a) * D(theta_e) D(theta_r) / sqrt(r_e r_r / T^2) * w(t - (r_e + r_r) / c)

with w a Hann-windowed toneburst, D(theta) = cos(theta) a crude element
directivity, and 1/sqrt(r) cylindrical (2-D) spreading.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..io.bristolfe import ForwardLibrary
from ..utils.validation import check_positive

#: Longitudinal wave speed in steel, m/s (BristolFE examples: E = 210 GPa, nu = 0.3).
C_L_STEEL = 5900.0


def toneburst(tau: np.ndarray, f_c: float, n_cycles: float) -> np.ndarray:
    """Hann-windowed toneburst centred on ``tau = 0``: cos(2 pi f_c tau) * hann."""
    half = 0.5 * n_cycles / f_c
    inside = np.abs(tau) <= half
    return np.where(inside, np.cos(2 * np.pi * f_c * tau) * np.cos(np.pi * tau / (2 * half)) ** 2, 0.0)


def linear_array_positions(n_elements: int, pitch: float) -> np.ndarray:
    """(n_elements, 2) element centres [x, z] in metres, centred on x = 0, z = 0."""
    x = (np.arange(n_elements) - 0.5 * (n_elements - 1)) * pitch
    return np.column_stack([x, np.zeros_like(x)])


def toy_fmc_library(
    crack_sizes: ArrayLike,
    *,
    n_elements: int = 64,
    pitch: float = 0.6e-3,
    f_c: float = 5e6,
    n_cycles: float = 5.0,
    c: float = C_L_STEEL,
    thickness: float = 20e-3,
    crack_x: float = 0.0,
    fs: float = 25e6,
    tip_amplitude: float = 0.3,
    corner_amplitude: float = 1.0,
    include_backwall: bool = False,
    time: np.ndarray | None = None,
) -> ForwardLibrary:
    """Build a toy FMC forward library with ``dims = ("crack", "tx", "rx", "time")``.

    Defaults describe a 64-element, 5 MHz, 0.6 mm pitch (~lambda/2 in steel) linear
    array on a 20 mm steel plate with a backwall-breaking crack under the array
    centre. ``fs = 25 MHz`` (5x the centre frequency) keeps the 64x64 FMC small;
    the time window is gated to cover the earliest tip and latest corner arrivals
    unless an explicit ``time`` vector [s] is given.

    Crack sizes are in metres. Use :func:`toy_fmc_truth` for an off-grid "true"
    response on the same time base as an existing library.
    """
    a = np.atleast_1d(check_positive(crack_sizes, "crack_sizes"))
    if np.any(a >= thickness):
        raise ValueError("crack_sizes must be smaller than the plate thickness.")
    for name, v in (("pitch", pitch), ("f_c", f_c), ("c", c), ("thickness", thickness), ("fs", fs)):
        check_positive(v, name)

    pos = linear_array_positions(n_elements, pitch)
    xe = pos[:, 0]
    lam = c / f_c
    half_pulse = 0.5 * n_cycles / f_c

    def legs(z: float) -> tuple[np.ndarray, np.ndarray]:
        """Distance and cos(angle) from each element to the point (crack_x, z)."""
        r = np.hypot(xe - crack_x, z)
        return r, z / r

    # Gate: earliest possible tip arrival (largest crack) to latest corner arrival.
    r_min, _ = legs(thickness - a.max())
    r_max, _ = legs(thickness)
    t0 = 2 * r_min.min() / c - 2 * half_pulse
    t1 = 2 * r_max.max() / c + 2 * half_pulse
    if include_backwall:
        t1 = max(t1, 2 * np.hypot(0.5 * (xe[-1] - xe[0]), thickness) / c + 2 * half_pulse)
    if time is None:
        time = t0 + np.arange(int(np.ceil((t1 - t0) * fs))) / fs
    else:
        time = np.asarray(time, dtype=float)
        fs = 1.0 / float(np.median(np.diff(time)))

    def scatterer(z: float, amp: float) -> np.ndarray:
        r, cos_t = legs(z)
        delay = (r[:, None] + r[None, :]) / c                       # (tx, rx)
        gain = amp * (cos_t[:, None] * cos_t[None, :]) / np.sqrt(r[:, None] * r[None, :] / thickness**2)
        return gain[..., None] * toneburst(time[None, None, :] - delay[..., None], f_c, n_cycles)

    backwall = 0.0
    if include_backwall:  # specular reflection, midpoint between tx and rx
        half_sep = 0.5 * np.abs(xe[:, None] - xe[None, :])
        path = 2 * np.hypot(half_sep, thickness)
        cos_t = thickness / (0.5 * path)
        gain = 2.0 * cos_t**2 / (path / (2 * thickness))
        backwall = gain[..., None] * toneburst(time[None, None, :] - path[..., None] / c, f_c, n_cycles)

    responses = np.empty((a.size, n_elements, n_elements, time.size))
    for i, ai in enumerate(a):
        corner = corner_amplitude * (1.0 - np.exp(-ai / lam))       # grows from 0 for tiny cracks
        responses[i] = scatterer(thickness - ai, tip_amplitude) + scatterer(thickness, corner) + backwall

    return ForwardLibrary(
        crack_sizes=a,
        responses=responses,
        dims=("crack", "tx", "rx", "time"),
        time=time,
        fs=fs,
        element_positions=pos,
        units={"crack_sizes": "m", "time": "s", "fs": "Hz", "responses": "arb. (toy model)"},
        source="toy_fmc_library",
        metadata={
            "n_elements": n_elements, "pitch_m": pitch, "f_c_Hz": f_c, "n_cycles": n_cycles,
            "c_m_per_s": c, "thickness_m": thickness, "crack_x_m": crack_x,
            "include_backwall": include_backwall, "tip_amplitude": tip_amplitude,
            "corner_amplitude": corner_amplitude,
        },
    )


def toy_fmc_truth(a_true: float, library: ForwardLibrary) -> np.ndarray:
    """Clean toy response at an arbitrary (off-grid) ``a_true`` [m], on ``library``'s time base.

    ``library`` must come from :func:`toy_fmc_library`; its stored parameters are reused.
    """
    if library.source != "toy_fmc_library":
        raise ValueError("toy_fmc_truth needs a library built by toy_fmc_library.")
    m = library.metadata
    one = toy_fmc_library(
        [a_true], n_elements=m["n_elements"], pitch=m["pitch_m"], f_c=m["f_c_Hz"], n_cycles=m["n_cycles"],
        c=m["c_m_per_s"], thickness=m["thickness_m"], crack_x=m["crack_x_m"],
        include_backwall=m["include_backwall"], time=library.time,
        tip_amplitude=m["tip_amplitude"], corner_amplitude=m["corner_amplitude"],
    )
    return one.responses[0]
