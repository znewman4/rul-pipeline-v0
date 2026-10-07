"""Synthetic noisy measurements y = F(a_true) + e for testing the inversion.

Two ways to choose the clean signal:

1. ``measurement_from_library``: take library entry i as the truth (fast, but
   commits the "inverse crime" of generating and inverting data with the same
   discretised model -- the posterior will look better than it should).
2. ``add_measurement_noise`` on any clean array you supply, e.g. a separate,
   off-grid BristolFE run or ``toy_fmc_library([a_true])``. Prefer this.

Noise level can be set as
    sigma          absolute standard deviation (same unit as y)
    rms_fraction   sigma = rms_fraction * RMS(y_clean)
    snr_db         sigma = ref(y_clean) / 10^(snr_db / 20), ref = peak |y| (NDT
                   convention, default) or RMS(y)

Noise colour:
    "white"        i.i.d. N(0, sigma^2) -- exactly the V0 likelihood assumption
    "bandlimited"  white noise passed through a Gaussian pass-band around the
                   array centre frequency (e.g. 5 MHz, 70 % -6 dB bandwidth),
                   rescaled to sigma. This mimics receiver electronic noise
                   after the transducer/amplifier band-pass, and is correlated
                   in time, so the V0 i.i.d. likelihood is then (deliberately)
                   mis-specified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.bristolfe import ForwardLibrary
from ..utils.validation import check_finite, check_positive


@dataclass(frozen=True)
class SyntheticMeasurement:
    """A noisy measurement and the ground truth that produced it."""

    y: np.ndarray            # noisy measurement, same shape as one library response
    y_clean: np.ndarray      # noise-free truth F(a_true)
    a_true: float            # true crack size [m]
    sigma: float             # noise standard deviation actually used


def noise_sigma(
    y_clean: np.ndarray,
    *,
    sigma: float | None = None,
    rms_fraction: float | None = None,
    snr_db: float | None = None,
    snr_reference: str = "peak",
) -> float:
    """Resolve exactly one of ``sigma`` / ``rms_fraction`` / ``snr_db`` to an absolute sigma."""
    given = [k for k, v in (("sigma", sigma), ("rms_fraction", rms_fraction), ("snr_db", snr_db)) if v is not None]
    if len(given) != 1:
        raise ValueError(f"Specify exactly one of sigma, rms_fraction, snr_db; got {given or 'none'}.")
    y_clean = check_finite(y_clean, "y_clean")
    rms = float(np.sqrt(np.mean(y_clean**2)))
    if sigma is not None:
        return float(check_positive(sigma, "sigma"))
    if rms_fraction is not None:
        return float(check_positive(rms_fraction, "rms_fraction")) * rms
    if snr_reference == "peak":
        ref = float(np.max(np.abs(y_clean)))
    elif snr_reference == "rms":
        ref = rms
    else:
        raise ValueError("snr_reference must be 'peak' or 'rms'.")
    if ref == 0:
        raise ValueError("Cannot set noise by SNR for an all-zero clean signal.")
    return ref / 10 ** (float(snr_db) / 20)


def bandlimited_noise(
    shape: tuple[int, ...],
    sigma: float,
    *,
    fs: float,
    f_c: float = 5e6,
    fractional_bandwidth: float = 0.7,
    time_axis: int = -1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Gaussian noise with a Gaussian pass-band spectrum, rescaled to sample std ``sigma``.

    The -6 dB (half-amplitude) full bandwidth is ``fractional_bandwidth * f_c``,
    i.e. |H(f)| = exp(-(f - f_c)^2 / (2 s^2)) with s = (B/2) / sqrt(2 ln 2).
    """
    rng = np.random.default_rng() if rng is None else rng
    white = rng.standard_normal(shape)
    n_t = shape[time_axis]
    f = np.fft.rfftfreq(n_t, d=1.0 / fs)
    s = 0.5 * fractional_bandwidth * f_c / np.sqrt(2 * np.log(2))
    h = np.exp(-((f - f_c) ** 2) / (2 * s**2))
    spec = np.fft.rfft(white, axis=time_axis)
    spec *= np.expand_dims(h, tuple(ax for ax in range(len(shape)) if ax != time_axis % len(shape)))
    coloured = np.fft.irfft(spec, n=n_t, axis=time_axis)
    return sigma * coloured / coloured.std()


def add_measurement_noise(
    y_clean: np.ndarray,
    *,
    sigma: float | None = None,
    rms_fraction: float | None = None,
    snr_db: float | None = None,
    snr_reference: str = "peak",
    colour: str = "white",
    fs: float | None = None,
    f_c: float = 5e6,
    fractional_bandwidth: float = 0.7,
    time_axis: int = -1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Return ``(y_clean + e, sigma)`` with e white or band-limited Gaussian noise."""
    rng = np.random.default_rng() if rng is None else rng
    y_clean = check_finite(y_clean, "y_clean")
    s = noise_sigma(y_clean, sigma=sigma, rms_fraction=rms_fraction, snr_db=snr_db, snr_reference=snr_reference)
    if colour == "white":
        e = s * rng.standard_normal(y_clean.shape)
    elif colour == "bandlimited":
        if fs is None:
            raise ValueError("Band-limited noise needs the sampling frequency fs.")
        e = bandlimited_noise(y_clean.shape, s, fs=fs, f_c=f_c, fractional_bandwidth=fractional_bandwidth,
                              time_axis=time_axis, rng=rng)
    else:
        raise ValueError("colour must be 'white' or 'bandlimited'.")
    return y_clean + e, s


def measurement_from_library(
    library: ForwardLibrary,
    true_index: int,
    *,
    rng: np.random.Generator | None = None,
    **noise_kwargs,
) -> SyntheticMeasurement:
    """Use ``library.responses[true_index]`` as truth and add noise (see module docs)."""
    if not -library.n_candidates <= true_index < library.n_candidates:
        raise IndexError(f"true_index {true_index} out of range for {library.n_candidates} candidates.")
    y_clean = library.responses[true_index]
    if noise_kwargs.get("colour") == "bandlimited":
        noise_kwargs.setdefault("fs", library.fs)
        if "time" in library.measurement_dims:
            noise_kwargs.setdefault("time_axis", library.measurement_dims.index("time"))
    y, s = add_measurement_noise(y_clean, rng=rng, **noise_kwargs)
    return SyntheticMeasurement(y=y, y_clean=y_clean.copy(), a_true=float(library.crack_sizes[true_index]), sigma=s)
