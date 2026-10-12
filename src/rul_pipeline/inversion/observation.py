"""Common observation operator for real data and candidates (applied identically to both).

``Preprocess``: one global time shift of the REAL data, an optional zero-phase band-pass, a time
window. Nothing is ever normalised per trace; amplitude calibration is a separate, explicit gain
in the likelihood. The shift is a single scalar for the whole acquisition (calibration, to be
reported), never a per-trace alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


@dataclass(frozen=True)
class Preprocess:
    window_us: tuple[float, float] = (8.0, 25.0)
    band_mhz: tuple[float, float] | None = (2.5, 7.5)
    shift_us: float = 0.0           # applied to the REAL data: real(t) -> real(t + shift)

    def describe(self) -> str:
        band = "none" if self.band_mhz is None else f"{self.band_mhz[0]:g}-{self.band_mhz[1]:g} MHz zero-phase"
        return (f"window {self.window_us[0]:g}-{self.window_us[1]:g} us, band-pass {band}, "
                f"real-data shift {self.shift_us:+.3f} us")


def fractional_shift(x: np.ndarray, shift_s: float, dt: float) -> np.ndarray:
    """y(t) = x(t + shift_s) along the last axis via the Fourier shift theorem (zero-padded)."""
    n = x.shape[-1]
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, dt)
    spec = np.fft.rfft(x, nfft, axis=-1) * np.exp(2j * np.pi * f * shift_s)
    return np.fft.irfft(spec, nfft, axis=-1)[..., :n]


def bandpass(x: np.ndarray, fs: float, band_mhz) -> np.ndarray:
    if band_mhz is None:
        return x
    sos = butter(4, [band_mhz[0] * 1e6, band_mhz[1] * 1e6], "band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def window_slice(time: np.ndarray, window_us) -> slice:
    idx = np.flatnonzero((time * 1e6 >= window_us[0]) & (time * 1e6 <= window_us[1]))
    if idx.size < 8:
        raise ValueError("Time window selects fewer than 8 samples.")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def apply(x: np.ndarray, time: np.ndarray, pre: Preprocess, *, is_real: bool) -> np.ndarray:
    """Filter, (real only) shift, then window. ``x`` has time on the last axis."""
    dt = float(time[1] - time[0])
    y = bandpass(np.asarray(x, dtype=float), 1.0 / dt, pre.band_mhz)
    if is_real and pre.shift_us != 0.0:
        y = fractional_shift(y, pre.shift_us * 1e-6, dt)
    return y[..., window_slice(time, pre.window_us)]


def estimate_time_shift(real: np.ndarray, cand0: np.ndarray, time: np.ndarray, pre: Preprocess,
                        span_us: float = 1.5, step_us: float = 0.04):
    """Global shift of the real data that maximises the envelope correlation with ONE candidate.

    Returns (best_shift_us, shifts_us, correlations). Envelope (not RF) correlation is used because
    RF phase is far more sensitive to geometry error than the shift we want to calibrate.
    """
    shifts = np.arange(-span_us, span_us + 1e-9, step_us)
    ref = np.abs(hilbert(apply(cand0, time, pre, is_real=False), axis=-1)).ravel()
    ref = ref - ref.mean()
    corr = []
    for s in shifts:
        p = Preprocess(pre.window_us, pre.band_mhz, float(s))
        env = np.abs(hilbert(apply(real, time, p, is_real=True), axis=-1)).ravel()
        env = env - env.mean()
        corr.append(float(env @ ref / np.sqrt((env @ env) * (ref @ ref))))
    corr = np.asarray(corr)
    return float(shifts[int(np.argmax(corr))]), shifts, corr
