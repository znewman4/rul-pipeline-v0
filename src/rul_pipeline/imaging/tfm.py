"""Basic longitudinal-wave Total Focusing Method imaging.

Borrowed from znewman4/ultrasonic-forward-model (``src/imaging/tfm.py``).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def analytic_signal(data: ArrayLike) -> NDArray[np.complex128]:
    """Analytic signal along the last axis, as in Bristol's BRAIN ``fn_fast_DAS``.

    BRAIN (``ndtatbristol/brain1``) applies the Hilbert transform in the
    frequency domain: keep spectrum bins ``k`` (1-based, as in MATLAB) with
    ``k < Nt/2``, zero the rest (negative frequencies *and* Nyquist), inverse
    transform and multiply by 2 (the GPU path of ``fn_fast_DAS3``). Unlike
    ``scipy.signal.hilbert`` there is no special handling of the DC and Nyquist
    bins, and no zero-padding to a power of two inside the delay-and-sum.
    """
    arr = np.asarray(data, dtype=float)
    nt = arr.shape[-1]
    keep = (np.arange(1, nt + 1) < nt / 2).astype(float)
    return np.fft.ifft(np.fft.fft(arr, axis=-1) * keep, axis=-1) * 2.0


def tfm_image(
    fmc_data: ArrayLike,
    time_s: ArrayLike,
    element_coordinates: ArrayLike,
    x_grid_m: ArrayLike,
    z_grid_m: ArrayLike,
    wave_speed_m_s: float,
    *,
    pixel_chunk_size: int = 128,
    hilbert_on: bool = True,
) -> FloatArray:
    """Form a TFM image by linearly interpolating and summing FMC data.

    For every image pixel, samples are evaluated at the transmitter-pixel-
    receiver travel times. Values outside the recorded time interval contribute
    zero. The coherent sum is returned as its absolute magnitude, without
    internal normalization.

    With ``hilbert_on`` (default, as in Bristol's BRAIN ``focal_law.hilbert_on``)
    each A-scan is converted to its analytic signal before delay-and-sum, so the
    magnitude is the envelope image. With ``hilbert_on=False`` the real RF data
    are summed, giving carrier fringes in the image. Complex ``fmc_data`` is
    accepted and used as-is.

    Returns:
        Image with shape ``(len(z_grid_m), len(x_grid_m))``.
    """
    fmc = np.asarray(fmc_data)
    time = np.asarray(time_s, dtype=float)
    elements = np.asarray(element_coordinates, dtype=float)
    x_grid = np.asarray(x_grid_m, dtype=float)
    z_grid = np.asarray(z_grid_m, dtype=float)
    if fmc.ndim != 3 or fmc.shape[0] != fmc.shape[1]:
        raise ValueError("fmc_data must have shape (N, N, Nt)")
    n, _, nt = fmc.shape
    if time.shape != (nt,) or not np.all(np.isfinite(time)):
        raise ValueError("time_s must be finite with shape (Nt,)")
    if nt < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time_s must be strictly increasing with at least 2 samples")
    if elements.shape != (n, 2) or not np.all(np.isfinite(elements)):
        raise ValueError("element_coordinates must be finite with shape (N, 2)")
    if x_grid.ndim != 1 or z_grid.ndim != 1 or x_grid.size == 0 or z_grid.size == 0:
        raise ValueError("x_grid_m and z_grid_m must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(x_grid)) or not np.all(np.isfinite(z_grid)):
        raise ValueError("imaging grids must contain finite values")
    try:
        speed = float(wave_speed_m_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("wave_speed_m_s must be positive and finite") from exc
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("wave_speed_m_s must be positive and finite")
    if not isinstance(pixel_chunk_size, int) or pixel_chunk_size <= 0:
        raise ValueError("pixel_chunk_size must be a positive integer")

    xx, zz = np.meshgrid(x_grid, z_grid)
    pixels = np.column_stack((xx.ravel(), zz.ravel()))
    if hilbert_on and not np.iscomplexobj(fmc):
        fmc = analytic_signal(fmc)
    traces = fmc.reshape(n * n, nt)
    pair_indices = np.arange(n * n)[np.newaxis, :]
    image = np.empty(pixels.shape[0], dtype=float)

    for start in range(0, pixels.shape[0], pixel_chunk_size):
        stop = min(start + pixel_chunk_size, pixels.shape[0])
        distances = np.linalg.norm(
            pixels[start:stop, np.newaxis, :] - elements[np.newaxis, :, :], axis=2
        )
        delays = (
            distances[:, :, np.newaxis] + distances[:, np.newaxis, :]
        ).reshape(stop - start, n * n) / speed
        upper = np.searchsorted(time, delays, side="right")
        valid = (delays >= time[0]) & (delays <= time[-1])
        upper = np.clip(upper, 1, nt - 1)
        lower = upper - 1
        t0 = time[lower]
        weight = (delays - t0) / (time[upper] - t0)
        sampled = (
            traces[pair_indices, lower] * (1.0 - weight)
            + traces[pair_indices, upper] * weight
        )
        sampled[~valid] = 0.0
        image[start:stop] = np.abs(np.sum(sampled, axis=1))
    return image.reshape(z_grid.size, x_grid.size)
