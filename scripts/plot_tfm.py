#!/usr/bin/env python
"""TFM of the BristolFE baseline FMC, laid out like ultrasonic-forward-model/scripts/09_experiment001_tfm.py.

    python scripts/plot_tfm.py

Two panels (raw time, and time-zero corrected), jet colormap, dB re the global
maximum of both images, -40..0 dB, 0.16 mm pixels, Hilbert on, linear interpolation.

Differences forced by the data (BristolFE .mat v7.3, not BRAIN half-matrix):
  * h5py reverses MATLAB axes, so ``fmc`` is transposed to (tx, rx, nt);
  * the record is only 12.9 us, so 09's two-back-wall-echo offset is not
    available. The offset is instead the first back-wall echo time minus the
    nominal 2 * couplant / c_water + 2 * aluminium_height / c_L.
  * geometry (element positions, array y, speeds) is read from the file;
    z is depth below the array surface, so the array is at z = 0.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

from rul_pipeline.imaging import tfm_image  # noqa: E402

FILE = Path("data/forwardlibrary/baseline_fmc.mat")
OUT = Path("outputs/tfm_comparison.png")
STEP_MM = 0.16
BACKWALL_WINDOW_S = (8.3e-6, 9.0e-6)


def time_zero_offset(fmc, time_s, expected_backwall_s, window=BACKWALL_WINDOW_S):
    """Offset = t_backwall(measured) - expected, median over pulse-echo traces."""
    dt = time_s[1] - time_s[0]
    m = (time_s >= window[0]) & (time_s <= window[1])
    t1 = []
    for i in range(fmc.shape[0]):
        env = np.abs(hilbert(fmc[i, i]))
        k = int(np.argmax(np.where(m, env, -1)))
        y0, y1, y2 = env[k - 1], env[k], env[k + 1]
        t1.append(time_s[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * dt)
    t1 = float(np.median(t1))
    return t1 - expected_backwall_s, t1


def main() -> None:
    with h5py.File(FILE, "r") as f:
        g = f["fmc_data"]
        fmc = np.transpose(g["fmc"][()], (2, 1, 0))  # (tx, rx, nt)
        time_s = g["time"][()].ravel()
        ep = g["element_positions"][()]  # (2, N) after HDF5 reversal: rows are x, y
        width, height = g["model_width"][()].item(), g["model_height"][()].item()
        c_l = g["longitudinal_velocity"][()].item()
        c_w = g["couplant_velocity"][()].item()
        couplant = g["couplant_thickness"][()].item()
    elements = np.column_stack((ep[0], ep[1].max() - ep[1]))  # depth below array surface
    depth = height + couplant
    expected = 2 * couplant / c_w + 2 * height / c_l
    offset, t1 = time_zero_offset(fmc, time_s, expected)
    print(f"FMC {fmc.shape}, back wall {t1*1e6:.3f} us (nominal {expected*1e6:.3f}), offset {offset*1e6:.3f} us")

    x = np.arange(0.0, width * 1e3 + 1e-9, STEP_MM) * 1e-3
    z = np.arange(0.0, depth * 1e3 + 1e-9, STEP_MM) * 1e-3

    images = {}
    for label, shift in (("raw_time", 0.0), ("offset_corrected", offset)):
        images[label] = tfm_image(fmc, time_s - shift, elements, x, z, c_l)

    fig, axes = plt.subplots(1, 2, figsize=(12, 11), sharey=True)
    ref = max(i.max() for i in images.values())
    for ax, (label, img) in zip(axes, images.items()):
        db = 20 * np.log10(np.maximum(img / ref, 1e-6))
        im = ax.imshow(db, extent=[x[0]*1e3, x[-1]*1e3, z[-1]*1e3, z[0]*1e3], aspect="equal",
                       cmap="jet", vmin=-40, vmax=0)
        ax.set(title=f"baseline_fmc: {label}", xlabel="x (mm)")
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(im, ax=axes, label="dB re max", shrink=0.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
