#!/usr/bin/env python
"""Reproduce the BRAIN TFM images of the real steel notch specimen.

    python scripts/reproduce_brain_tfm.py

For each view in data/confirmation_baselines/ this loads the half-matrix FMC
(experimental_fmc.mat, BRAIN ``exp_data``), expands it to a full matrix by
reciprocity, runs ``rul_pipeline.imaging.tfm_image`` on exactly the pixel grid
of the BRAIN figure, and plots it next to the BRAIN image (extracted from the
.fig by matlab/extract_fig_tfm.m into experimental_tfm_brain.mat) with the same colour map
and display range. Writes outputs/tfm_brain_reproduction.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy.io import loadmat  # noqa: E402

from rul_pipeline.imaging import tfm_image  # noqa: E402

ROOT = Path("data/confirmation_baselines")
VIEWS = {"above_view": "experimental_fmc.mat", "side_view": "experimental_fmc.mat"}
OUT = Path("outputs/tfm_brain_reproduction.png")


def load_full_fmc(path: Path):
    """BRAIN half-matrix exp_data -> full (tx, rx, nt) cube, time [s], elements (N, 2) [m], speed [m/s]."""
    d = loadmat(path, simplify_cells=True)["exp_data"]
    tx, rx = d["tx"].astype(int) - 1, d["rx"].astype(int) - 1
    n = int(max(tx.max(), rx.max())) + 1
    nt = d["time_data"].shape[0]
    fmc = np.zeros((n, n, nt))
    fmc[tx, rx] = d["time_data"].T
    fmc[rx, tx] = d["time_data"].T  # reciprocity
    elements = np.column_stack((d["array"]["el_xc"], d["array"]["el_zc"]))
    speed = float(np.ravel(d["material"]["vel_spherical_harmonic_coeffs"])[0])
    return fmc, np.asarray(d["time"], float), elements, speed


def main() -> None:
    fig, axes = plt.subplots(len(VIEWS), 2, figsize=(9, 17), sharey=True)
    for row, (view, fname) in zip(axes, VIEWS.items()):
        fmc, time_s, elements, c = load_full_fmc(ROOT / view / fname)
        ref = loadmat(ROOT / view / "experimental_tfm_brain.mat")
        brain = ref["cdata"]
        xe, ze = ref["x_mm"].ravel(), ref["z_mm"].ravel()  # image XData/YData: first/last pixel centres
        x = np.linspace(xe[0], xe[-1], brain.shape[1]) * 1e-3
        z = np.linspace(ze[0], ze[-1], brain.shape[0]) * 1e-3
        vmin, vmax = ref["clim"].ravel()
        cmap = ListedColormap(ref["cmap"])

        img = tfm_image(fmc, time_s, elements, x, z, c, pixel_chunk_size=256)
        db = 20 * np.log10(np.maximum(img / img.max(), 1e-6))
        r = np.corrcoef(np.maximum(db, -60).ravel(), np.maximum(brain, -60).ravel())[0, 1]
        pk = lambda a: (x[np.unravel_index(a.argmax(), a.shape)[1]] * 1e3, z[np.unravel_index(a.argmax(), a.shape)[0]] * 1e3)
        print(f"{view}: c={c:.0f} m/s, FMC {fmc.shape}, peak python (x,z)=({pk(db)[0]:.2f},{pk(db)[1]:.2f}) mm, "
              f"BRAIN ({pk(brain)[0]:.2f},{pk(brain)[1]:.2f}) mm, dB-image correlation {r:.3f}")

        ext = [x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3]
        for ax, data, title in zip(row, (db, brain), ("Python (rul_pipeline)", "BRAIN")):
            im = ax.imshow(data, extent=ext, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set(title=f"{view}: {title}", xlabel="x (mm)")
        row[0].set_ylabel("z (mm)")
        fig.colorbar(im, ax=list(row), label="dB re max", shrink=0.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
