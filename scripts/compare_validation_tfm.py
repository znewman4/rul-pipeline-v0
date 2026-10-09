#!/usr/bin/env python
"""TFM of the BristolFE steel validation FMC next to the BRAIN TFM of the real specimen.

    python scripts/compare_validation_tfm.py

Processing is identical to reproduce_brain_tfm.py: same ``tfm_image`` call (Hilbert on, linear
interpolation, no filter / gating / time shift), BRAIN's pixel grid (0.15 mm, x -19.845..19.755 mm,
z 0..116.85 mm), BRAIN's colour map and display range, dB re image max. The array is shifted to be
centred on x = 0 and placed at z = 0 (array on the top face); FMC data are untouched. The velocity
is the file's ``brain_tfm_velocity`` (5850 m/s, what BRAIN used), not the FE speed (5940 m/s).

Left: Python TFM of the validation FMC. Right: BRAIN TFM of the real specimen. Both real views are
compared numerically; the figure shows the one set by REAL_VIEW.
Writes outputs/tfm_validation_vs_real.png.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy.io import loadmat  # noqa: E402

from rul_pipeline.imaging import tfm_image  # noqa: E402

FILE = Path("data/confirmation_baselines/above_view/simulated_bristolfe_fmc.mat")
REAL = Path("data/confirmation_baselines")
REAL_VIEW = "above_view"
OUT = Path("outputs/tfm_validation_vs_real.png")


def main() -> None:
    with h5py.File(FILE, "r") as f:
        g = f["fmc_data"]
        fmc = np.transpose(g["fmc"][()], (2, 1, 0))  # (tx, rx, nt)
        time_s = g["time"][()].ravel()
        ep = g["element_positions"][()]  # (2, N): x, y
        c = g["brain_tfm_velocity"][()].item()
    elements = np.column_stack((ep[0] - ep[0].mean(), np.zeros(ep.shape[1])))

    refs = {v: loadmat(REAL / v / "experimental_tfm_brain.mat") for v in ("above_view", "side_view")}
    ref = refs[REAL_VIEW]
    shape = ref["cdata"].shape
    xe, ze = ref["x_mm"].ravel(), ref["z_mm"].ravel()
    x = np.linspace(xe[0], xe[-1], shape[1]) * 1e-3
    z = np.linspace(ze[0], ze[-1], shape[0]) * 1e-3
    vmin, vmax = ref["clim"].ravel()
    cmap = ListedColormap(ref["cmap"])

    img = tfm_image(fmc, time_s, elements, x, z, c, pixel_chunk_size=256)
    db = 20 * np.log10(np.maximum(img / img.max(), 1e-6))
    iz, ix = np.unravel_index(db.argmax(), db.shape)
    print(f"c={c:.0f} m/s, FMC {fmc.shape}, time {time_s[-1]*1e6:.1f} us, "
          f"peak (x,z)=({x[ix]*1e3:.2f},{z[iz]*1e3:.2f}) mm")
    for v, r in refs.items():
        corr = np.corrcoef(np.maximum(db, -60).ravel(), np.maximum(r["cdata"], -60).ravel())[0, 1]
        print(f"  dB-image correlation with BRAIN {v}: {corr:.3f}")

    ext = [x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3]
    fig, axes = plt.subplots(1, 2, figsize=(9, 8.5), sharey=True)
    for ax, data, title in zip(axes, (db, ref["cdata"]),
                               ("BristolFE validation: Python (rul_pipeline)", f"real specimen: BRAIN ({REAL_VIEW})")):
        im = ax.imshow(data, extent=ext, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(title=title, xlabel="x (mm)")
        ax.title.set_fontsize(9)
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(im, ax=list(axes), label="dB re max", shrink=0.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
