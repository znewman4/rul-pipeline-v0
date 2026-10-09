#!/usr/bin/env python
"""TFM of the BristolFE synthetic FMC, processed and formatted exactly like reproduce_brain_tfm.py.

    python scripts/synthetic_tfm_brain_style.py

Same ``tfm_image`` call (Hilbert on, linear interpolation, no filter / gating / time shift),
BRAIN's pixel grid (0.15 mm, x -19.845..19.755 mm, z 0..116.85 mm), BRAIN's side_view colour
map and -40..0 dB display range, dB re image max. The only geometry edit is shifting the
array to be centred on x = 0 (BRAIN convention); the FMC data are untouched.
Writes outputs/tfm_synthetic_brain_style.png.
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

FILE = Path("data/forwardlibrary/baseline_fmc.mat")
REF = Path("data/confirmation_baselines/side_view/experimental_tfm_brain.mat")  # grid / colours only
OUT = Path("outputs/tfm_synthetic_brain_style.png")


def main() -> None:
    with h5py.File(FILE, "r") as f:
        g = f["fmc_data"]
        fmc = np.transpose(g["fmc"][()], (2, 1, 0))  # (tx, rx, nt)
        time_s = g["time"][()].ravel()
        ep = g["element_positions"][()]  # (2, N): x, y
        c = g["longitudinal_velocity"][()].item()
    elements = np.column_stack((ep[0] - ep[0].mean(), np.zeros(ep.shape[1])))  # centred, array at z = 0

    ref = loadmat(REF)
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

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 17))
    im = ax.imshow(db, extent=[x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3],
                   aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(title="synthetic baseline_fmc: Python (rul_pipeline)", xlabel="x (mm)", ylabel="z (mm)")
    fig.colorbar(im, ax=ax, label="dB re max", shrink=0.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
