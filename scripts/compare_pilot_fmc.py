#!/usr/bin/env python
"""FMC residuals and TFM comparison for the BristolFE crack-length pilot.

    python scripts/compare_pilot_fmc.py <pilot_folder> [--start-us 8 --end-us 30] [--tfm]

<pilot_folder> holds crack_<n>um_top.mat (one per crack extension, 0 = notch only). Writes
residual_metrics.csv, pairwise_residuals.png, baseline_residuals.png (and tfm_pilot_comparison.png
with --tfm) into --out-dir (default outputs/crack_length_pilot). FMC residuals are raw differences (no per-trace normalisation).

TFM uses rul_pipeline.imaging.tfm_image, the same call as compare_validation_tfm.py (Hilbert on, no
filter / gating / time shift, BRAIN velocity 5850 m/s). The array is on the top face (y = 80 mm), so
images are drawn in depth below the array and x relative to the array centre (the notch axis).
The real BRAIN panel is a DIFFERENT specimen (50 x 85 mm, other hole layout) - compare qualitatively.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy.io import loadmat  # noqa: E402

from rul_pipeline.imaging import tfm_image  # noqa: E402

REAL = Path("data/steel_notch_initial_confirmation")


def load_cases(folder: Path):
    cases = []
    for path in sorted(folder.glob("crack_*_top.mat")):
        data = loadmat(path, squeeze_me=True)
        cases.append((float(data["crack_length_m"]), path, data))
    cases.sort(key=lambda c: c[0])
    if len(cases) < 2 or cases[0][0] != 0:
        raise ValueError("Need a notch-only baseline and at least one cracked case.")
    base = cases[0][2]
    for _, path, data in cases:
        for key in ("time", "element_positions", "pulse", "cL", "cS", "crack_angle_deg"):
            if np.shape(data[key]) != np.shape(base[key]) or not np.allclose(data[key], base[key], rtol=1e-10, atol=1e-15):
                raise ValueError(f"{path.name}: incompatible {key}")
        if data["fmc"].shape != base["fmc"].shape or not np.isfinite(data["fmc"]).all():
            raise ValueError(f"{path.name}: invalid FMC")
    return cases


def fmc_residuals(cases, folder: Path, start_us: float, end_us: float) -> None:
    ref = cases[0][2]["fmc"]
    time = np.asarray(cases[0][2]["time"]).ravel()
    mask = (time * 1e6 >= start_us) & (time * 1e6 <= end_us)
    if mask.sum() < 2:
        raise ValueError("Selected time window is empty or too short.")
    sel = [d["fmc"][:, :, mask] for _, _, d in cases]
    scale = np.linalg.norm(sel[0])
    rows = []
    pairwise = np.zeros((len(cases),) * 2)
    for i, (length, path, _) in enumerate(cases):
        delta = sel[i] - sel[0]
        rows.append([path.name, length * 1e3, np.sqrt(np.mean(delta**2)), np.linalg.norm(delta) / scale])
        for j in range(len(cases)):
            pairwise[i, j] = np.linalg.norm(sel[i] - sel[j]) / scale
    with (folder / "residual_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "crack_mm", "residual_rms", "residual_norm_over_baseline"])
        w.writerows(rows)
    for r in rows:
        print(f"  {r[0]}: crack {r[1]:g} mm  residual/baseline = {r[3]:.4f}")

    labels = [f"{c[0] * 1e3:g}" for c in cases]
    fig, ax = plt.subplots()
    im = ax.imshow(pairwise, origin="lower", cmap="viridis")
    ax.set_xticks(range(len(cases)), labels)
    ax.set_yticks(range(len(cases)), labels)
    ax.set(xlabel="Crack extension [mm]", ylabel="Crack extension [mm]",
           title=f"FMC difference: {start_us:g}-{end_us:g} µs")
    fig.colorbar(im, ax=ax, label="Difference norm / notch-only norm")
    fig.tight_layout()
    fig.savefig(folder / "pairwise_residuals.png", dpi=180)
    plt.close(fig)

    tx = min(31, ref.shape[0] - 1)
    res = [v[tx] - sel[0][tx] for v in sel[1:]]
    lim = max(float(np.max(np.abs(v))) for v in res) or 1
    fig, axes = plt.subplots(len(res), 1, squeeze=False, figsize=(9, 3 * len(res)))
    for ax, delta, case in zip(axes[:, 0], res, cases[1:]):
        im = ax.imshow(delta, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[time[mask][0] * 1e6, time[mask][-1] * 1e6, .5, ref.shape[1] + .5])
        ax.set(xlabel="Time [µs]", ylabel="Receiver", title=f"Tx {tx + 1}: {case[0] * 1e3:g} mm minus notch-only")
        fig.colorbar(im, ax=ax, label="Displacement residual")
    fig.tight_layout()
    fig.savefig(folder / "baseline_residuals.png", dpi=180)
    plt.close(fig)
    print("Saved residual_metrics.csv, pairwise_residuals.png, baseline_residuals.png")
    print("These are sensitivity metrics, not noise-calibrated identifiability tests.")


def tfm_comparison(cases, folder: Path, velocity: float, grid_mm: float, view: str, offset_us: float) -> None:
    base = cases[0][2]
    time = np.asarray(base["time"]).ravel() + offset_us * 1e-6  # offset_us shifts the time axis (0 = as validated)
    ep = np.asarray(base["element_positions"])
    top = ep[:, 1].max()
    elements = np.column_stack((ep[:, 0] - ep[:, 0].mean(), np.zeros(len(ep))))  # array at depth 0, centred
    width, height = float(np.asarray(base["params"]["width"]).item()), float(np.asarray(base["params"]["height"]).item())
    x = np.arange(-width / 2, width / 2 + 1e-9, grid_mm * 1e-3)
    z = np.arange(0, height + 1e-9, grid_mm * 1e-3)
    notch_tip_depth = (top - float(np.asarray(base["notch_pts"])[:, 1].max())) * 1e3

    ref = loadmat(REAL / view / "experimental_tfm_brain.mat")
    cmap = ListedColormap(ref["cmap"])
    vmin, vmax = ref["clim"].ravel()
    xe, ze = ref["x_mm"].ravel(), ref["z_mm"].ravel()

    panels = [("Real BRAIN (different specimen)", ref["cdata"], [xe[0], xe[-1], ze[-1], ze[0]])]
    for length, path, data in cases:
        img = tfm_image(data["fmc"], time, elements, x, z, velocity, pixel_chunk_size=256)
        db = 20 * np.log10(np.maximum(img / img.max(), 1e-6))
        iz, ix = np.unravel_index(db.argmax(), db.shape)
        label = "Notch only" if length == 0 else f"Crack +{length * 1e3:g} mm"
        print(f"  {label}: peak at x={x[ix] * 1e3:.2f}, depth={z[iz] * 1e3:.2f} mm "
              f"(expected tip ~{notch_tip_depth - length * 1e3:.1f} mm)")
        np.savez_compressed(folder / (path.stem + "_tfm.npz"), tfm=img, x_mm=x * 1e3, depth_mm=z * 1e3,
                            velocity_m_s=velocity)
        panels.append((label, db, [x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3]))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 6.5))
    for ax, (title, db, ext) in zip(axes, panels):
        im = ax.imshow(db, extent=ext, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("depth below array (mm)")
        ax.set_ylim(80, 0)
    fig.colorbar(im, ax=list(axes), label="dB re image max", shrink=0.6)
    fig.suptitle(f"TFM at {velocity:g} m/s; each panel normalised to its own max", fontsize=10)
    out = folder / "tfm_pilot_comparison.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="folder with crack_*_top.mat (read only)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/crack_length_pilot"))
    p.add_argument("--start-us", type=float, default=8)
    p.add_argument("--end-us", type=float, default=30)
    p.add_argument("--tfm", action="store_true")
    p.add_argument("--velocity", type=float, default=5850)
    p.add_argument("--grid-mm", type=float, default=0.25)
    p.add_argument("--real-view", default="above_view", choices=["above_view", "side_view"])
    p.add_argument("--time-offset-us", type=float, default=0.0)
    a = p.parse_args()
    if a.end_us <= a.start_us:
        p.error("End time must exceed start time.")
    cases = load_cases(a.folder)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fmc_residuals(cases, a.out_dir, a.start_us, a.end_us)
    if a.tfm:
        tfm_comparison(cases, a.out_dir, a.velocity, a.grid_mm, a.real_view, a.time_offset_us)


if __name__ == "__main__":
    main()
