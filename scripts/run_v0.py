#!/usr/bin/env python
"""Run the complete Version 0 chain once and save diagnostic plots.

    forward library {a_i, F(a_i)}  ->  y = F(a_true) + e  ->  p(a | y)
    -> a^(i) ~ p(a | y)  ->  Delta K^(i)  ->  RUL^(i)  ->  next-inspection interval

Without --library a toy 64-element 5 MHz FMC library (stand-in for BristolFE)
is built in memory. With --library, a BristolFE export is loaded and the truth
is one of its entries (--true-index), which is an inverse crime: use only to
check plumbing.

Examples
--------
    python scripts/run_v0.py
    python scripts/run_v0.py --snr-db -16 --alpha 1e-3 --delta-sigma-mpa 120
    python scripts/run_v0.py --library data/bristolfe/lib.mat --true-index 15 --sigma 1e-12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rul_pipeline.crack_growth import ParisModel  # noqa: E402
from rul_pipeline.inspection import select_inspection_interval  # noqa: E402
from rul_pipeline.inversion import invert_grid, sample_posterior  # noqa: E402
from rul_pipeline.io import load_forward_library  # noqa: E402
from rul_pipeline.prognosis import probability_of_failure, propagate_rul  # noqa: E402
from rul_pipeline.structural import critical_crack_size  # noqa: E402
from rul_pipeline.synthetic import (  # noqa: E402
    add_measurement_noise,
    measurement_from_library,
    toy_fmc_library,
    toy_fmc_truth,
)

MM = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("forward library")
    g.add_argument("--library", type=Path, help=".mat/.npz BristolFE export (default: toy library)")
    g.add_argument("--dims", nargs="+", help="axis names of stored responses, if not in the file")
    g.add_argument("--crack-axis", type=int, default=0)
    g.add_argument("--crack-size-unit", default=None)
    g.add_argument("--n-elements", type=int, default=64, help="toy array elements")
    g.add_argument("--grid-mm", type=float, nargs=3, default=(0.5, 5.0, 0.1), metavar=("MIN", "MAX", "STEP"))

    g = p.add_argument_group("synthetic measurement")
    g.add_argument("--a-true-mm", type=float, default=2.03, help="toy truth (off-grid allowed)")
    g.add_argument("--true-index", type=int, help="library entry used as truth (file libraries)")
    noise = g.add_mutually_exclusive_group()
    noise.add_argument("--snr-db", type=float, default=-19.0, help="peak SNR in dB (default -19, see README)")
    noise.add_argument("--sigma", type=float, help="absolute noise sigma (same unit as responses)")
    g.add_argument("--noise", choices=("white", "bandlimited"), default="white")

    g = p.add_argument_group("structure, growth and inspection (SI inputs given in mm / MPa)")
    g.add_argument("--delta-sigma-mpa", type=float, default=100.0, help="constant-amplitude stress range")
    g.add_argument("--sigma-max-mpa", type=float, default=200.0, help="peak stress for fracture a_c")
    g.add_argument("--K-Ic-mpa-sqrt-m", type=float, default=60.0)
    g.add_argument("--Y", type=float, default=1.12)
    g.add_argument("--wall-limit-mm", type=float, default=10.0, help="a_c = min(fracture, this limit)")
    g.add_argument("--alpha", type=float, default=1e-2, help="acceptable risk for the next inspection")
    g.add_argument("--a-limit-mm", type=float, help="inspection limit size (default a_c)")

    p.add_argument("--n-samples", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    noise_kw = {"sigma": args.sigma} if args.sigma is not None else {"snr_db": args.snr_db}
    noise_kw["colour"] = args.noise

    # 1. Forward library ---------------------------------------------------------------
    if args.library:
        lib = load_forward_library(args.library, dims=args.dims, crack_axis=args.crack_axis,
                                   crack_size_unit=args.crack_size_unit)
    else:
        lo, hi, step = args.grid_mm
        grid = np.round(np.arange(lo, hi + 0.5 * step, step), 9) * MM
        lib = toy_fmc_library(grid, n_elements=args.n_elements)
    print(f"[1] Forward library: {lib.n_candidates} crack sizes "
          f"{lib.crack_sizes[0] / MM:.3g}-{lib.crack_sizes[-1] / MM:.3g} mm, "
          f"response shape {lib.measurement_shape} {lib.measurement_dims} (source: {lib.source})")

    # 2-3. Synthetic truth and noisy measurement ----------------------------------------
    if args.library or args.true_index is not None:
        idx = args.true_index if args.true_index is not None else lib.n_candidates // 3
        m = measurement_from_library(lib, idx, rng=rng, **noise_kw)
        y, y_clean, a_true, sigma = m.y, m.y_clean, m.a_true, m.sigma
        print("    WARNING: truth is a library entry (inverse crime) - plumbing check only.")
    else:
        a_true = args.a_true_mm * MM
        y_clean = toy_fmc_truth(a_true, lib)
        if args.noise == "bandlimited":
            noise_kw.update(fs=lib.fs, time_axis=-1)
        y, sigma = add_measurement_noise(y_clean, rng=rng, **noise_kw)
    print(f"[2] True crack size a_true = {a_true / MM:.4g} mm")
    print(f"[3] Added {args.noise} Gaussian noise, sigma = {sigma:.4g} "
          f"(peak |F(a_true)| = {np.abs(y_clean).max():.4g}, M = {y.size} samples)")

    # 4-5. Grid Bayesian inversion -> p(a | y) ------------------------------------------
    post = invert_grid(y, lib, sigma)
    print(f"[4-5] {post.summary()}")

    # 6. Posterior samples (jittered within grid cells) ---------------------------------
    a_samples = sample_posterior(post, args.n_samples, rng=rng, jitter=True)
    print(f"[6] Drew {a_samples.size} samples a^(i) ~ p(a | y); sample mean {a_samples.mean() / MM:.4g} mm")

    # 7. Structural map K_I ----------------------------------------------------------------
    model = ParisModel(delta_sigma=args.delta_sigma_mpa * 1e6, Y=args.Y)
    a_c_fracture = critical_crack_size(args.K_Ic_mpa_sqrt_m * 1e6, args.sigma_max_mpa * 1e6, args.Y)
    a_c = min(a_c_fracture, args.wall_limit_mm * MM)
    rul = propagate_rul(a_samples, model, a_c)
    dK = rul.delta_K_samples / 1e6
    print(f"[7] Delta K^(i) = Y Delta_sigma sqrt(pi a^(i)): median {np.median(dK):.4g} MPa sqrt(m), "
          f"90% range [{np.quantile(dK, 0.05):.4g}, {np.quantile(dK, 0.95):.4g}]")

    # 8-9. Paris RUL ---------------------------------------------------------------------
    print(f"[8] Paris: C = {model.C:.4g} m/cycle/(Pa sqrt(m))^{model.m:g}, m = {model.m:g}, "
          f"Delta_sigma = {args.delta_sigma_mpa:g} MPa, a_c = {a_c / MM:.4g} mm "
          f"(fracture limit {a_c_fracture / MM:.4g} mm, wall limit {args.wall_limit_mm:g} mm)")
    print(f"[9] {rul.summary()}")
    print(f"    Plug-in RUL(a_MAP) = {model.rul(post.map, a_c):.4g} cycles (for comparison only; not used)")

    # 10. Next inspection interval -------------------------------------------------------
    a_lim = (args.a_limit_mm * MM) if args.a_limit_mm else a_c
    sched = select_inspection_interval(a_samples, model, a_lim, args.alpha)
    print(f"[10] {sched.summary()}")

    # 11. Plots --------------------------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = make_plots(lib, y, y_clean, a_true, post, rul, sched, args.outdir)
    print(f"[11] Saved plots: {', '.join(str(p) for p in paths)}")


def make_plots(lib, y, y_clean, a_true, post, rul, sched, outdir: Path) -> list[Path]:
    a_mm = lib.crack_sizes / MM
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = outdir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(a_mm, post.log_likelihood - post.log_likelihood.max(), marker=".")
    ax.axvline(a_true / MM, linestyle="--", color="k", label="a_true")
    ax.set(xlabel="crack size a [mm]", ylabel="log p(y | a) - max", title="Log-likelihood over the library grid")
    ax.legend()
    save(fig, "01_log_likelihood.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(a_mm, post.posterior, width=0.8 * np.min(np.diff(a_mm)), label="p(a | y)")
    ax.axvline(a_true / MM, linestyle="--", color="k", label="a_true")
    ax.axvspan(*(np.array(post.credible_interval) / MM), alpha=0.2, label=f"{post.credible_level:.0%} CI")
    ax.set(xlabel="crack size a [mm]", ylabel="posterior mass", title="Posterior p(a | y)")
    ax.legend()
    save(fig, "02_posterior.png")

    if lib.measurement_dims == ("tx", "rx", "time") and lib.time is not None:
        e = lib.measurement_shape[0] // 2
        t_us = lib.time * 1e6
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t_us, y[e, e], label="measured y", linewidth=0.8)
        ax.plot(t_us, y_clean[e, e], label="clean F(a_true)")
        ax.plot(t_us, lib.responses[post.map_index][e, e], linestyle="--", label="best fit F(a_MAP)")
        ax.set(xlabel="time [us]", ylabel="amplitude", title=f"Pulse-echo A-scan, element {e + 1} (inversion uses all {y.size} samples)")
        ax.legend()
        save(fig, "03_measured_vs_best_fit.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rul.rul_samples, bins=60, density=True)
    for q in (*rul.credible_interval, rul.median):
        ax.axvline(q, linestyle="--" if q != rul.median else "-", color="k")
    ax.set(xlabel="remaining useful life [cycles]", ylabel="density",
           title=f"p(RUL | y): median and {rul.credible_level:.0%} CI")
    save(fig, "04_rul_histogram.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sched.candidate_intervals, sched.risk, label="risk(Delta N)")
    ax.axhline(sched.alpha, linestyle="--", color="k", label=f"alpha = {sched.alpha:g}")
    ax.axvline(sched.interval, linestyle=":", color="k", label=f"selected Delta N = {sched.interval:.3g}")
    ax.plot(sched.candidate_intervals, probability_of_failure(rul.rul_samples, sched.candidate_intervals),
            linestyle="--", label="P(RUL <= Delta N) to a_c (failure-time check)")
    ax.set(xlabel="inspection interval Delta N [cycles]", ylabel="probability", yscale="log",
           ylim=(max(1.0 / rul.rul_samples.size, 1e-5), 1.0),
           title=f"Risk vs interval (Delta_sigma = {sched.delta_sigma / 1e6:g} MPa assumed)")
    ax.legend()
    save(fig, "05_risk_vs_interval.png")
    return paths


if __name__ == "__main__":
    main()
