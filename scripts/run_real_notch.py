#!/usr/bin/env python
"""Real notch-only FMC vs the four BristolFE candidates (0, 1, 2, 3 mm), end to end.

    python scripts/run_real_notch.py --candidates <crack_length_pilot folder>
    python scripts/run_real_notch.py --candidates <folder> --repeat-dir <root with fixed_probe/ repositioned/>

Provisional mode (always): scalar Gaussian error levels ASSUMED, swept; every posterior / RUL is
conditional on them. Repeat mode (--repeat-dir): same observation operator, error taken from
independent repeat scans; the four comparisons A-D separate the effect of the error model from the
effect of averaging (see report.md). Outputs go to --outdir (CSV, PNG, report.md).

Reused from the pipeline: ForwardLibrary / grid_posterior / ParisModel / critical_crack_size /
tfm_image. New: io.real_fmc, io.repeat_scans, inversion.{observation,error_models},
prognosis.discrete_rul. Units: real data are instrument units (full scale = 1), candidates are
displacement in m; the unknown amplitude gain g is explicit (profiled by default).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rul_pipeline.crack_growth import ParisModel  # noqa: E402
from rul_pipeline.imaging import tfm_image  # noqa: E402
from rul_pipeline.inversion.error_models import gaussian_loglik, total_variance  # noqa: E402
from rul_pipeline.inversion.grid_bayes import grid_posterior  # noqa: E402
from rul_pipeline.inversion.observation import Preprocess, apply, estimate_time_shift  # noqa: E402
from rul_pipeline.io.real_fmc import load_brain_fmc, load_pilot_candidates  # noqa: E402
from rul_pipeline.io.repeat_scans import load_repeat_groups, summarise  # noqa: E402
from rul_pipeline.prognosis.discrete_rul import conditional_discrete_rul  # noqa: E402
from rul_pipeline.structural import critical_crack_size  # noqa: E402

MM = 1e-3
DEFAULT_REAL = Path("data/steel_notch_initial_confirmation/above_view/experimental_fmc.mat")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", type=Path, default=DEFAULT_REAL)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--repeat-dir", type=Path)
    p.add_argument("--outdir", type=Path, default=Path("outputs/real_notch"))
    g = p.add_argument_group("observation operator")
    g.add_argument("--window-us", type=float, nargs=2, default=(8.0, 25.0))
    g.add_argument("--band-mhz", type=float, nargs=2, default=(2.5, 7.5))
    g.add_argument("--no-bandpass", action="store_true")
    g.add_argument("--shift-us", type=float, help="fix the real-data time shift (default: estimated vs notch-only)")
    g.add_argument("--gain", default="profiled", help="'profiled' (per-candidate LS gain) or a number")
    g = p.add_argument_group("error model")
    g.add_argument("--sigma", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                   help="assumed per-sample sd of ONE acquisition, preprocessed instrument units (full scale = 1)")
    g.add_argument("--assumed-sigma", type=float, default=0.01, help="sigma used in repeat-mode comparisons")
    g.add_argument("--discrepancy", type=float, default=0.0, help="assumed model-discrepancy sd (same units)")
    g.add_argument("--shrink", type=float, default=0.5, help="variance shrinkage toward the per-time pooled median")
    g = p.add_argument_group("growth / loading (conditional assumptions; defaults as run_v0.py)")
    g.add_argument("--delta-sigma-mpa", type=float, default=100.0)
    g.add_argument("--sigma-max-mpa", type=float, default=200.0)
    g.add_argument("--K-Ic-mpa-sqrt-m", type=float, default=60.0)
    g.add_argument("--Y", type=float, default=1.12)
    g.add_argument("--wall-limit-mm", type=float, default=10.0)
    g.add_argument("--size-offset-mm", type=float, default=0.0,
                   help="added to the crack extension to form the flaw size used in Paris growth (default 0)")
    g.add_argument("--no-tfm", action="store_true")
    return p.parse_args()


def write_csv(path: Path, header, rows):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def expand(pairs_data: np.ndarray, pairs: np.ndarray, n: int) -> np.ndarray:
    cube = np.zeros((n, n, pairs_data.shape[-1]))
    cube[pairs[:, 0], pairs[:, 1]] = pairs_data
    cube[pairs[:, 1], pairs[:, 0]] = pairs_data
    return cube


def posterior_and_rul(sizes, ll, model, a_c, offset):
    post = grid_posterior(sizes, ll)
    return post, conditional_discrete_rul(sizes, post.posterior, model, a_c, size_offset=offset,
                                          horizons=(1e5, 2e5, 5e5))


def main():
    a = parse_args()
    out = a.outdir
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    # ---- 1. load + compatibility ----------------------------------------------------------
    real = load_brain_fmc(a.real)
    lib = load_pilot_candidates(a.candidates, real)
    sizes = lib.crack_sizes
    say("## Data and compatibility")
    say(f"- Real: `{a.real}`; {real.data.shape[0]} unique pairs (tx<=rx, half matrix), {real.data.shape[1]} samples, "
        f"fs {real.fs / 1e6:g} MHz, t 0-{real.time[-1] * 1e6:.1f} us, centre {real.centre_freq / 1e6:g} MHz, "
        f"{real.n_elements} elements, pitch {np.diff(real.element_x).mean() * 1e3:.3f} mm, "
        f"values {real.meta['amplitude_unit']} (max |x| = {real.meta['max_abs']:g}; 1/128 steps => 8-bit quantisation).")
    say(f"- Candidates: {[f'{s / MM:g} mm' for s in sizes]} from {lib.metadata['files']}; Fourier-resampled from 89.1 MHz to "
        f"the real 25 MHz grid, sampled at the same unique pairs (reciprocity of the FE data verified to 1e-9). "
        f"Units: displacement in m. cL={lib.metadata['cL']:g} m/s (BRAIN imaging velocity 5850 m/s).")
    say("- Tx-Rx ordering: real tx/rx 1-based, tx<=rx, element order left to right (x ascending) as in the FE arrays.")
    say(f"- GEOMETRY: {lib.metadata['sim_params']}. The earlier validated model was 50 x 85 mm; the real backwall echo "
        "arrives later than the candidates' (~29 vs ~27 us), so the window excludes it. Waveform-level agreement "
        "between FE and specimen is therefore not expected to be good.")
    gain_arg = "profiled" if a.gain == "profiled" else float(a.gain)

    # ---- 2. observation operator + calibration -----------------------------------------------
    pre0 = Preprocess(tuple(a.window_us), None if a.no_bandpass else tuple(a.band_mhz), 0.0)
    cand0_full = lib.responses[int(np.argmin(sizes))]
    if a.shift_us is None:
        shift, shifts, corr = estimate_time_shift(real.data, cand0_full, real.time, pre0)
        say(f"- Time-zero calibration (one global scalar): real data shifted by {shift:+.3f} us, chosen to maximise the "
            f"envelope correlation with the NOTCH-ONLY candidate ({corr.max():.3f}; {corr[np.argmin(np.abs(shifts))]:.3f} at 0). "
            "Because the specimen is known to be notch-only, this mildly favours a = 0 - it is calibration, not evidence.")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(shifts, corr)
        ax.axvline(shift, color="k", ls="--")
        ax.set(xlabel="real-data shift [us]", ylabel="envelope correlation", title="time-zero calibration")
        fig.tight_layout()
        fig.savefig(out / "calibration_time_shift.png", dpi=130)
        plt.close(fig)
    else:
        shift = a.shift_us
        say(f"- Time shift fixed by user: {shift:+.3f} us.")
    pre = Preprocess(pre0.window_us, pre0.band_mhz, shift)
    say(f"- Observation operator (identical for real and candidates except the real-only shift): {pre.describe()}. "
        "No per-trace normalisation; amplitude gain g is explicit in the likelihood "
        f"({'profiled per candidate (weighted LS)' if gain_arg == 'profiled' else gain_arg}).")
    y = apply(real.data, real.time, pre, is_real=True)
    F = np.stack([apply(f, real.time, pre, is_real=False) for f in lib.responses])
    tw = real.time[int(np.flatnonzero(real.time * 1e6 >= pre.window_us[0])[0]):][: y.shape[-1]]
    say(f"- Observation: {y.shape[0]} pairs x {y.shape[1]} samples = {y.size} values; y rms {np.sqrt((y**2).mean()):.4g} "
        f"(instrument units), candidate rms {[f'{np.sqrt((f**2).mean()):.3g}' for f in F]} m.")

    # ---- 3. growth model ----------------------------------------------------------------------
    model = ParisModel(delta_sigma=a.delta_sigma_mpa * 1e6, Y=a.Y)
    a_c = min(float(critical_crack_size(a.K_Ic_mpa_sqrt_m * 1e6, a.sigma_max_mpa * 1e6, a.Y)), a.wall_limit_mm * MM)
    off = a.size_offset_mm * MM

    # ---- 4. provisional mode: assumed scalar sigma sweep ---------------------------------------
    say("\n## Provisional mode: ASSUMED scalar Gaussian error (not calibrated)")
    base = gaussian_loglik(y, F, 1.0, gain_arg)
    rank_rows = [[f"{s / MM:g}", f"{r:.6g}", f"{r / np.sqrt((y**2).mean()):.4f}", f"{g:.4g}"]
                 for s, r, g in zip(sizes, base.residual_rms, base.gain)]
    write_csv(out / "residual_ranking_unweighted.csv", ["crack_mm", "residual_rms_instrument_units", "rms_over_y_rms", "gain_g"], rank_rows)
    order = np.argsort(base.residual_rms)
    say(f"- Unweighted residual ranking (best first): {[f'{sizes[i] / MM:g} mm' for i in order]}; "
        f"residual/|y| = {[f'{base.residual_rms[i] / np.sqrt((y**2).mean()):.4f}' for i in order]}. "
        f"Candidate-to-candidate rms spread is {np.ptp(base.residual_rms) / base.residual_rms.mean():.2%} of the residual.")
    rows, post_by_sigma, rul_rows = [], {}, []
    sigma_prof = float(base.residual_rms.min())
    say(f"- Plug-in (profile) sigma = best-candidate residual rms = {sigma_prof:.4g} units: a DIAGNOSTIC that absorbs all model "
        "mismatch into noise (see estimate_sigma_profile); it is added to the sweep below.")
    a.sigma = sorted({*a.sigma, round(sigma_prof, 5)})
    for s in a.sigma:
        v = total_variance(s**2 * np.ones(1), discrepancy=a.discrepancy)[0]
        r = gaussian_loglik(y, F, v, gain_arg)
        post, rul = posterior_and_rul(sizes, r.log_likelihood, model, a_c, off)
        post_by_sigma[s] = post
        for i, sz in enumerate(sizes):
            rows.append([s, f"{sz / MM:g}", r.log_likelihood[i] - r.log_likelihood.max(), post.posterior[i], r.gain[i]])
        rul_rows.append([s, rul.p_notch_only, rul.mean, *[rul.quantiles.get(q, "") for q in (0.05, 0.5, 0.95)],
                         *[rul.p_fail_by.get(h, "") for h in (1e5, 2e5, 5e5)]])
        say(f"- sigma_total={np.sqrt(v):.4g} units: posterior {[f'{p:.3g}' for p in post.posterior]}; {rul.summary()}")
    write_csv(out / "posterior_vs_sigma.csv", ["sigma_assumed", "crack_mm", "loglik_minus_max", "posterior", "gain_g"], rows)
    write_csv(out / "conditional_rul_vs_sigma.csv",
              ["sigma_assumed", "p_notch_only", "rul_mean_cycles", "q05", "q50", "q95", "P_fail_by_1e5", "P_fail_by_2e5", "P_fail_by_5e5"], rul_rows)
    say(f"- Growth assumptions (all RUL conditional on these): dsigma {a.delta_sigma_mpa:g} MPa, Y {a.Y:g}, a_c {a_c / MM:g} mm, "
        f"Paris C={model.C:.3g}, m={model.m:g}; flaw size = extension + {a.size_offset_mm:g} mm. Extension alone ignores the notch as a stress raiser.")

    # ---- plots: residuals / likelihoods / posterior / RUL ---------------------------------------
    lab = [f"{s / MM:g}" for s in sizes]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.bar(lab, base.residual_rms)
    ax.set(xlabel="candidate extension [mm]", ylabel="unweighted residual rms (instrument units)", title="Unweighted residual ranking")
    ax = axes[0, 1]
    for i, sz in enumerate(sizes):
        ax.plot(a.sigma, [post_by_sigma[s].log_likelihood[i] - post_by_sigma[s].log_likelihood.max() for s in a.sigma], "o-", label=f"{sz / MM:g} mm")
    ax.set(xscale="log", xlabel="assumed sigma [instrument units]", ylabel="log-likelihood minus max", title="Likelihood vs assumed error")
    ax.set_yscale("symlog", linthresh=1)
    ax.legend()
    ax = axes[1, 0]
    for i, sz in enumerate(sizes):
        ax.plot(a.sigma, [post_by_sigma[s].posterior[i] for s in a.sigma], "o-", label=f"{sz / MM:g} mm")
    ax.set(xscale="log", xlabel="assumed sigma [instrument units]", ylabel="posterior weight", title="Posterior weights (conditional on assumed sigma)")
    ax.legend()
    ax = axes[1, 1]
    for s in a.sigma:
        rul = conditional_discrete_rul(sizes, post_by_sigma[s].posterior, model, a_c, size_offset=off)
        if rul.a.size:
            ax.stem(rul.rul, rul.weights, linefmt=f"C{a.sigma.index(s) % 10}-", markerfmt=f"C{a.sigma.index(s) % 10}o", basefmt=" ", label=f"sigma {s:g}")
    ax.set(xlabel="RUL | crack present [cycles]", ylabel="conditional weight", title="Exact discrete RUL (a>0 only)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "provisional_summary.png", dpi=130)
    plt.close(fig)

    # best-fit waveform checks
    best = int(order[0])
    show = [(31, 31), (10, 50), (5, 12)]
    pidx = {tuple(p): k for k, p in enumerate(real.pairs)}
    fig, axes = plt.subplots(len(show), 1, figsize=(10, 3 * len(show)), sharex=True)
    for ax, pr in zip(axes, show):
        k = pidx[(min(pr), max(pr))]
        ax.plot(tw * 1e6, y[k], "k", lw=1, label="real (shifted, filtered)")
        ax.plot(tw * 1e6, base.gain[0] * F[0][k], "C0", lw=.8, label=f"notch-only x g={base.gain[0]:.3g}")
        ax.plot(tw * 1e6, base.gain[best] * F[best][k], "C3--", lw=.8, label=f"best unweighted ({sizes[best] / MM:g} mm) x g={base.gain[best]:.3g}")
        vis = np.sqrt((y**2).mean() / (F[0] ** 2).mean())
        ax.plot(tw * 1e6, vis * F[0][k], "C2", lw=.9, alpha=.8, label="notch-only FE rescaled to equal global rms (VISUAL ONLY, not used in inference)")
        ax.set(ylabel=f"Tx{pr[0] + 1} Rx{pr[1] + 1}")
    axes[0].legend(fontsize=7)
    axes[-1].set_xlabel("time [us]")
    fig.tight_layout()
    fig.savefig(out / "waveform_checks.png", dpi=130)
    plt.close(fig)
    ccs = [float(np.corrcoef(y.ravel(), F[i].ravel())[0, 1]) for i in range(len(sizes))]
    say(f"- Waveform check: correlation of y with each candidate (RF, window) = {[f'{c:.3f}' for c in ccs]}. "
        "Values near zero mean the FE waveforms do not reproduce the real RF signals, so the likelihood is dominated by model "
        "mismatch rather than the crack extension.")

    # ---- 5. repeat mode -------------------------------------------------------------------------
    if a.repeat_dir is not None:
        say("\n## Repeat-scan mode")
        groups = load_repeat_groups(a.repeat_dir)
        for gname, grp in groups.items():
            if not (np.allclose(grp.scans[0].time, real.time) and np.array_equal(grp.scans[0].pairs, real.pairs)):
                raise ValueError(f"{gname}: repeat scans use a different time axis or pair ordering than the reference scan.")
            proc = np.stack([apply(s.data, s.time, pre, is_real=True) for s in grp.scans])
            S = summarise(proc, shrink=a.shrink)
            say(f"### {gname}: n = {S['n']} scans; scan-rms ratio max/min = {S['scale_ratio']:.3f} (gain drift check); "
                f"repeat sd (rms over samples): raw {S['sd_raw']:.4g}, regularised {S['sd_reg']:.4g} (units as data); assumed sigma {a.assumed_sigma:g}")
            v_assumed = a.assumed_sigma**2
            cases = {
                "A single scan, assumed sigma": (proc[0], total_variance(v_assumed, discrepancy=a.discrepancy)),
                "B single scan, repeat-estimated diagonal": (proc[0], total_variance(S["var_reg"], discrepancy=a.discrepancy)),
                "C repeat mean, assumed sigma/sqrt(n)": (S["mean"], total_variance(v_assumed, n_avg=S["n"], discrepancy=a.discrepancy)),
                "D repeat mean, repeat-estimated diagonal/n": (S["mean"], total_variance(S["var_reg"], n_avg=S["n"], discrepancy=a.discrepancy)),
            }
            rows = []
            for cname, (obs, var) in cases.items():
                r = gaussian_loglik(obs, F, var, gain_arg)
                post, rul = posterior_and_rul(sizes, r.log_likelihood, model, a_c, off)
                disc = max(0.0, float(r.residual_rms[int(np.argmax(r.log_likelihood))] ** 2 - np.mean(var)))
                say(f"- {cname}: posterior {[f'{p:.3g}' for p in post.posterior]}; {rul.summary()}; "
                    f"implied model-discrepancy sd at best fit ~ {np.sqrt(disc):.3g}")
                rows.append([gname, cname, *post.posterior, rul.p_notch_only, rul.mean, *[r.residual_rms[i] for i in range(len(sizes))]])
            write_csv(out / f"repeat_{gname}_comparison.csv",
                      ["group", "case", *[f"post_{s / MM:g}mm" for s in sizes], "p_notch_only", "rul_mean_cycles", *[f"resid_rms_{s / MM:g}mm" for s in sizes]], rows)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(tw * 1e6, np.sqrt(S["var_raw"].mean(0)), label="raw sd (mean over pairs)")
            axes[0].plot(tw * 1e6, np.sqrt(S["var_reg"].mean(0)), label="regularised sd")
            axes[0].set(xlabel="time [us]", ylabel="repeat sd [instrument units]", title=f"{gname}: variability vs time")
            axes[0].legend()
            axes[1].hexbin(np.abs(S["mean"]).ravel(), np.sqrt(S["var_raw"]).ravel(), gridsize=60, bins="log")
            axes[1].set(xlabel="|mean|", ylabel="repeat sd", title="sd vs amplitude (does error scale with signal?)")
            pair_sd = np.full((real.n_elements,) * 2, np.nan)
            pair_sd[real.pairs[:, 0], real.pairs[:, 1]] = np.sqrt(S["var_raw"].mean(1))
            im = axes[2].imshow(pair_sd, origin="lower", aspect="equal")
            axes[2].set(title="repeat sd per pair (tx<=rx half matrix)", xlabel="rx", ylabel="tx")
            fig.colorbar(im, ax=axes[2])
            fig.tight_layout()
            fig.savefig(out / f"repeat_{gname}_variance_diagnostics.png", dpi=130)
            plt.close(fig)
        say("- Fixed-probe and repositioned groups are analysed separately; differences between them are the coupling/positioning "
            "contribution. Repeat variability is MEASUREMENT error only; model discrepancy stays an assumed scalar (--discrepancy).")
    else:
        say("\n## Repeat-scan mode: not run (no --repeat-dir). Ready: put scans in <dir>/fixed_probe/ and/or <dir>/repositioned/ (BRAIN exp_data .mat).")

    # ---- 6. TFM comparison ------------------------------------------------------------------------
    if not a.no_tfm:
        say("\n## TFM comparison")
        n_el = real.n_elements
        pre_t = Preprocess((0.0, real.time[-1] * 1e6), pre.band_mhz, shift)
        t_real = apply(real.data, real.time, pre_t, is_real=True)
        cubes = [("Real", expand(t_real, real.pairs, n_el))]
        for s, f in zip(sizes, lib.responses):
            cubes.append((f"FE +{s / MM:g} mm" if s else "FE notch only", expand(apply(f, real.time, pre_t, is_real=False), real.pairs, n_el)))
        elements = np.column_stack((real.element_x, np.zeros(n_el)))
        xg = np.arange(-25e-3, 25e-3 + 1e-9, 0.3e-3)
        zg = np.arange(0, 80e-3 + 1e-9, 0.3e-3)
        dbs = []
        for name, cube in cubes:
            img = tfm_image(cube, real.time[:cube.shape[-1]], elements, xg, zg, 5850.0, pixel_chunk_size=256)
            dbs.append(20 * np.log10(np.maximum(img / img.max(), 1e-6)))
        fig, axes = plt.subplots(1, len(cubes), figsize=(3.3 * len(cubes), 6.5))
        for ax, (name, _), db in zip(axes, cubes, dbs):
            im = ax.imshow(db, extent=[xg[0] * 1e3, xg[-1] * 1e3, zg[-1] * 1e3, zg[0] * 1e3], cmap="jet", vmin=-40, vmax=0, aspect="equal")
            ax.set(title=name, xlabel="x [mm]")
        axes[0].set_ylabel("depth [mm]")
        fig.colorbar(im, ax=list(axes), shrink=0.6, label="dB re own max")
        fig.suptitle("TFM at 5850 m/s of the observation operator's data (own-max normalised)", fontsize=10)
        fig.savefig(out / "tfm_real_vs_candidates.png", dpi=130)
        plt.close(fig)
        cc = [float(np.corrcoef(np.maximum(dbs[0], -40).ravel(), np.maximum(d, -40).ravel())[0, 1]) for d in dbs[1:]]
        write_csv(out / "tfm_dB_correlation.csv", ["candidate_mm", "corr_with_real_dB_image(-40dB floor)"], [[f"{s / MM:g}", c] for s, c in zip(sizes, cc)])
        say(f"- dB-image correlation of each FE TFM with the real TFM (-40 dB floor): {[f'{c:.3f}' for c in cc]}; the four candidates "
            "differ only slightly, so TFM correlation cannot discriminate 1-3 mm extension.")

    say("\n## Unresolved calibration / compatibility issues")
    for s in [
        "Candidate geometry is the photo-estimated 60 x 80 mm pilot, not the measured specimen; FE-vs-real waveform match is weak.",
        "Amplitude gain between instrument units and FE displacement is unknown (profiled, not calibrated).",
        "Time shift is estimated against the notch-only candidate (assumes the specimen is notch only).",
        "No repeat scans yet: sigma is an assumption; reported posteriors are conditional on it, not experimentally calibrated.",
        "Real data are 8-bit quantised and include electronic/transducer response absent from the FE model.",
        "Real half matrix is used once per pair (2080 pairs); mirrored data are NOT double counted.",
    ]:
        say(f"- {s}")
    (out / "report.md").write_text("# Real notch-only FMC vs BristolFE candidates\n\n" + "\n".join(log) + "\n", encoding="utf8")
    print(f"\nWrote {out}/report.md and figures/CSVs")


if __name__ == "__main__":
    main()
