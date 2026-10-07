# rul-pipeline-v0

**Probabilistic ultrasonic NDT → crack-size posterior → remaining useful life → next inspection.**
This is the Python side of a PhD pipeline. Forward ultrasonic responses come from MATLAB / [BristolFE](https://github.com/ndtatbristol/BristolFE-v2); Python handles inversion, uncertainty propagation, crack growth, prognosis and (later) sequential Bayesian updating.

## Version 0 objective

> Build the **simplest complete probabilistic chain** first, validate every stage against hand calculations and analytic limits, and only then replace individual stages with more advanced methods.

Each stage is deliberately elementary: a 1‑D grid posterior, a closed-form $K_I$, constant-amplitude Paris growth, and Monte Carlo sampling. What V0 does insist on is that **uncertainty is handed between stages as a probability distribution**, never as a point estimate.

```
BristolFE forward library {a_i, F(a_i)}
        │
        ▼
Bayesian grid inversion          y  →  p(a | y)
        │
        ▼  a^(i) ~ p(a | y)
Structural map                   K_I^(i) = Y σ √(π a^(i))
        │
        ▼
Paris law                        RUL^(i) = H(a^(i); C, m, Δσ, a_c)
        │
        ▼
p(RUL | y)                       median, CI, P(failure before horizon)
        │
        ▼
Adaptive inspection interval     max ΔN  s.t.  P(a(N_k+ΔN) ≥ a_lim | y_1:k) ≤ α
        │
        ▼  (scaffold)
Sequential Bayesian update       p(a_k+1 | y_1:k+1) ∝ p(y_k+1 | a_k+1) p(a_k+1 | y_1:k)
```

## Quick start

```bash
pip install -e ".[dev]"          # numpy, scipy, matplotlib, h5py (+ pytest)
pytest                           # 91 tests, ~4 s
python scripts/run_v0.py         # full chain on a toy 64-element 5 MHz FMC library; plots -> outputs/
```

Example output of `scripts/run_v0.py` (default settings, seed 0):

```
[1] Forward library: 46 crack sizes 0.5-5 mm, response shape (64, 64, 157) ('tx', 'rx', 'time')
[2] True crack size a_true = 2.03 mm
[3] Added white Gaussian noise, sigma = 7.211 (peak |F(a_true)| = 0.8091, M = 643072 samples)
[4-5] p(a | y): MAP = 2 mm, mean = 2.005 mm, std = 0.125 mm, 95% CI = [1.4, 2.1] mm
[7] Delta K^(i) = Y Delta_sigma sqrt(pi a^(i)): median 8.914 MPa sqrt(m), 90% range [8.772, 9.164]
[8] Paris: C = 1.648e-29 m/cycle/(Pa sqrt(m))^3, m = 3, Delta_sigma = 100 MPa, a_c = 10 mm
[9] p(RUL | y): median = 1.904e+05, mean = 1.919e+05, std = 1.34e+04 cycles, 90% CI = [1.81e+05, 1.96e+05] cycles
[10] Next inspection in Delta N = 1.794e+05 cycles: largest interval with P(a >= a_lim) <= 0.01, assuming Delta_sigma = 100 MPa
```

The notebook [`notebooks/01_forward_library_and_inversion.ipynb`](notebooks/01_forward_library_and_inversion.ipynb) (committed with outputs) walks through the forward library, posterior vs SNR, cycle-skipping and the sample-based handover.

## Project structure

```
src/rul_pipeline/
    io/bristolfe.py              ForwardLibrary container, .mat (v5 + v7.3) / .npz loaders, shape checks   → io/README.md
    synthetic/                   toy FMC forward model (BristolFE stand-in) + noisy measurements          → synthetic/README.md
    inversion/grid_bayes.py      log-space Gaussian likelihood, grid posterior, summaries, sampling        → inversion/README.md
    structural/mode_i.py         K_I = Y σ √(πa), critical crack size                                       → structural/README.md
    crack_growth/paris.py        Paris law: closed form (m≠2, m=2), quadrature, ODE stepping, unit conversion → crack_growth/README.md
    prognosis/monte_carlo.py     a^(i) → RUL^(i), summaries, P(failure ≤ horizon)                           → prognosis/README.md
    inspection/scheduling.py     risk curve vs ΔN and largest interval with risk ≤ α                         → inspection/README.md
    update/bayes_update.py       grid predict (growth) + update (new likelihood) scaffold                   → update/README.md
    utils/validation.py          input checks, log-sum-exp normalisation
tests/                           one file per stage + an end-to-end smoke test
scripts/run_v0.py                the whole chain, printed summaries, 5 plots
matlab/export_forward_library.m  proposed MATLAB export of a BristolFE crack-size sweep
notebooks/                       executed walkthrough
data/bristolfe/, data/synthetic/ place exports here (contents git-ignored)
```

Each sub-package has its own README that catalogues exactly how that stage is implemented, with the equations, code excerpts, tests and references.

## Notation and units

SI is used everywhere inside the package. Inputs in mm or MPa are converted at the boundary (`load_forward_library(..., crack_size_unit="mm")`, `paris_C_to_SI`, the `*_mm` / `*_mpa` script flags).

| Symbol | Meaning | Unit |
|---|---|---|
| $y$ | measurement (FMC: tx × rx × time), flattened to length $M$ | as exported |
| $F(a)$ | forward model / library prediction | same as $y$ |
| $a$ | crack size (depth) | m |
| $p(a \mid y)$ | posterior crack-size distribution (discrete on the grid) | – |
| $\sigma$ | measurement-noise standard deviation | same as $y$ |
| $K_I,\ \Delta K$ | Mode-I stress intensity factor and its range | Pa√m |
| $Y$ | geometry factor | – |
| $\sigma_\text{max},\ \Delta\sigma$ | peak stress and stress range | Pa |
| $C,\ m$ | Paris-law coefficient and exponent | m/cycle/(Pa√m)$^m$, – |
| $a_c,\ a_\text{lim}$ | critical and inspection-limit crack sizes | m |
| RUL, $N_f$ | remaining cycles to $a_c$ | cycles |
| $\alpha$ | acceptable risk for the next-inspection decision | – |

## The probability handover (design principle)

```math
y \;\xrightarrow{\text{grid Bayes}}\; p(a\mid y)
\;\xrightarrow{\text{sample}}\; a^{(i)}
\;\xrightarrow{G}\; K_I^{(i)} = Y\sigma\sqrt{\pi a^{(i)}}
\;\xrightarrow{H}\; \mathrm{RUL}^{(i)} = H\!\left(a^{(i)}; C, m, \Delta\sigma\right)
```

The posterior is never collapsed to the MAP before crack-growth analysis. This matters for two reasons:

1. **Jensen bias.** $\mathrm{RUL}(a) \propto a^{1-m/2} - a_c^{1-m/2}$ is convex in $a$, so $\mathbb{E}[\mathrm{RUL}] \ge \mathrm{RUL}(\mathbb{E}[a])$, and plugging in a point estimate biases the RUL (tested in `tests/test_prognosis.py`).
2. **Multimodality.** Ultrasonic misfits have cycle-skipping minima (below). A secondary posterior mode in $a$ becomes a separate RUL mode, which a point estimate cannot represent.

## Findings from V0 on synthetic data

These are properties of the method, not bugs. They set the agenda for V1.

- **Over-confidence of the i.i.d. likelihood.** With $\Sigma = \sigma^2 I$ over $M \approx 6.4\times10^5$ FMC samples, every sample counts as independent information. At any realistic SNR (≥ 0 dB peak) the posterior collapses into a single 0.1 mm grid cell. The demo therefore uses an *effective* peak SNR of −19 dB as a stand-in for the unmodelled errors (correlated noise, model discrepancy). Ignoring model discrepancy produces biased and over-confident parameter estimates ([Brynjarsdóttir & O'Hagan, 2014](https://doi.org/10.1088/0266-5611/30/11/114007); [Kennedy & O'Hagan, 2001](https://doi.org/10.1111/1467-9868.00294)).
- **Cycle skipping ⇒ multimodality.** Moving the crack tip by $\Delta a$ shifts the pulse-echo arrival by $2\Delta a/c$. The misfit therefore has secondary minima near $\Delta a = c/(2f_c) \approx 0.59$ mm in steel at 5 MHz. In a seeded notebook case at −20 dB, the posterior locks onto the mode one cycle away and its 95 % CI excludes the truth. The exhaustive 1-D grid can represent this; local optimisers and Laplace approximations cannot.
- **Inverse crime.** Inverting data generated from a library entry flatters the method ([Kaipio & Somersalo, 2007](https://doi.org/10.1016/j.cam.2005.09.027)). The default truth (2.03 mm) is off-grid and generated separately. A fully honest test needs truth from a different (finer-mesh) BristolFE run.

## What Version 0 deliberately neglects

**Measurement**
- realistic correlated noise: band-limited electronic noise is available for generating data, but the likelihood assumes white noise; coherent grain noise is absent;
- coupling, positioning and calibration uncertainty (gain, wedge/couplant delay, probe offset);
- unknown or estimated $\sigma$ (it is treated as known; `estimate_sigma_profile` is only a diagnostic).

**Forward model**
- crack orientation (tilt), morphology (branching, roughness) and defect class (crack vs. void vs. inclusion);
- model discrepancy between BristolFE and reality, and FE discretisation error;
- material variability (velocity, attenuation, anisotropy).

**Inversion**
- high-dimensional parameter spaces (size + position + orientation + …), where grids are infeasible;
- systematic treatment of multimodality;
- simulation-based inference ([Cranmer et al., 2020](https://doi.org/10.1073/pnas.1912789117)), MCMC ([Hastings, 1970](https://doi.org/10.1093/biomet/57.1.97)), and manifold / reduced-order methods.

**Structural**
- irregular geometry and finite-width / a/W effects on $Y$;
- mixed-mode loading ($K_{II}$, $K_{III}$);
- FE-derived $K$ and $Y(a)$ (the code already accepts a callable $Y(a)$ in the numerical Paris integrator);
- residual stress, crack closure and plasticity.

**Crack growth**
- load-interaction / retardation models, e.g. Wheeler ([Wheeler, 1972](https://doi.org/10.1115/1.3425362));
- NASGRO / Forman ([Forman et al., 1967](https://doi.org/10.1115/1.3609637)) with $R$-ratio, threshold and $K_c$ asymptotes;
- variable-amplitude and spectrum loading;
- uncertain $C$ and $m$, including their strong correlation and specimen-to-specimen scatter ([Virkler et al., 1979](https://doi.org/10.1115/1.3443666));
- model-form uncertainty.

**Prognosis**
- Bayesian model averaging over growth laws;
- hierarchical Bayesian uncertainty (population → component);
- dynamic Bayesian networks ([Straub, 2009](https://doi.org/10.1061/(ASCE)EM.1943-7889.0000024)) and particle filtering ([Arulampalam et al., 2002](https://doi.org/10.1109/78.978374)).

**Inspection**
- value of information and full risk-based inspection planning ([Straub & Faber, 2005](https://doi.org/10.1016/j.strusafe.2005.04.001));
- Bayesian experimental design: where, how and with what to inspect ([Lindley, 1956](https://doi.org/10.1214/aoms/1177728069); [Chaloner & Verdinelli, 1995](https://doi.org/10.1214/ss/1177009939));
- optimisation of continuous monitoring vs. periodic inspection;
- probability of detection (PoD) and false-call modelling.

## BristolFE data interface: current assumptions

BristolFE v2 FMC results (e.g. `main.doms{1}.res.fmc` in [`subdomain_array_example.m`](https://github.com/ndtatbristol/BristolFE-v2/blob/main/examples/subdomain_array_example.m)) store `time` $(n_t\times1)$ and `time_data` $(n_t \times n_\text{pairs})$, with `tx`, `rx` element indices (1-based) per column. The loader assumes **no fixed schema**. Instead it:

- reads `.mat` v5/v7 (`scipy.io.loadmat`, never squeezed) and v7.3 (`h5py`, HDF5 dimension reversal undone), plus `.npz`;
- maps canonical names to whatever is in the file via `key_map` (dotted struct paths allowed, e.g. `"lib.fmc.time_data"`);
- requires every axis of `responses` to be **named** (`dims`), and moves the crack axis to the front only when the caller says so (`crack_axis=`);
- never transposes silently. A wrong axis order raises an error that suggests the fix.

The proposed export is [`matlab/export_forward_library.m`](matlab/export_forward_library.m): `crack_sizes` (N×1, m), `responses` (N × n_t × n_pairs), `dims = {'crack','time','pair'}`, `time`, `fs`, `tx`, `rx`, `element_positions`, `crack_size_unit`, `response_unit`.

### The single next dependency from MATLAB

> **The exact exported forward-library schema**: variable names, axis order of `responses`, whether `time_data` is the scattered field only or total field, crack-size units, `tx`/`rx` convention (FMC vs. HMC), and the time base/gating shared across all crack sizes.

Once a real file exists, the only change on the Python side should be a `key_map` / `dims` argument (or none at all if the template above is used).

## Testing

`pytest` runs 91 tests in about 4 s. They check behaviour, not just execution. Examples: posterior peaks at the truth and broadens with noise; agreement with the analytic linear-Gaussian posterior; analytic vs. numerical Paris RUL for $m \in \{1.5, 2, 2^+, 2.5, 3, 4\}$; RUL monotone in $a_0$ and $\Delta\sigma$ with $N \propto \Delta\sigma^{-m}$; Jensen bias; risk-based intervals that shrink with stricter $\alpha$, larger cracks and heavier loading; a mock MATLAB v7.3 file; and malformed-data errors.

## Key references

- Holmes, Drinkwater & Wilcox (2005), *Post-processing of the full matrix of ultrasonic transmit–receive array data for NDE*, NDT&E Int. 38(8):701–711. [doi:10.1016/j.ndteint.2005.04.002](https://doi.org/10.1016/j.ndteint.2005.04.002)
- Drinkwater & Wilcox (2006), *Ultrasonic arrays for non-destructive evaluation: a review*, NDT&E Int. 39(7):525–541. [doi:10.1016/j.ndteint.2006.03.006](https://doi.org/10.1016/j.ndteint.2006.03.006)
- Tarantola (2005), *Inverse Problem Theory and Methods for Model Parameter Estimation*, SIAM. [doi:10.1137/1.9780898717921](https://doi.org/10.1137/1.9780898717921)
- Kaipio & Somersalo (2005), *Statistical and Computational Inverse Problems*, Springer. [doi:10.1007/b138659](https://doi.org/10.1007/b138659)
- Gelman et al. (2013), *Bayesian Data Analysis*, 3rd ed., CRC. [doi:10.1201/b16018](https://doi.org/10.1201/b16018)
- Paris & Erdogan (1963), *A critical analysis of crack propagation laws*, J. Basic Eng. 85(4):528–534. [doi:10.1115/1.3656900](https://doi.org/10.1115/1.3656900)
- Anderson (2017), *Fracture Mechanics: Fundamentals and Applications*, 4th ed., CRC. [doi:10.1201/9781315370293](https://doi.org/10.1201/9781315370293)
- Tada, Paris & Irwin (2000), *The Stress Analysis of Cracks Handbook*, 3rd ed., ASME. [doi:10.1115/1.801535](https://doi.org/10.1115/1.801535)
- BSI (2019), *BS 7910:2019 Guide to methods for assessing the acceptability of flaws in metallic structures*. [BSI Knowledge](https://knowledge.bsigroup.com/products/guide-to-methods-for-assessing-the-acceptability-of-flaws-in-metallic-structures)
- Straub (2009), *Stochastic modeling of deterioration processes through dynamic Bayesian networks*, J. Eng. Mech. 135(10):1089–1099. [doi:10.1061/(ASCE)EM.1943-7889.0000024](https://doi.org/10.1061/(ASCE)EM.1943-7889.0000024)
