# Stage 3 — Bayesian grid inversion (`rul_pipeline.inversion`)

**Role:** turn one measurement $y$ into the posterior crack-size distribution $p(a \mid y)$ on the library grid $a_1 < \dots < a_N$.

## 1. Model

Measurement model with additive Gaussian noise ([Tarantola, 2005](https://doi.org/10.1137/1.9780898717921), ch. 1; [Kaipio & Somersalo, 2005](https://doi.org/10.1007/b138659), ch. 3):

```math
y = F(a) + e, \qquad e \sim \mathcal{N}(0, \Sigma), \qquad \Sigma = \sigma^2 I_M \;\;(\text{V0})
```

For each candidate:

```math
r_i = y - F(a_i), \qquad
\log p(y \mid a_i) = -\frac{\lVert r_i\rVert^2}{2\sigma^2} - \frac{M}{2}\log\!\left(2\pi\sigma^2\right)
```

Bayes' rule on the grid, with a **discrete** prior mass $p(a_i)$:

```math
\log p(a_i \mid y) = \log p(y \mid a_i) + \log p(a_i) - \log Z,
\qquad
\log Z = \log p(y) = \operatorname{logsumexp}_i\!\left[\log p(y\mid a_i) + \log p(a_i)\right]
```

## 2. Implementation

### Log-likelihood (`gaussian_log_likelihood`)

Residuals are formed **directly**, a few candidates at a time. The algebraically equivalent $\lVert y\rVert^2 - 2y^\top F_i + \lVert F_i\rVert^2$ cancels catastrophically when the fit is good, which is exactly where accuracy matters.

```python
y_flat = flatten_response(y, library)                    # same C-order as the library
F = library.flat_responses()                             # (N, M)
for start in range(0, library.n_candidates, chunk):
    r = y_flat[None, :] - F[start : start + chunk]
    ssr[start : start + chunk] = np.einsum("ij,ij->i", r, r)
return -0.5 * ssr / sigma**2 - 0.5 * m * np.log(2 * np.pi * sigma**2)
```

The Gaussian constant is kept, so `log_evidence` is a proper $\log p(y)$ that can compare noise models or libraries later. These are log **densities** and may be positive when $\sigma$ is small.

### Why log space

For a 64-element FMC, $M \approx 6.4\times10^5$, and log-likelihood *differences* between candidates reach $10^3$–$10^{13}$. Here $\exp(\cdot)$ underflows to 0 for every candidate. Normalisation therefore uses log-sum-exp (`utils.normalise_log_weights`):

```python
log_z = float(logsumexp(log_w))
return np.exp(log_w - log_z), log_z
```

### Prior (`make_prior`)

- `None`: uniform **mass** $1/N$ per grid point. On a non-uniform grid this is *not* a uniform density in $a$.
- array: any non-negative weights of length $N$, normalised. Zeros are allowed and give $\log 0 = -\infty$, i.e. that crack size is ruled out.

### Summaries (`grid_posterior` → `GridPosterior`)

```math
\hat a_\text{MAP} = a_{\arg\max_i p_i},\qquad
\mathbb{E}[a\mid y] = \sum_i a_i p_i,\qquad
\mathrm{sd}[a\mid y] = \Big(\sum_i (a_i - \mathbb{E}[a\mid y])^2 p_i\Big)^{1/2}
```

The **equal-tailed credible interval** uses the discrete quantile $Q(q) = \min\{a_i : \mathrm{CDF}(a_i) \ge q\}$, giving $[Q(\tfrac{1-\ell}{2}), Q(\tfrac{1+\ell}{2})]$. Its endpoints are grid points and its coverage is $\ge \ell$.

```python
cdf = np.cumsum(pmf); cdf /= cdf[-1]
idx = np.searchsorted(cdf, np.asarray(q) - 1e-12, side="left")
```

`GridPosterior` holds `crack_sizes, log_likelihood, prior, posterior, log_evidence, map_index, map, mean, std, credible_interval, credible_level`.

### Posterior sampling (`sample_posterior`): the handover to prognosis

```math
a^{(i)} \sim \mathrm{Categorical}(p_1,\dots,p_N)
```

```python
idx = rng.choice(a.size, size=n_samples, p=result.posterior)
if not jitter:
    return a[idx]
mid = 0.5 * (a[1:] + a[:-1])                             # cell edges at grid midpoints
lower = np.concatenate([[a[0] - (mid[0] - a[0])], mid])
upper = np.concatenate([mid, [a[-1] + (a[-1] - mid[-1])]])
return rng.uniform(lower[idx], upper[idx])
```

With `jitter=True`, each mass $p_i$ is spread uniformly over its grid cell (a piecewise-constant density). Samples are continuous, so Paris-law RUL histograms are not spiky, and no structure finer than the grid resolution is invented.

### Noise-level diagnostic (`estimate_sigma_profile`)

$\hat\sigma = \sqrt{\min_i \lVert y - F(a_i)\rVert^2 / M}$, the maximum-likelihood $\sigma$ at the best grid point. This is a diagnostic only, because it absorbs model discrepancy into "noise".

## 3. Behaviour observed on synthetic FMC data

| Peak SNR | 64-element posterior sd (toy, $a_\text{true}$ = 2.03 mm, 0.1 mm grid) |
|---|---|
| +20 dB | 0 (all mass in one cell) |
| −10 dB | 2×10⁻⁴ mm |
| −18 dB | 0.04–0.17 mm |
| −20 dB | ≈ 0.6 mm, sometimes multimodal |

1. **Over-confidence.** The i.i.d. likelihood counts all $M$ samples as independent, so the posterior variance shrinks like $\sigma^2/\lVert\partial F/\partial a\rVert^2$, which is tiny for FMC. Real posteriors are wider because of correlated noise and model discrepancy ([Kennedy & O'Hagan, 2001](https://doi.org/10.1111/1467-9868.00294); [Brynjarsdóttir & O'Hagan, 2014](https://doi.org/10.1088/0266-5611/30/11/114007)). V0 only exposes this. V1 should replace $\sigma^2 I$ with $\Sigma = \sigma^2 I + \Sigma_\delta$.
2. **Cycle skipping.** The misfit $\lVert F(a_\text{true}) - F(a)\rVert^2$ has secondary minima about $c/(2f_c) \approx 0.59$ mm apart, because a $\Delta a$ shift of the tip changes the pulse-echo time by $2\Delta a/c$. At low SNR the posterior can lock onto the wrong cycle, which the notebook demonstrates. The exhaustive grid always *sees* all modes. Gradient-based or Laplace approximations would not.

## 4. Usage

```python
from rul_pipeline.inversion import invert_grid, sample_posterior
post = invert_grid(y, lib, sigma)               # optional prior=..., credible_level=0.95
print(post.summary())                           # MAP, mean, std, CI in mm
a_samples = sample_posterior(post, 20_000, rng=rng, jitter=True)
```

## 5. Tests (`tests/test_grid_bayes.py`)

| Test | What it establishes |
|---|---|
| peaks at true crack size (low noise) | MAP = $a_\text{true}$, $p > 0.99$ |
| sums to one | for $\sigma$ from $10^{-3}$ to 1 |
| broadens with noise | sd strictly increases over 4 noise levels (10-seed average); CI widens from 0 to > 0.5 mm |
| MAP sensible, off-grid truth | within one grid step; MAP and mean inside the CI |
| **analytic linear-Gaussian** | for $F(a) = a\,g$, flat prior: mean $= g^\top y/g^\top g$ (rel 1e-4), sd $= \sigma/\lVert g\rVert$ (rel 1 %) |
| log-space stability | finite posterior at log-likelihood differences around $10^{13}$ |
| user prior | zero-mass region gets zero posterior |
| Gaussian constant | exact values for a 2-sample case |
| sampling | empirical frequencies match $p_i$ (200k draws); jitter stays within the cell |
| $\hat\sigma$ recovery | within 2 % |
