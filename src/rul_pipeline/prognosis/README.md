# Stage 6 — Monte Carlo RUL (`rul_pipeline.prognosis`)

**Role:** turn the crack-size posterior into a remaining-life distribution by sampling, never by plugging in a point estimate.

## 1. Propagation

```math
a^{(i)} \sim p(a \mid y), \qquad
\Delta K^{(i)} = Y\Delta\sigma\sqrt{\pi a^{(i)}}, \qquad
\mathrm{RUL}^{(i)} = H\!\left(a^{(i)}; C, m, \Delta\sigma, Y, a_c\right), \qquad i = 1,\dots,n
```

$\{\mathrm{RUL}^{(i)}\}$ are i.i.d. draws from $p(\mathrm{RUL} \mid y)$ **conditional on** $C$, $m$, the assumed loading and $a_c$. Only the measurement-derived uncertainty is propagated in V0.

```python
def propagate_rul(a_samples, model, a_c, *, credible_level=0.90):
    a = check_positive(a_samples, "a_samples (m)").reshape(-1)
    dK = stress_intensity_mode_i(a, model.delta_sigma, model.Y)       # structural map G
    rul = model.rul(a, a_c, on_exceed="zero")                          # Paris map H, vectorised
    lo, hi = np.quantile(rul, [(1 - level) / 2, (1 + level) / 2])
    return RULResult(a_samples=a, delta_K_samples=dK, rul_samples=rul, ...,
                     fraction_already_failed=float(np.mean(a >= a_c)))
```

Samples with $a^{(i)} \ge a_c$ get RUL = 0 and are counted in `fraction_already_failed`. They are not dropped, because dropping them would bias the risk downward.

## 2. Summaries (plain sample statistics, no KDE)

| Quantity | Estimator |
|---|---|
| mean, median, sd | `np.mean`, `np.median`, `np.std(ddof=1)` |
| credible interval (level $\ell$) | equal-tailed empirical quantiles |
| $P(\mathrm{RUL} \le h \mid y)$ | empirical CDF, `searchsorted(sorted_rul, h, side="right") / n` |
| Monte Carlo error of a probability | $\sqrt{\hat p(1-\hat p)/n}$ (`mc_standard_error`) |

```python
def probability_of_failure(rul_samples, horizon):
    r = np.sort(rul_samples.reshape(-1))
    return np.searchsorted(r, np.asarray(horizon, float), side="right") / r.size
```

For small risks the MC error matters. Resolving $\alpha = 10^{-3}$ to ±10 % needs $n \approx 10^5$ samples. Rare-event methods (importance or subset sampling) are a later concern.

## 3. Why not the MAP?

$\mathrm{RUL}(a) \propto a^{1-m/2} - a_c^{1-m/2}$ is **convex** in $a$ for $m > 0$. By Jensen's inequality:

```math
\mathbb{E}[\mathrm{RUL}(a)] \;\ge\; \mathrm{RUL}\!\left(\mathbb{E}[a]\right)
```

A plug-in estimate is therefore biased, and it also hides multimodality. In `scripts/run_v0.py`, a 4 % secondary posterior mode at 1.4 mm (cycle skipping) produces a separate RUL mode near 2.6×10⁵ cycles that no point estimate would show. The script prints `RUL(a_MAP)` only for comparison.

## 4. Tests (`tests/test_prognosis.py`)

- RUL samples are positive, ordered between the deterministic RULs of the extreme samples, and equal to the per-sample map;
- the Monte Carlo mean matches the exact grid expectation $\sum_i p_i\,\mathrm{RUL}(a_i)$ within 4 standard errors;
- Jensen bias;
- RUL spread grows with posterior spread, and the median falls for larger cracks;
- a delta posterior gives deterministic RUL;
- $P_f$ is non-decreasing in horizon, runs from 0 to 1, and equals 0.5 at the median;
- already-failed samples are counted;
- quantiles are correct.
