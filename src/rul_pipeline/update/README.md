# Stage 8 — Sequential Bayesian update scaffold (`rul_pipeline.update`)

**Role:** close the loop so that each new inspection updates the crack-size belief carried forward from the previous one. V0 provides a minimal, generic **grid (point-mass) filter**. It is not a DBN or particle filter yet.

## 1. Recursion

Between inspections $k$ and $k+1$ ($\Delta N$ cycles):

```math
\underbrace{p(a_{k+1}\mid y_{1:k})}_{\text{predicted prior}}
= \sum_{i} p(a_{k+1}\mid a_k = a_i)\; p(a_k = a_i \mid y_{1:k})
\qquad\text{(predict)}
```

```math
p(a_{k+1}\mid y_{1:k+1}) \;\propto\; p(y_{k+1}\mid a_{k+1})\; p(a_{k+1}\mid y_{1:k})
\qquad\text{(update)}
```

This is the discrete-state Bayes filter that DBN formulations of deterioration generalise ([Straub, 2009](https://doi.org/10.1061/(ASCE)EM.1943-7889.0000024)). Particle filters ([Arulampalam et al., 2002](https://doi.org/10.1109/78.978374)) are its sampling-based counterpart.

## 2. Predict: deterministic growth on a fixed grid (`predict_grid`)

V0 transition: $a_i \mapsto g(a_i) = a(\Delta N; a_i)$ (Paris closed form). The image $g(a_i)$ generally falls between grid nodes $a_j \le g(a_i) < a_{j+1}$, so its mass is **split linearly**:

```math
w = \frac{g(a_i) - a_j}{a_{j+1} - a_j}, \qquad
\tilde p_j \mathrel{+}= (1-w)\,p_i, \qquad \tilde p_{j+1} \mathrel{+}= w\,p_i
```

This conserves total mass and the mean ($\sum_j a_j\tilde p_j = \sum_i g(a_i)p_i$ for in-grid mass). Mass mapped above $a_N$, including runaway ($a = \infty$), cannot be represented by the library and is returned separately as `p_beyond`.

```python
beyond = a_new > a[-1]
a_in = np.clip(a_new, a[0], a[-1])
j = np.clip(np.searchsorted(a, a_in, side="right") - 1, 0, a.size - 2)
w_right = (a_in - a[j]) / (a[j + 1] - a[j])
np.add.at(pred, j,     np.where(beyond, 0.0, p * (1 - w_right)))
np.add.at(pred, j + 1, np.where(beyond, 0.0, p * w_right))
```

Optional **growth-model error**: a Gaussian transition kernel with standard deviation `growth_noise_std` [m], column-normalised on the grid. This is the simplest hook for uncertain $C$, $m$ or loading.

`propagate` is any callable `a -> a_new`, so Paris can be swapped for another law without touching the filter (`paris_propagator(model, delta_N)` is the V0 default).

## 3. Update: reuse of the inversion (`update_grid`, `sequential_update`)

The update is exactly the V0 grid inversion with the predicted prior in place of the uniform prior. It reuses `inversion.grid_posterior`, so summaries, credible intervals and sampling are identical:

```python
def sequential_update(posterior_k, model, delta_N, log_likelihood_next, *, growth_noise_std=0.0):
    pred, p_beyond = predict_grid(posterior_k.crack_sizes, posterior_k.posterior,
                                  paris_propagator(model, delta_N), growth_noise_std=growth_noise_std)
    return update_grid(posterior_k.crack_sizes, pred, log_likelihood_next), p_beyond
```

`log_likelihood_next` is `inversion.gaussian_log_likelihood(y_next, lib, sigma)`, i.e. the same forward library evaluated against the new measurement.

## 4. Tests (`tests/test_bayes_update.py`)

| Test | Check |
|---|---|
| zero growth | prediction is the identity |
| shift by $\delta$ | mass conserved; mean shifts by exactly $\delta$ |
| off-grid growth | `p_beyond` reported; total mass = 1 |
| Paris prediction | mean moves to larger $a$ |
| repeat measurement, no growth | sd shrinks by $\sqrt{2}$ (Gaussian product), within 2 % |
| growth + new measurement | posterior tracks $a(\Delta N)$ and narrows |

## 5. Next steps

Uncertain growth parameters as additional state (augmented grid, or particles), DBN discretisation, and the preposterior use of this recursion in inspection planning (value of information).
