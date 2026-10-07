# Stage 7 — Adaptive next-inspection interval (`rul_pipeline.inspection`)

**Role:** turn $p(a_k \mid y_{1:k})$ into a decision: how many cycles until the next inspection.

## 1. Criterion

At the current inspection (cycle $N_k$), with posterior samples $a_k^{(i)}$, a growth model with an **assumed** future loading, a limit size $a_\text{lim}$ (critical size or an action threshold) and an acceptable risk $\alpha$:

```math
\Delta N^\star = \max\left\{\Delta N \in \mathcal{G} \;:\; P\!\left(a(N_k + \Delta N) \ge a_\text{lim} \,\middle|\, y_{1:k}\right) \le \alpha\right\}
```

or equivalently in failure-time form, with $T_\text{lim}$ the cycles to reach $a_\text{lim}$:

```math
\Delta N^\star = \max\left\{\Delta N \in \mathcal{G} \;:\; P\!\left(T_\text{lim} \le N_k + \Delta N \,\middle|\, y_{1:k}\right) \le \alpha\right\}
```

$\mathcal{G}$ is a grid of candidate intervals. By default it has 401 points from 0 to the 99.9 % quantile of $T_\text{lim}$.

With deterministic Paris growth, $a(\cdot)$ is increasing, so $\{a(\Delta N) \ge a_\text{lim}\} = \{T_\text{lim} \le \Delta N\}$ and the two criteria coincide (tested). They are kept separate because they diverge once growth is stochastic, e.g. with uncertain $C$ or $m$ or random loading. This is a simplified, single-component form of risk-based inspection planning ([Straub & Faber, 2005](https://doi.org/10.1016/j.strusafe.2005.04.001)).

## 2. Implementation

**Exceedance risk.** Grow every sample in closed form to every candidate interval:

```math
\widehat{R}(\Delta N) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\!\left[a\!\left(\Delta N;\, a_k^{(i)}\right) \ge a_\text{lim}\right]
```

```python
step = max(1, 2_000_000 // a.size)                 # bound the (intervals x samples) block
for k in range(0, dn.size, step):
    a_future = model.crack_size_after(a[None, :], dn[k : k + step, None])
    risk[k : k + step] = np.mean(a_future >= float(a_limit), axis=1)
```

**Failure-time risk.** Empirical CDF of $T_\text{lim}^{(i)} = \mathrm{RUL}(a_k^{(i)} \to a_\text{lim})$, computed in $O(n\log n)$:

```python
t_lim = np.sort(model.rul(a, a_limit, on_exceed="zero"))
return np.searchsorted(t_lim, dn, side="right") / t_lim.size
```

**Selection.** Walk up the grid and stop at the first violation. This returns the largest *contiguous* safe interval even if Monte Carlo noise makes the risk curve non-monotone further out:

```python
violations = np.flatnonzero(risk > alpha)
if violations.size == 0:   interval, capped = dn[-1], True     # even the largest candidate is safe
elif violations[0] == 0:   interval = 0.0                      # risk already > alpha now: act / inspect now
else:                      interval = dn[violations[0] - 1]
```

The returned `InspectionSchedule` contains `interval, candidate_intervals, risk, alpha, criterion, a_limit, delta_sigma, capped`.

## 3. Conditionality on loading

The selected $\Delta N^\star$ is **only as good as `model.delta_sigma`**. If the real stress range is higher, the true risk at $\Delta N^\star$ exceeds $\alpha$. Because $N \propto \Delta\sigma^{-m}$, a true stress range 10 % above the assumed one shortens every time-to-limit by $1 - 1.1^{-3} \approx 25\,\%$ (for $m = 3$). The assumed $\Delta\sigma$ is therefore stored in the result, printed in `summary()`, and shown in the plot title.

## 4. Behaviour (default `run_v0.py`)

With $p(a\mid y)$ concentrated near 2.0 mm (a 4 % mode at 1.4 mm), $\Delta\sigma$ = 100 MPa, $a_\text{lim} = a_c$ = 10 mm and $\alpha = 10^{-2}$, the result is $\Delta N^\star \approx 1.79\times10^5$ cycles. That is about 6 % below the median RUL of $1.90\times10^5$. The interval is set by the lower tail of $p(\mathrm{RUL}\mid y)$, not by its centre.

## 5. Tests (`tests/test_scheduling.py`)

- the interval **decreases as $\alpha$ becomes stricter** (0.2 → 10⁻³);
- the interval **decreases for larger initial cracks** (1.5 → 5 mm);
- it decreases for heavier assumed loading;
- the selected interval is the largest safe candidate, with the next one violating;
- the two criteria agree exactly;
- returns "act now" when $P(a_k \ge a_\text{lim}) > \alpha$, and flags the capped case;
- invalid $\alpha$ and criterion are rejected.

## 6. Not modelled

Value of information and cost-optimal planning, probability of detection, Bayesian experimental design ([Lindley, 1956](https://doi.org/10.1214/aoms/1177728069); [Chaloner & Verdinelli, 1995](https://doi.org/10.1214/ss/1177009939)), continuous monitoring vs periodic inspection, and the *preposterior* effect: the next inspection will itself update $p(a)$.
