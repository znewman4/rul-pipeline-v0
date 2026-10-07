# Stage 4 — Structural interpretation (`rul_pipeline.structural`)

**Role:** map crack size to the crack-driving force, $a^{(i)} \mapsto K_I^{(i)}$.

> **Placeholder.** This closed form will be replaced by an FE-derived structural interpretation, e.g. $K$ from J-integral or displacement extrapolation over a crack-size sweep, giving $Y(a)$ for the real geometry.

## 1. Equation

Linear-elastic fracture mechanics, Mode I ([Anderson, 2017](https://doi.org/10.1201/9781315370293), ch. 2; [Tada, Paris & Irwin, 2000](https://doi.org/10.1115/1.801535)):

```math
K_I = Y\,\sigma\,\sqrt{\pi a}
```

| Symbol | Meaning | SI unit |
|---|---|---|
| $a$ | crack depth | m |
| $\sigma$ | remote normal stress (use $\Delta\sigma$ to get $\Delta K$) | Pa |
| $Y$ | geometry factor | – |
| $K_I$ | stress intensity factor | Pa√m (1 MPa√m = 10⁶ Pa√m) |

Default $Y = 1.12$: the classical edge crack in a semi-infinite plate under remote tension ($Y \approx 1.1215$). For a finite plate $Y = Y(a/W)$ grows with depth. For semi-elliptical surface cracks see [Newman & Raju (1981)](https://doi.org/10.1016/0013-7944(81)90116-8).

**Hand check** (tested): $a$ = 2 mm, $\sigma$ = 100 MPa, $Y$ = 1.12 →
$K_I = 1.12 \times 10^8 \times \sqrt{\pi \times 0.002} = 8.878\times10^6$ Pa√m = 8.878 MPa√m.

## 2. Implementation

```python
def stress_intensity_mode_i(a, sigma, Y=Y_EDGE_CRACK):
    a = check_positive(a, "a (crack size, m)")
    sigma = check_positive(sigma, "sigma (stress, Pa)")
    Y = check_positive(Y, "Y (geometry factor)")
    K = Y * sigma * np.sqrt(np.pi * a)
    return K if K.ndim else K[()]
```

- **Broadcasting.** `a`, `sigma` and `Y` broadcast with NumPy rules. Posterior samples $a^{(i)}$ combine with scalar loads now and with sampled $Y^{(i)}, \sigma^{(i)}$ later (uncertainty propagation) without code changes. For example, `a` of shape `(4,)` with `sigma` of shape `(2, 1)` gives `(2, 4)`.
- **Units are a convention, not a library.** Every argument name carries its unit in the error message, so passing 2 (mm) instead of 0.002 (m) is not caught here. The loader's mm warning and the SI-only API are the defence.
- Compressive $\sigma \le 0$ is rejected because V0 does not model crack closure.

### Critical crack size

```math
K_I(a_c, \sigma_\text{max}) = K_{Ic} \;\;\Rightarrow\;\; a_c = \frac{1}{\pi}\left(\frac{K_{Ic}}{Y\sigma_\text{max}}\right)^2
```

```python
a_c = (K_Ic / (Y * sigma_max)) ** 2 / np.pi
```

This holds for constant $Y$ only. `scripts/run_v0.py` uses $a_c = \min(a_c^\text{fracture}, a_\text{wall})$; with $K_{Ic}$ = 60 MPa√m and $\sigma_\text{max}$ = 200 MPa, $a_c^\text{fracture}$ = 22.8 mm exceeds the 10 mm wall limit, so the wall limit governs. This mirrors the BS 7910 practice of checking both fracture and plastic collapse ([BSI, 2019](https://knowledge.bsigroup.com/products/guide-to-methods-for-assessing-the-acceptability-of-flaws-in-metallic-structures)).

## 3. Role in the probability handover

```math
K_I^{(i)} = G\!\left(a^{(i)}\right) = Y\,\Delta\sigma\sqrt{\pi a^{(i)}}, \qquad a^{(i)} \sim p(a \mid y)
```

`prognosis.propagate_rul` stores these as `delta_K_samples`, so the driving-force distribution is available for inspection, e.g. its distance from the threshold $\Delta K_{th}$ in a later version.

## 4. Tests (`tests/test_structural.py`)

Hand calculation; vector input with $K \propto \sqrt{a}$; broadcasting of sampled $Y$ and $\sigma$; five invalid-input cases; $K_I(a_c) = K_{Ic}$.
