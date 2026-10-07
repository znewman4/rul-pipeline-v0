# Stage 5 — Paris-law crack growth (`rul_pipeline.crack_growth`)

**Role:** the deterministic map $H$ from current crack size to remaining cycles, $\mathrm{RUL}^{(i)} = H(a^{(i)})$, and its inverse $a(N)$.

## 1. Model and V0 assumptions

[Paris & Erdogan (1963)](https://doi.org/10.1115/1.3656900):

```math
\frac{da}{dN} = C\,(\Delta K)^m, \qquad \Delta K = Y\,\Delta\sigma\,\sqrt{\pi a}
```

V0 assumes constant-amplitude loading, Mode I only, constant $Y$, and fixed (not uncertain) $C$ and $m$, valid in the Paris (region II) regime with no threshold and no $K_c$ asymptote.

### Parameters: BS 7910 steel in air

The BristolFE examples model **steel** ($E$ = 210 GPa, $\nu$ = 0.3), so V0 uses the BS 7910 simplified Paris law for steels in air ([BSI, 2019](https://knowledge.bsigroup.com/products/guide-to-methods-for-assessing-the-acceptability-of-flaws-in-metallic-structures)). It is a mean + 2 SD design curve for $R < 0.5$:

```math
\frac{da}{dN}\,[\text{mm/cycle}] = 5.21\times10^{-13}\,\left(\Delta K\,[\text{N mm}^{-3/2}]\right)^{3}
```

Because the numerical value of $C$ depends on the units of $\Delta K$ **and** on $m$, the conversion is explicit (`paris_C_to_SI`):

```math
C_\text{SI} = C \cdot \frac{\ell_{da}}{\kappa^{m}}, \qquad
\ell_{da} = 10^{-3}\ \text{m/mm},\quad
\kappa = 1\ \text{N mm}^{-3/2} = 10^{6}\sqrt{10^{-3}}\ \text{Pa}\sqrt{\text{m}} = 31\,622.8\ \text{Pa}\sqrt{\text{m}}
```

```math
C_\text{SI} = 5.21\times10^{-13}\cdot\frac{10^{-3}}{(31\,622.8)^3} = 1.6475\times10^{-29}\ \ \frac{\text{m/cycle}}{(\text{Pa}\sqrt{\text{m}})^{3}},
\qquad m = 3
```

Equivalently $1.65\times10^{-11}$ m/cycle with $\Delta K$ in MPa√m (cross-checked in the tests). This is the same order as Barsom & Rolfe's ferritic–pearlitic steel law ([Barsom & Rolfe, 1999](https://doi.org/10.1520/MNL41-3RD-EB)).

```python
MPA_SQRT_M = 1e6
N_MM_POW_MINUS_3_2 = 1e6 * np.sqrt(1e-3)

def paris_C_to_SI(C, m, *, da_unit_m, dK_unit_Pa_sqrt_m):
    return float(C) * da_unit_m / dK_unit_Pa_sqrt_m ** float(m)

C_BS7910_AIR_SI = paris_C_to_SI(5.21e-13, 3.0, da_unit_m=1e-3, dK_unit_Pa_sqrt_m=N_MM_POW_MINUS_3_2)
```

## 2. Closed-form RUL (constant $Y$, $\Delta\sigma$, $C$, $m$)

Write $B = C\,(Y\Delta\sigma\sqrt{\pi})^m$ so that $da/dN = B\,a^{m/2}$, and let $\varepsilon = m/2 - 1$. Separating variables:

```math
N_f = \int_{a_0}^{a_c} \frac{da}{B\,a^{m/2}} =
\begin{cases}
\dfrac{a_0^{-\varepsilon} - a_c^{-\varepsilon}}{\varepsilon B} = \dfrac{a_0^{\,1-m/2} - a_c^{\,1-m/2}}{(m/2-1)\,C\,(Y\Delta\sigma\sqrt{\pi})^m}, & m \ne 2\\[2ex]
\dfrac{\ln(a_c/a_0)}{B}, & m = 2
\end{cases}
```

As $m \to 2$ the first form becomes $0/0$ and loses precision. The implementation rewrites $x^{-\varepsilon} = \exp(-\varepsilon\ln x)$ and uses `expm1`, which is exact to machine precision for all $m$ including $m \approx 2$:

```python
B = C * (Y * delta_sigma * np.sqrt(np.pi)) ** m
eps = m / 2.0 - 1.0
if eps == 0.0:
    N = np.log(a_c / a0_eff) / B
else:
    N = (np.expm1(-eps * np.log(a0_eff)) - np.expm1(-eps * np.log(a_c))) / (eps * B)
N = np.where(exceeded, 0.0, N)
```

**Hand check** (tested): $a_0$ = 2 mm, $a_c$ = 10 mm, $\Delta\sigma$ = 100 MPa, $Y$ = 1.12, BS 7910 constants:
$B = 1.6475\times10^{-29}(1.12\times10^8\sqrt\pi)^3 = 1.29\times10^{-4}$ m$^{-1/2}$/cycle and
$N_f = (0.002^{-1/2} - 0.01^{-1/2})/(0.5B) = (22.36 - 10)/(6.45\times10^{-5}) \approx 1.92\times10^5$ cycles.

### Crack size after $N$ cycles (inverse)

```math
a(N) = a_0\left(1 - \varepsilon B N a_0^{\varepsilon}\right)^{-1/\varepsilon}
\;\;\Longleftrightarrow\;\;
\ln a(N) = \ln a_0 - \tfrac{1}{\varepsilon}\,\mathrm{log1p}\!\left(-\varepsilon B N a_0^{\varepsilon}\right),
\qquad a(N) = a_0 e^{BN}\ (m=2)
```

For $m > 2$ the bracket reaches 0 at a finite $N$ (unstable "runaway" growth), after which $a = \infty$ is returned. This is used by the inspection scheduler and the update scaffold.

## 3. Numerical routes (for generality)

| Function | Method | Use |
|---|---|---|
| `rul_numerical` | `scipy.integrate.quad` of $N = \int da / (C\,\Delta K(a)^m)$ in $\ln a$ (`da = a dln a`) | any callable $Y(a)$, e.g. FE-derived or finite-width |
| `grow_crack_numerical` | `solve_ivp` (RK45) on $d\ln a/dN = C\Delta K^m / a$, optional terminal event at $a_\text{stop}$ | cycle stepping; template for variable amplitude later |

Integrating in $\ln a$ keeps the integrand smooth when $a$ spans decades.

## 4. Invalid input handling

| Case | Behaviour |
|---|---|
| $a_0 \le 0$, non-finite $a_0$, $a_c$, $C$, $m$, $\Delta\sigma$ | `ValueError` naming the argument and its unit |
| $C \le 0$ | `ValueError: C (Paris coefficient) must be strictly positive` |
| $m \notin (0, 10]$ | `ValueError: ... outside the physically plausible range` |
| $a_0 \ge a_c$ | `ValueError` by default; `on_exceed="zero"` returns RUL = 0 (used for posterior samples already beyond $a_c$) |

`ParisModel(C, m, delta_sigma, Y)` is a frozen, validated parameter bundle. The **assumed future loading** $\Delta\sigma$ travels with the model, so every downstream number is visibly conditional on it.

## 5. Tests (`tests/test_paris.py`)

- BS 7910 unit conversion (and the MPa√m cross-check);
- closed form against the hand formula;
- **analytical vs numerical** RUL to $10^{-7}$ relative for $m \in \{1.5, 2, 2+10^{-9}, 2.5, 3, 4\}$;
- continuity of the $m = 2$ branch;
- callable $Y(a)$ equals constant $Y$;
- **RUL decreases as $a_0$ grows** and **as $\Delta\sigma$ increases**, with $N \propto \Delta\sigma^{-m}$ exactly;
- $a(N_f) = a_c$;
- runaway gives inf;
- RK45 stepping matches the closed form to $10^{-6}$;
- 12 invalid-input cases for both RUL routes;
- `on_exceed="zero"`.

## 6. Not modelled (see top-level README)

Threshold and fracture asymptotes (NASGRO, [Forman et al., 1967](https://doi.org/10.1115/1.3609637)), load interaction ([Wheeler, 1972](https://doi.org/10.1115/1.3425362)), variable amplitude, and scatter and correlation of $C$ and $m$ ([Virkler et al., 1979](https://doi.org/10.1115/1.3443666)).
