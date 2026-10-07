# Stage 1 — BristolFE data interface (`rul_pipeline.io`)

**Role in the chain:** provide the forward library $\{a_i, F(a_i)\}_{i=1}^N$ that the inversion compares the measurement against.

```
BristolFE (MATLAB)  ──.mat / .npz──►  load_forward_library  ──►  ForwardLibrary  ──►  inversion
```

## 1. Canonical representation

| Field | Shape | Unit | Notes |
|---|---|---|---|
| `crack_sizes` | `(N,)` | m | strictly increasing, > 0 |
| `responses` | `(N, *meas)` | as exported | `responses[i]` $= F(a_i)$ |
| `dims` | `len == responses.ndim` | – | axis names; `dims[0] == "crack"` |
| `time` | `(n_t,)` | s | optional; must match the `"time"` axis |
| `fs` | scalar | Hz | optional; must agree with $1/\Delta t$ to 0.1 % |
| `tx`, `rx` | `(n_pairs,)` | index | optional; as exported (MATLAB: 1-based) |
| `element_positions` | `(n_el, 2\|3)` | m | optional |
| `units`, `source`, `metadata["provenance"]` | – | – | record every conversion the loader applied |

Typical `dims`:

| `dims` | Meaning |
|---|---|
| `("crack", "tx", "rx", "time")` | FMC cube, one A-scan per transmit/receive pair ([Holmes et al., 2005](https://doi.org/10.1016/j.ndteint.2005.04.002)) |
| `("crack", "time", "pair")` | BristolFE native `res.fmc.time_data` $(n_t \times n_\text{pairs})$ stacked over cracks |
| `("crack", "feature")` | any pre-flattened measurement vector |

The container is a frozen dataclass that validates itself on construction:

```python
@dataclass(frozen=True)
class ForwardLibrary:
    crack_sizes: np.ndarray
    responses: np.ndarray
    dims: tuple[str, ...]
    time: np.ndarray | None = None
    fs: float | None = None
    tx: np.ndarray | None = None
    rx: np.ndarray | None = None
    element_positions: np.ndarray | None = None
    units: dict[str, str] = ...
    source: str | None = None
    metadata: dict[str, Any] = ...

    def __post_init__(self) -> None:
        validate_forward_library(self)
```

## 2. Consistent flattening

The inversion works with vectors of length $M = \prod(\text{meas shape})$. Flattening must be identical for the library and the measurement, otherwise sample $j$ of $y$ would be compared with a different sample of $F(a_i)$. Both paths therefore use one C-order reshape. The measurement must match `measurement_shape` exactly, and it is **never** transposed to fit:

```python
def flat_responses(self) -> np.ndarray:
    return self.responses.reshape(self.n_candidates, -1)            # (N, M)

def flatten_response(y, lib):
    if y.shape != lib.measurement_shape:
        raise ValueError(f"Measurement shape {y.shape} does not match library measurement shape "
                         f"{lib.measurement_shape} ... Reorder the measurement explicitly (np.moveaxis)")
    return check_finite(y, "measurement").reshape(-1)
```

## 3. Loading `.mat` (v5/v7 and v7.3) and `.npz`

```python
lib = load_forward_library(
    "data/bristolfe/sweep.mat",
    key_map={"responses": "lib.fmc_all", "crack_sizes": "lib.a"},   # optional: names in the file
    dims=("time", "pair", "crack"),                                  # axis names as STORED
    crack_axis=-1,                                                   # explicit: crack axis is last
    crack_size_unit="mm",                                            # explicit unit conversion
)
```

| Format | Reader | Treatment |
|---|---|---|
| `.npz` | `np.load(allow_pickle=False)` | as stored |
| `.mat` v5/v7 | `scipy.io.loadmat(squeeze_me=False, struct_as_record=False)` | MATLAB index order is preserved. Not squeezed, so singleton axes never disappear. Dotted `key_map` paths walk struct fields. |
| `.mat` v7.3 | `h5py` (detected with `h5py.is_hdf5`) | HDF5 stores MATLAB's column-major arrays with **reversed** dimensions. The loader reverses them back (`.T`) and records this in provenance. `char` and `cell` datasets are decoded. |

The only axis reorderings ever applied:

1. undoing the v7.3 HDF5 reversal (restores MATLAB order), and
2. `np.moveaxis(responses, crack_axis, 0)` when the caller passes `crack_axis`.

Both are appended to `lib.metadata["provenance"]`, e.g. `['read lib.mat as mat-v5', 'moved crack axis -1 to front (explicit crack_axis)', 'converted crack_sizes from mm to m']`.

## 4. Validation and error messages

`validate_forward_library` checks, in order, and raises `ValueError` naming the problem:

| Check | Example message |
|---|---|
| crack grid 1-D, finite, strictly increasing, > 0 | `crack_sizes must be strictly increasing (sorted, no duplicates).` |
| `responses.shape[0] == N` | `responses.shape[0] = 4 but there are 3 crack sizes ... Axis 2 has length 3; if that is the crack axis, pass crack_axis=2 to the loader (no automatic transposing).` |
| `responses` finite | `responses contains 480 non-finite value(s) (NaN or inf).` |
| one name per axis, `dims[0] == "crack"`, unique | `dims has 3 names ... but responses has 4 axes` |
| `time` length vs `"time"` axis; `fs` vs $1/\Delta t$ | `fs = 5e+07 Hz disagrees with 1/dt = 2.5e+07 Hz.` |
| `tx`/`rx` together, equal length, match `"pair"` axis | `tx and rx must be given together.` |
| plausibility | **warning** if $\max a > 0.5$ m: *"was the library exported in mm?"* |

## 5. BristolFE pair layout → FMC cube

BristolFE (like Bristol's BRAIN toolbox) stores FMC as columns of `time_data` with `tx(k)`, `rx(k)` per column. `pairs_to_fmc_cube` places column $k$ at `cube[tx[k], rx[k], :]` using explicit indices. Pairs that are not listed stay zero; for half-matrix capture, reciprocity must be applied explicitly.

```python
tx = np.asarray(tx, dtype=int) - (1 if one_based else 0)
rx = np.asarray(rx, dtype=int) - (1 if one_based else 0)
cube = np.zeros(lead + (n_elements, n_elements, n_t))
cube[..., tx, rx, :] = np.moveaxis(d, -1, -2)       # d: (..., n_t, n_pairs)
```

## 6. Proposed MATLAB export

[`matlab/export_forward_library.m`](../../../matlab/export_forward_library.m) stacks a BristolFE crack-size sweep into the canonical schema: `responses (N × n_t × n_pairs)`, `dims = {'crack','time','pair'}`, `crack_sizes` in metres, plus `time`, `fs`, `tx`, `rx`, `element_positions`, `crack_size_unit`, `response_unit`. It saves `-v7` below 2 GB and `-v7.3` above. Python side: `load_forward_library(fname)` with no extra arguments.

**Assumptions to confirm with the MATLAB side:** all crack sizes share one time base and one `tx`/`rx` ordering; `time_data` is the scattered (defect) field or the total field consistently; and crack size means depth in metres.

## Tests (`tests/test_bristolfe_io.py`)

- canonical shapes and `flat_responses` / `flatten_response` row agreement;
- refusal to accept a transposed measurement;
- eight malformed-library cases, each with a specific message, plus the mm warning;
- `.npz` round trip;
- `.mat` v5 with struct paths, mm units and the crack axis stored last;
- a mock MATLAB v7.3 HDF5 file (reversed dims, `char` unit) that loads back in MATLAB order;
- missing variables, missing `dims` and unsupported extensions;
- `pairs_to_fmc_cube` placement.
