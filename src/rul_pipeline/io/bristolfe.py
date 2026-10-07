"""Forward-library container and loaders for BristolFE (MATLAB) exports.

Canonical Python representation
-------------------------------
A forward library is the set of forward-model predictions over a 1-D grid of
crack sizes::

    crack_sizes : (N,)            a_1 < a_2 < ... < a_N             [m]
    responses   : (N, *meas)      responses[i] = F(a_i)             [arbitrary units]
    dims        : tuple[str]      one name per axis of ``responses``; dims[0] == "crack"

Typical ``dims``:

    ("crack", "tx", "rx", "time")    FMC cube, one A-scan per transmit/receive pair
    ("crack", "time", "pair")        BristolFE native ``res.fmc.time_data`` (n_t, n_pairs)
    ("crack", "feature")             any pre-flattened measurement vector

Nothing in this module ever transposes or squeezes ``responses`` implicitly.
The only axis reordering is (a) moving the crack axis to the front when the
caller explicitly passes ``crack_axis``, and (b) undoing the dimension reversal
that every HDF5 reader applies to MATLAB v7.3 files (so the array has the same
index order as in MATLAB). Both are recorded in ``metadata["provenance"]``.

The exact BristolFE export schema is not fixed yet, so ``load_forward_library``
takes a ``key_map`` that maps canonical names to whatever variables (or dotted
MATLAB struct paths, e.g. ``"lib.fmc.time_data"``) the export actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from ..utils.validation import check_finite, check_increasing_grid

#: Canonical variable names looked up in a file when no ``key_map`` override is given.
CANONICAL_KEYS: tuple[str, ...] = (
    "crack_sizes",        # (N,) crack sizes
    "responses",          # (N, ...) forward predictions
    "dims",               # cell/str array naming each axis of responses
    "time",               # (n_t,) time vector [s]
    "fs",                 # scalar sampling frequency [Hz]
    "tx",                 # (n_pairs,) transmitter index per pair (as exported)
    "rx",                 # (n_pairs,) receiver index per pair (as exported)
    "element_positions",  # (n_elements, 2|3) element centres [m]
    "crack_size_unit",    # "m" | "mm" | "um"
    "response_unit",      # free text, e.g. "displacement (m)" or "arb."
)

#: Metres per unit for the crack-size units we accept.
_LENGTH_UNITS: dict[str, float] = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6}


@dataclass(frozen=True)
class ForwardLibrary:
    """Forward-model library ``{a_i, F(a_i)}`` over a 1-D crack-size grid.

    All lengths in metres, time in seconds, frequency in Hz. ``responses`` keeps
    whatever physical unit the forward model produced (recorded in ``units``);
    the inversion only needs it to match the measurement's unit.
    """

    crack_sizes: np.ndarray
    responses: np.ndarray
    dims: tuple[str, ...]
    time: np.ndarray | None = None
    fs: float | None = None
    tx: np.ndarray | None = None
    rx: np.ndarray | None = None
    element_positions: np.ndarray | None = None
    units: dict[str, str] = field(default_factory=lambda: {"crack_sizes": "m"})
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_forward_library(self)

    @property
    def n_candidates(self) -> int:
        """Number of crack sizes N in the library."""
        return int(self.crack_sizes.size)

    @property
    def measurement_shape(self) -> tuple[int, ...]:
        """Shape of a single response F(a_i) (everything after the crack axis)."""
        return tuple(self.responses.shape[1:])

    @property
    def measurement_dims(self) -> tuple[str, ...]:
        """Axis names of a single response F(a_i)."""
        return self.dims[1:]

    def flat_responses(self) -> np.ndarray:
        """Responses as an (N, M) matrix, M = prod(measurement_shape), C order."""
        return self.responses.reshape(self.n_candidates, -1)


def validate_forward_library(lib: ForwardLibrary) -> None:
    """Check internal consistency of a :class:`ForwardLibrary`; raise ``ValueError``."""
    a = check_increasing_grid(lib.crack_sizes, "crack_sizes")
    if np.any(a <= 0):
        raise ValueError("crack_sizes must be strictly positive (metres).")
    if a.max() > 0.5:
        warnings.warn(
            f"Largest crack size is {a.max():.3g} m. crack_sizes are expected in metres; "
            "was the library exported in mm? Pass crack_size_unit='mm' to the loader.",
            stacklevel=3,
        )
    r = np.asarray(lib.responses)
    if r.ndim < 2:
        raise ValueError(
            f"responses must have shape (N, ...) with at least 2 axes; got shape {r.shape}."
        )
    if r.shape[0] != a.size:
        hint = ""
        matches = [ax for ax, n in enumerate(r.shape) if n == a.size]
        if matches:
            hint = (
                f" Axis {matches[0]} has length {a.size}; if that is the crack axis, "
                f"pass crack_axis={matches[0]} to the loader (no automatic transposing)."
            )
        raise ValueError(
            f"responses.shape[0] = {r.shape[0]} but there are {a.size} crack sizes; "
            f"responses must be ordered (N, ...).{hint}"
        )
    check_finite(r, "responses")
    if len(lib.dims) != r.ndim:
        raise ValueError(
            f"dims has {len(lib.dims)} names {lib.dims} but responses has {r.ndim} axes "
            f"(shape {r.shape}). Name every axis, e.g. ('crack', 'tx', 'rx', 'time')."
        )
    if lib.dims[0] != "crack":
        raise ValueError(f"dims[0] must be 'crack'; got {lib.dims[0]!r}.")
    if len(set(lib.dims)) != len(lib.dims):
        raise ValueError(f"dims must be unique; got {lib.dims}.")

    if lib.time is not None:
        t = check_increasing_grid(lib.time, "time")
        if "time" in lib.dims and r.shape[lib.dims.index("time")] != t.size:
            raise ValueError(
                f"time has {t.size} samples but the 'time' axis of responses has "
                f"{r.shape[lib.dims.index('time')]}."
            )
        if lib.fs is not None:
            fs_from_t = 1.0 / float(np.median(np.diff(t)))
            if not np.isclose(fs_from_t, lib.fs, rtol=1e-3):
                raise ValueError(f"fs = {lib.fs:.6g} Hz disagrees with 1/dt = {fs_from_t:.6g} Hz.")
    if lib.fs is not None and not (np.isfinite(lib.fs) and lib.fs > 0):
        raise ValueError(f"fs must be a positive finite frequency in Hz; got {lib.fs!r}.")

    if (lib.tx is None) != (lib.rx is None):
        raise ValueError("tx and rx must be given together.")
    if lib.tx is not None:
        if np.shape(lib.tx) != np.shape(lib.rx) or np.ndim(lib.tx) != 1:
            raise ValueError(
                f"tx and rx must be 1-D and equal length; got {np.shape(lib.tx)}, {np.shape(lib.rx)}."
            )
        if "pair" in lib.dims and r.shape[lib.dims.index("pair")] != np.size(lib.tx):
            raise ValueError(
                f"tx/rx list {np.size(lib.tx)} pairs but the 'pair' axis of responses has "
                f"{r.shape[lib.dims.index('pair')]}."
            )
    if lib.element_positions is not None:
        p = check_finite(lib.element_positions, "element_positions")
        if p.ndim != 2 or p.shape[1] not in (2, 3):
            raise ValueError(f"element_positions must be (n_elements, 2|3); got {p.shape}.")


def flatten_response(y: np.ndarray, lib: ForwardLibrary) -> np.ndarray:
    """Flatten one measurement exactly as :meth:`ForwardLibrary.flat_responses` does.

    ``y`` must have shape ``lib.measurement_shape`` exactly; it is never
    transposed to make it fit. This guarantees that element ``j`` of the
    flattened measurement corresponds to element ``j`` of every flattened F(a_i).
    """
    y = np.asarray(y, dtype=float)
    if y.shape != lib.measurement_shape:
        raise ValueError(
            f"Measurement shape {y.shape} does not match library measurement shape "
            f"{lib.measurement_shape} (dims {lib.measurement_dims}). Reorder the "
            "measurement explicitly (np.moveaxis) before inversion."
        )
    return check_finite(y, "measurement").reshape(-1)


def pairs_to_fmc_cube(
    time_data: np.ndarray,
    tx: Sequence[int],
    rx: Sequence[int],
    n_elements: int,
    *,
    one_based: bool = True,
    pair_axis: int = -1,
) -> np.ndarray:
    """Scatter BristolFE/BRAIN pair-ordered FMC data into a (tx, rx, time) cube.

    BristolFE returns ``res.fmc.time_data`` with shape ``(n_t, n_pairs)`` plus
    ``tx``/``rx`` vectors (MATLAB, so 1-based). This places column ``k`` at
    ``cube[tx[k], rx[k], :]``. Pairs that are not listed (e.g. half-matrix
    capture) stay zero, so if only half the matrix was simulated, mirror it with
    reciprocity explicitly before calling this. Extra leading axes (e.g. crack)
    are preserved: input ``(..., n_t, n_pairs)`` -> output ``(..., n_el, n_el, n_t)``.
    """
    d = np.moveaxis(np.asarray(time_data, dtype=float), pair_axis, -1)
    tx = np.asarray(tx, dtype=int) - (1 if one_based else 0)
    rx = np.asarray(rx, dtype=int) - (1 if one_based else 0)
    if d.shape[-1] != tx.size or tx.size != rx.size:
        raise ValueError(f"{d.shape[-1]} pairs in time_data but {tx.size} tx / {rx.size} rx indices.")
    if tx.min() < 0 or rx.min() < 0 or max(tx.max(), rx.max()) >= n_elements:
        raise ValueError(
            f"tx/rx indices out of range for n_elements={n_elements} (one_based={one_based})."
        )
    lead, n_t = d.shape[:-2], d.shape[-2]
    cube = np.zeros(lead + (n_elements, n_elements, n_t))
    cube[..., tx, rx, :] = np.moveaxis(d, -1, -2)
    return cube


# --------------------------------------------------------------------------- loading


def load_forward_library(
    path: str | Path,
    *,
    key_map: Mapping[str, str] | None = None,
    dims: Sequence[str] | None = None,
    crack_axis: int = 0,
    crack_size_unit: str | None = None,
) -> ForwardLibrary:
    """Load a forward library from ``.mat`` (v5/v7 or v7.3/HDF5) or ``.npz``.

    Parameters
    ----------
    path:
        File to read.
    key_map:
        Optional ``{canonical_name: name_in_file}`` overrides, e.g.
        ``{"responses": "lib.fmc_all", "crack_sizes": "a_grid"}``. Dotted names
        address fields of MATLAB structs. Unmapped names use :data:`CANONICAL_KEYS`.
    dims:
        Axis names for ``responses`` *as stored in the file* (before any crack-axis
        move). Required unless the file contains a ``dims`` variable.
    crack_axis:
        Which stored axis indexes crack size. MATLAB users often stack along the
        last axis (``crack_axis=-1``). The axis is moved to the front explicitly.
    crack_size_unit:
        Unit of the stored crack sizes ("m", "mm", "um"). Overrides any
        ``crack_size_unit`` variable in the file; defaults to "m".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    keys = {k: k for k in CANONICAL_KEYS}
    keys.update(key_map or {})
    unknown = set(key_map or {}) - set(CANONICAL_KEYS)
    if unknown:
        raise KeyError(f"Unknown canonical name(s) in key_map: {sorted(unknown)}; valid: {CANONICAL_KEYS}.")

    reader, fmt = _open_reader(path)
    provenance = [f"read {path.name} as {fmt}"]
    if fmt == "mat-v7.3":
        provenance.append("reversed HDF5 dims to restore MATLAB index order")

    missing = [k for k in ("crack_sizes", "responses") if reader(keys[k]) is None]
    if missing:
        raise KeyError(
            f"{path.name}: required variable(s) {[keys[k] for k in missing]} not found. "
            "Pass key_map={'crack_sizes': ..., 'responses': ...} to point at the right names."
        )

    responses = np.asarray(reader(keys["responses"]), dtype=float)
    stored_dims = dims if dims is not None else _as_str_list(reader(keys["dims"]))
    if stored_dims is None:
        raise ValueError(
            f"{path.name}: responses has shape {responses.shape} but no axis names. Pass dims=..., "
            "e.g. dims=('crack', 'tx', 'rx', 'time'), or export a 'dims' cell array."
        )
    stored_dims = tuple(stored_dims)
    if len(stored_dims) != responses.ndim:
        raise ValueError(f"dims {stored_dims} has {len(stored_dims)} names; responses has shape {responses.shape}.")
    ax = crack_axis % responses.ndim
    if ax != 0:
        responses = np.moveaxis(responses, ax, 0)
        stored_dims = (stored_dims[ax],) + stored_dims[:ax] + stored_dims[ax + 1 :]
        provenance.append(f"moved crack axis {crack_axis} to front (explicit crack_axis)")

    unit = crack_size_unit or _as_str(reader(keys["crack_size_unit"])) or "m"
    if unit not in _LENGTH_UNITS:
        raise ValueError(f"crack_size_unit {unit!r} not recognised; use one of {list(_LENGTH_UNITS)}.")
    a = _as_vector(reader(keys["crack_sizes"]), "crack_sizes") * _LENGTH_UNITS[unit]
    if unit != "m":
        provenance.append(f"converted crack_sizes from {unit} to m")

    time = reader(keys["time"])
    fs = reader(keys["fs"])
    tx, rx = reader(keys["tx"]), reader(keys["rx"])
    pos = reader(keys["element_positions"])
    response_unit = _as_str(reader(keys["response_unit"])) or "unspecified"
    return ForwardLibrary(
        crack_sizes=a,
        responses=responses,
        dims=stored_dims,
        time=None if time is None else _as_vector(time, "time"),
        fs=None if fs is None else float(np.squeeze(fs)),
        tx=None if tx is None else _as_vector(tx, "tx").astype(int),
        rx=None if rx is None else _as_vector(rx, "rx").astype(int),
        element_positions=None if pos is None else np.atleast_2d(np.asarray(pos, dtype=float)),
        units={"crack_sizes": "m", "time": "s", "fs": "Hz", "responses": response_unit},
        source=str(path),
        metadata={"provenance": provenance, "format": fmt},
    )


def save_forward_library_npz(lib: ForwardLibrary, path: str | Path) -> Path:
    """Write ``lib`` to ``.npz`` in the canonical schema (round-trips through the loader)."""
    path = Path(path)
    out: dict[str, Any] = {
        "crack_sizes": lib.crack_sizes,
        "responses": lib.responses,
        "dims": np.array(lib.dims),
        "crack_size_unit": np.array("m"),
        "response_unit": np.array(lib.units.get("responses", "unspecified")),
    }
    for name in ("time", "fs", "tx", "rx", "element_positions"):
        value = getattr(lib, name)
        if value is not None:
            out[name] = np.asarray(value)
    np.savez_compressed(path, **out)
    return path


def _open_reader(path: Path):
    """Return ``(reader, format)`` where ``reader(name)`` gives an array or ``None``."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        return (lambda name: data.get(name)), "npz"
    if suffix != ".mat":
        raise ValueError(f"Unsupported file type {suffix!r}; expected .mat or .npz.")

    import h5py

    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            cache: dict[str, Any] = {}
            for name in _all_h5_paths(f):
                cache[name] = _read_h5_matlab(f, f[name])
        return (lambda name: cache.get(name.replace(".", "/"))), "mat-v7.3"

    from scipy.io import loadmat

    # squeeze_me=False: never drop singleton axes from responses behind the user's back.
    raw = loadmat(path, squeeze_me=False, struct_as_record=False)

    def reader(name: str):
        obj: Any = raw
        for part in name.split("."):
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
                    obj = obj.item()
                obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    return reader, "mat-v5"


def _all_h5_paths(f) -> list[str]:
    names: list[str] = []
    f.visititems(lambda n, o: names.append(n) if o.__class__.__name__ == "Dataset" else None)
    return [n for n in names if not n.startswith("#refs#")]


def _read_h5_matlab(f, ds):
    """Read one MATLAB v7.3 dataset; reverse dims back to MATLAB order."""
    cls = ds.attrs.get("MATLAB_class", b"")
    cls = cls.decode() if isinstance(cls, bytes) else str(cls)
    if cls == "char":
        return "".join(chr(c) for c in np.asarray(ds[()]).T.ravel())
    if cls == "cell":
        refs = np.asarray(ds[()]).T
        return np.array([_read_h5_matlab(f, f[r]) for r in refs.ravel()], dtype=object).reshape(refs.shape)
    return np.asarray(ds[()]).T


def _as_vector(x, name: str) -> np.ndarray:
    """Accept (n,), (n, 1) or (1, n) and return (n,). Anything else is an error."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a vector; got shape {arr.shape}.")
    return arr


def _as_str(x) -> str | None:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.dtype == object or arr.size == 1:
        arr = np.asarray(arr.ravel()[0]) if arr.size else arr
    return str(arr.item() if arr.ndim == 0 else "".join(map(str, arr.ravel())))


def _as_str_list(x) -> list[str] | None:
    """Convert a MATLAB cell array of char / numpy str array to ``list[str]``."""
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.dtype.kind in ("U", "S") and arr.ndim <= 1:
        return [str(s) for s in np.atleast_1d(arr)]
    if arr.dtype.kind in ("U", "S") and arr.ndim == 2:  # MATLAB char matrix, one row per name
        return ["".join(row).strip() for row in arr]
    return [_as_str(item) for item in arr.ravel()]
