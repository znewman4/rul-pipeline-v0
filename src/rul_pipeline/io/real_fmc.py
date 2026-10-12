"""Real (BRAIN) FMC and BristolFE pilot candidates on one common observation grid.

Real data: BRAIN ``exp_data`` .mat (``time_data`` (n_t, n_pairs), 1-based ``tx``/``rx``, half
matrix tx <= rx, 25 MHz, arbitrary instrument units, full scale +-1). Candidates: pilot
``crack_<n>um_top.mat`` (full symmetric FMC, displacement in m, 89.1 MHz).

Two facts shape everything downstream:

* The real half matrix holds each reciprocal pair ONCE. Using the mirrored full matrix in an
  independent-noise likelihood would count every off-diagonal pair twice, so observations are
  kept as the unique ``tx <= rx`` pairs (``pairs``) and candidates are sampled at the same pairs.
* Real and simulated amplitudes have unrelated units. Nothing here rescales anything; the
  amplitude gain is handled explicitly by the likelihood (``inversion.error_models``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample

from .bristolfe import ForwardLibrary


@dataclass(frozen=True)
class RealFMC:
    """Unique-pair real FMC. ``data`` is (n_pairs, n_t); ``pairs`` is (n_pairs, 2) 0-based (tx, rx)."""

    data: np.ndarray
    pairs: np.ndarray
    time: np.ndarray
    element_x: np.ndarray
    centre_freq: float
    source: str
    meta: dict = field(default_factory=dict)

    @property
    def fs(self) -> float:
        return 1.0 / float(self.time[1] - self.time[0])

    @property
    def n_elements(self) -> int:
        return int(self.element_x.size)


def load_brain_fmc(path: str | Path) -> RealFMC:
    """Load a BRAIN ``exp_data`` .mat and validate its structure."""
    path = Path(path)
    s = loadmat(path, squeeze_me=True, struct_as_record=False)["exp_data"]
    data = np.asarray(s.time_data, dtype=float)             # (n_t, n_pairs)
    tx = np.asarray(s.tx, dtype=int).ravel() - 1
    rx = np.asarray(s.rx, dtype=int).ravel() - 1
    time = np.asarray(s.time, dtype=float).ravel()
    xc = np.asarray(s.array.el_xc, dtype=float).ravel()
    n = xc.size
    if data.shape != (time.size, tx.size) or tx.size != rx.size:
        raise ValueError(f"{path.name}: time_data {data.shape} inconsistent with time/tx/rx.")
    if not np.all(np.diff(time) > 0) or not np.allclose(np.diff(time), time[1] - time[0], rtol=1e-6):
        raise ValueError(f"{path.name}: time axis must be uniform and increasing.")
    if not np.isfinite(data).all():
        raise ValueError(f"{path.name}: non-finite samples.")
    if tx.min() < 0 or max(tx.max(), rx.max()) >= n:
        raise ValueError(f"{path.name}: tx/rx indices out of range for {n} elements.")
    if not np.all(tx <= rx) or len({*zip(tx, rx)}) != tx.size or tx.size != n * (n + 1) // 2:
        raise ValueError(f"{path.name}: expected a complete unique half matrix (tx <= rx), {n * (n + 1) // 2} pairs.")
    return RealFMC(
        data=data.T.copy(), pairs=np.column_stack((tx, rx)), time=time, element_x=xc,
        centre_freq=float(s.array.centre_freq), source=str(path),
        meta={"amplitude_unit": "instrument units (full scale +-1)", "max_abs": float(np.abs(data).max())},
    )


def load_pilot_candidates(folder: str | Path, real: RealFMC) -> ForwardLibrary:
    """Pilot candidates (a = 0, 1, 2, 3 mm extension) on the real time grid and unique-pair set.

    Simulated traces are Fourier-resampled from the BristolFE grid to the real one (both records
    must span the same duration), then sampled at ``real.pairs``. The simulated array must have the
    same element count and pitch as the real one (element order is left-to-right in both).
    """
    folder = Path(folder)
    files = sorted(folder.glob("crack_*_top.mat"))
    if len(files) < 2:
        raise ValueError(f"{folder}: need at least two crack_*_top.mat files.")
    sizes, responses, prov = [], [], []
    ref = None
    for f in files:
        d = loadmat(f, squeeze_me=True)
        ep = np.asarray(d["element_positions"], dtype=float)
        t = np.asarray(d["time"], dtype=float).ravel()
        if ep.shape[0] != real.n_elements:
            raise ValueError(f"{f.name}: {ep.shape[0]} elements vs real {real.n_elements}.")
        pitch_sim, pitch_real = np.diff(ep[:, 0]).mean(), np.diff(real.element_x).mean()
        if not np.isclose(pitch_sim, pitch_real, rtol=1e-3):
            raise ValueError(f"{f.name}: pitch {pitch_sim * 1e3:.4f} mm vs real {pitch_real * 1e3:.4f} mm.")
        dur_sim = t[-1] + (t[1] - t[0])
        dur_real = real.time[-1] + (real.time[1] - real.time[0])
        if not np.isclose(dur_sim, dur_real, rtol=2e-3):
            raise ValueError(f"{f.name}: record length {dur_sim * 1e6:.3f} us vs real {dur_real * 1e6:.3f} us.")
        fmc = np.asarray(d["fmc"], dtype=float)
        if not np.isfinite(fmc).all():
            raise ValueError(f"{f.name}: non-finite FMC.")
        if np.abs(fmc - fmc.transpose(1, 0, 2)).max() > 1e-9 * np.abs(fmc).max():
            raise ValueError(f"{f.name}: FMC is not reciprocal; unique-pair sampling would be ambiguous.")
        fmc = resample(fmc, real.time.size, axis=-1)
        responses.append(fmc[real.pairs[:, 0], real.pairs[:, 1], :])
        sizes.append(float(d["crack_length_m"]))
        prov.append(f.name)
        ref = ref or d
    order = np.argsort(sizes)
    return ForwardLibrary(
        crack_sizes=np.asarray(sizes)[order], responses=np.stack(responses)[order],
        dims=("crack", "pair", "time"), time=real.time.copy(), fs=real.fs,
        units={"crack_sizes": "m (extension beyond the notch tip; 0 = notch only)", "time": "s",
               "responses": "m displacement, Fourier-resampled to the real time grid"},
        source=str(folder),
        metadata={"files": [prov[i] for i in order], "cL": float(ref["cL"]), "cS": float(ref["cS"]),
                  "sim_params": "photo-estimated pilot geometry 60 x 80 mm (NOT the measured specimen)"},
    )
