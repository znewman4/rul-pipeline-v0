"""Independent repeat acquisitions -> mean, repeat variability and variance diagnostics.

Layout accepted (each group is a SEPARATE experiment; they are never pooled):

    <root>/fixed_probe/*.mat      repeat acquisitions with the probe left in place
    <root>/repositioned/*.mat     acquisitions with the probe removed and re-coupled

Each file is a BRAIN ``exp_data`` .mat. All scans in a group must share time axis, tx/rx order,
element positions and centre frequency. Amplitude gain is not stored in the file, so it must be
held constant by the operator; ``scale_ratio`` (max/min scan rms) is reported as a check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .real_fmc import RealFMC, load_brain_fmc

GROUPS = ("fixed_probe", "repositioned")


@dataclass(frozen=True)
class RepeatGroup:
    name: str
    scans: tuple[RealFMC, ...]

    @property
    def n(self) -> int:
        return len(self.scans)


def load_repeat_groups(root: str | Path) -> dict[str, RepeatGroup]:
    root = Path(root)
    out = {}
    for g in GROUPS:
        files = sorted((root / g).glob("*.mat")) if (root / g).is_dir() else []
        if files:
            out[g] = RepeatGroup(g, tuple(load_brain_fmc(f) for f in files))
            validate_group(out[g])
    if not out:
        raise FileNotFoundError(f"No scans: expected {root}/fixed_probe/*.mat and/or {root}/repositioned/*.mat")
    return out


def validate_group(group: RepeatGroup) -> None:
    ref = group.scans[0]
    for s in group.scans[1:]:
        name = Path(s.source).name
        if s.data.shape != ref.data.shape or not np.allclose(s.time, ref.time, rtol=0, atol=1e-12):
            raise ValueError(f"{group.name}: {name} time axis / size differs from {Path(ref.source).name}.")
        if not np.array_equal(s.pairs, ref.pairs):
            raise ValueError(f"{group.name}: {name} tx/rx ordering differs.")
        if not np.allclose(s.element_x, ref.element_x, atol=1e-9) or s.centre_freq != ref.centre_freq:
            raise ValueError(f"{group.name}: {name} array geometry / centre frequency differs.")


def regularised_variance(var_raw: np.ndarray, n: int, *, shrink: float = 0.5, floor: float | None = None) -> np.ndarray:
    """Regularised diagonal variance of a SINGLE acquisition from n repeats.

    var = shrink * pooled_t + (1 - shrink) * raw, floored. pooled_t is the per-time-sample median
    across pairs. Raw per-sample variances from few repeats are very noisy (relative sd
    sqrt(2/(n-1))) and can be ~0 by chance, which would make the likelihood singular. ``floor``
    defaults to 1e-3 x the median raw variance.
    """
    if n < 2:
        raise ValueError("Need >= 2 independent acquisitions to estimate variability.")
    pooled = np.median(var_raw, axis=0, keepdims=True)
    v = shrink * pooled + (1.0 - shrink) * var_raw
    fl = 1e-3 * float(np.median(var_raw)) if floor is None else floor
    return np.maximum(v, max(fl, np.finfo(float).tiny))


def summarise(processed: np.ndarray, *, shrink: float = 0.5) -> dict:
    """processed: (n, n_pairs, n_t) preprocessed scans of ONE group -> mean / variances / checks."""
    n = processed.shape[0]
    raw = processed.var(0, ddof=1)
    reg = regularised_variance(raw, n, shrink=shrink)
    rms = np.sqrt((processed**2).mean(axis=(1, 2)))
    return {"n": n, "mean": processed.mean(0), "var_raw": raw, "var_reg": reg, "scan_rms": rms,
            "scale_ratio": float(rms.max() / rms.min()),
            "sd_raw": float(np.sqrt(raw.mean())), "sd_reg": float(np.sqrt(reg.mean()))}
