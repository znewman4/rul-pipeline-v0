"""Version 0 risk-based choice of the next inspection interval.

Given samples a^(i) ~ p(a_k | y_1:k) at the current inspection (cycle N_k), a
growth model with an ASSUMED future loading, a limit size a_lim and an
acceptable risk alpha, choose the largest interval Delta N on a candidate grid
such that

    criterion "exceedance":    P(a(N_k + Delta N) >= a_lim | y_1:k) <= alpha
    criterion "failure_time":  P(T_lim <= N_k + Delta N | y_1:k)   <= alpha

where T_lim is the cycle count at which a reaches a_lim. With deterministic
Paris growth the two events are identical (a(.) is increasing), so both give
the same answer; they are kept separate because they diverge once growth
itself is stochastic. Both risks are Monte Carlo estimates over the samples.

The answer is only as good as the loading assumption: if the real stress range
is higher than ``model.delta_sigma``, the true risk at the chosen interval is
higher than alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..crack_growth.paris import ParisModel
from ..utils.validation import check_nonnegative, check_positive, check_probability


@dataclass(frozen=True)
class InspectionSchedule:
    """Selected interval and the risk curve it was chosen from."""

    interval: float                  # selected Delta N [cycles]; 0 => act/inspect now
    candidate_intervals: np.ndarray  # Delta N grid [cycles]
    risk: np.ndarray                 # risk(Delta N) on that grid
    alpha: float
    criterion: str
    a_limit: float                   # [m]
    delta_sigma: float               # assumed future stress range [Pa]
    capped: bool                     # True if even the largest candidate satisfied the risk target

    def summary(self) -> str:
        note = " (capped at largest candidate)" if self.capped else ""
        return (
            f"Next inspection in Delta N = {self.interval:.4g} cycles{note}: largest interval with "
            f"P({'a >= a_lim' if self.criterion == 'exceedance' else 'T_lim <= Delta N'}) <= {self.alpha:g}, "
            f"assuming Delta_sigma = {self.delta_sigma / 1e6:.4g} MPa"
        )


def exceedance_risk(a_samples: ArrayLike, model: ParisModel, a_limit: float, intervals: ArrayLike) -> np.ndarray:
    """P(a(Delta N) >= a_limit) for each Delta N, by growing every sample in closed form."""
    a = check_positive(a_samples, "a_samples (m)").reshape(-1)
    dn = check_nonnegative(intervals, "intervals (cycles)").reshape(-1)
    risk = np.empty(dn.size)
    step = max(1, 2_000_000 // a.size)                                 # bound the (intervals x samples) block
    for k in range(0, dn.size, step):
        a_future = model.crack_size_after(a[None, :], dn[k : k + step, None])
        risk[k : k + step] = np.mean(a_future >= float(a_limit), axis=1)
    return risk


def failure_time_risk(a_samples: ArrayLike, model: ParisModel, a_limit: float, intervals: ArrayLike) -> np.ndarray:
    """P(T_lim <= Delta N) for each Delta N, from the samples of time-to-limit T_lim."""
    a = check_positive(a_samples, "a_samples (m)").reshape(-1)
    dn = check_nonnegative(intervals, "intervals (cycles)").reshape(-1)
    t_lim = np.sort(model.rul(a, a_limit, on_exceed="zero"))
    return np.searchsorted(t_lim, dn, side="right") / t_lim.size


def select_inspection_interval(
    a_samples: ArrayLike,
    model: ParisModel,
    a_limit: float,
    alpha: float,
    *,
    intervals: ArrayLike | None = None,
    criterion: str = "exceedance",
) -> InspectionSchedule:
    """Largest candidate Delta N whose risk does not exceed ``alpha`` (grid search).

    Default candidates: 401 points from 0 to the 99.9 % quantile of T_lim.
    The search walks up the grid and stops at the first violation, so the
    result is the largest *contiguous* safe interval even if Monte Carlo noise
    made the risk curve non-monotone further out.
    """
    a = check_positive(a_samples, "a_samples (m)").reshape(-1)
    a_limit = float(check_positive(a_limit, "a_limit (m)"))
    alpha = check_probability(alpha, "alpha")
    if intervals is None:
        t_max = float(np.quantile(model.rul(a, a_limit, on_exceed="zero"), 0.999))
        intervals = np.linspace(0.0, max(t_max, 1.0), 401)
    dn = np.sort(check_nonnegative(intervals, "intervals (cycles)").reshape(-1))
    if criterion == "exceedance":
        risk = exceedance_risk(a, model, a_limit, dn)
    elif criterion == "failure_time":
        risk = failure_time_risk(a, model, a_limit, dn)
    else:
        raise ValueError("criterion must be 'exceedance' or 'failure_time'.")
    violations = np.flatnonzero(risk > alpha)
    if violations.size == 0:
        interval, capped = float(dn[-1]), True
    elif violations[0] == 0:
        interval, capped = 0.0, False
    else:
        interval, capped = float(dn[violations[0] - 1]), False
    return InspectionSchedule(
        interval=interval, candidate_intervals=dn, risk=risk, alpha=alpha, criterion=criterion,
        a_limit=a_limit, delta_sigma=model.delta_sigma, capped=capped,
    )
