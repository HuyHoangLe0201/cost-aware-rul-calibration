"""An online calibrator that turns RUL predictions into cost-optimal replacement dates."""
from __future__ import annotations

from typing import Hashable, Iterable, Optional

import numpy as np

from .core import conformal_quantile, critical_level, gaussian_quantile, kaplan_meier_quantile, target_coverage

__all__ = ["CostAwareCalibrator"]

_METHODS = ("auto", "conformal", "kaplan-meier", "gaussian", "naive", "drop")


class CostAwareCalibrator:
    """Learns the multiplier ``q`` of the replacement date ``d = p * q`` from the units that have already been completed.

    Parameters
    ----------
    c_under, c_over
        Cost per unit time of running past failure and of wasting useful life; only their ratio matters.
    window
        ``None`` (default): use every label so far (cumulative calibration, the proposed policy).  An integer ``w``: use the last
        ``w`` labels only (rolling calibration, for drifting fleets; Proposition 7 of the article suggests ``w`` ~ s**(-2/3) for a drift of s per unit).
    method
        ``"auto"`` (default): conformal quantile while all labels are exact, Kaplan-Meier as soon as a censored label is present.
        ``"conformal"``, ``"kaplan-meier"``, ``"gaussian"`` select a rule explicitly.  ``"naive"`` (record a censored label as exact)
        and ``"drop"`` (discard censored labels) exist for benchmarking only: both bias the multiplier towards early replacement.
    min_pool, fallback
        With fewer than ``min_pool`` labels (or if the quantile is not identified) the multiplier is ``fallback`` (default 1).

    Labels
    ------
    ``update(predicted_rul, observed)``: the unit failed after ``observed`` time units from the alarm, whose point prediction was ``predicted_rul``.
    If the unit was replaced preventively after the lead time ``d`` without having failed, call ``update(predicted_rul, d, censored=True)``:
    its true RUL is then only known to exceed ``d``.

    Strata
    ------
    ``group`` keeps a separate pool of labels per group (stratified calibration, Proposition 6: worthwhile when a covariate known at
    the alarm shifts the distribution of the ratio).
    """

    def __init__(self, c_under: float, c_over: float = 1.0, *, window: Optional[int] = None, method: str = "auto",
                 min_pool: int = 5, fallback: float = 1.0):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if window is not None and window < 1:
            raise ValueError("window must be a positive integer or None")
        self.c_under, self.c_over = float(c_under), float(c_over)
        self.tau = critical_level(self.c_under, self.c_over)
        self.window, self.method, self.min_pool, self.fallback = window, method, int(min_pool), float(fallback)
        self._labels: dict = {}

    # ------------------------------------------------------------------ labels
    @property
    def target_coverage(self) -> float:
        return target_coverage(self.c_under, self.c_over)

    def n_labels(self, group: Hashable = None) -> int:
        return len(self._labels.get(group, ()))

    def update(self, predicted_rul: float, observed: float, *, censored: bool = False, group: Hashable = None) -> "CostAwareCalibrator":
        p = float(predicted_rul)
        if not p > 0:
            raise ValueError("predicted_rul must be positive")
        if not float(observed) >= 0:
            raise ValueError("observed must be non-negative")
        if censored and self.method == "conformal":
            raise ValueError("censored labels need method='auto' or 'kaplan-meier'; 'conformal' assumes exact labels")
        if censored and self.method == "gaussian":
            raise ValueError("the gaussian rule does not handle censored labels")
        if censored and self.method == "drop":
            return self
        exact = (not censored) or self.method == "naive"
        pool = self._labels.setdefault(group, [])
        pool.append((float(observed) / p, exact))
        if self.window is not None and len(pool) > self.window:
            del pool[: len(pool) - self.window]
        return self

    def fit(self, predicted_rul: Iterable[float], observed: Iterable[float], censored: Optional[Iterable[bool]] = None,
            group: Optional[Iterable[Hashable]] = None) -> "CostAwareCalibrator":
        """Batch version of :meth:`update` (labels are added in the order given)."""
        p, o = list(predicted_rul), list(observed)
        c = [False] * len(p) if censored is None else list(censored)
        g = [None] * len(p) if group is None else list(group)
        if not (len(p) == len(o) == len(c) == len(g)):
            raise ValueError("inputs must have the same length")
        for pi, oi, ci, gi in zip(p, o, c, g):
            self.update(pi, oi, censored=bool(ci), group=gi)
        return self

    # ------------------------------------------------------------------ decisions
    def multiplier(self, group: Hashable = None) -> float:
        """The multiplier q: replace after ``p * q`` where ``p`` is the point prediction at the alarm."""
        pool = self._labels.get(group, [])
        if not pool:
            return self.fallback
        values = np.array([v for v, _ in pool])
        exact = np.array([e for _, e in pool])
        m = self.method
        if m == "auto":
            m = "conformal" if exact.all() else "kaplan-meier"
        if m in ("conformal", "naive", "drop"):
            return conformal_quantile(values, self.tau, self.min_pool, self.fallback)
        if m == "kaplan-meier":
            return kaplan_meier_quantile(values, exact, self.tau, self.min_pool, self.fallback)
        return gaussian_quantile(values, self.tau, self.min_pool, self.fallback)

    def replacement_lead_time(self, predicted_rul, group: Hashable = None):
        """Planned replacement lead time ``p * q`` for the point prediction(s) ``predicted_rul`` made at the alarm."""
        out = np.asarray(predicted_rul, dtype=float) * self.multiplier(group)
        return float(out) if out.ndim == 0 else out

    def __repr__(self) -> str:
        return (f"CostAwareCalibrator(c_under={self.c_under:g}, c_over={self.c_over:g}, tau*={self.tau:.3f}, window={self.window}, "
                f"method={self.method!r}, labels={sum(len(v) for v in self._labels.values())})")
