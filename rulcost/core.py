"""Building blocks: the cost, the critical level and the quantile rules for the ratio  rho = R / p.

Setting (see the article): at an alarm the point prediction of the remaining useful life (RUL) is ``p`` and the unknown true RUL is
``R = p * rho``.  A policy replaces the unit after the lead time ``d = p * q`` for a multiplier ``q``.  With the linear asymmetric cost

    c(d, R) = c_under * max(d - R, 0) + c_over * max(R - d, 0)

(``c_under``: cost per unit time of running beyond failure; ``c_over``: cost per unit time of useful life wasted by early replacement)
the best multiplier is the tau*-quantile of rho, with ``tau* = c_over / (c_under + c_over)`` (Proposition 1 of the article).
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

__all__ = ["critical_level", "target_coverage", "asymmetric_cost", "conformal_quantile", "gaussian_quantile", "kaplan_meier_quantile", "risk_floor"]


def critical_level(c_under: float, c_over: float = 1.0) -> float:
    """tau* = c_over / (c_under + c_over): the level of the ratio quantile that minimises the expected cost."""
    if c_under <= 0 or c_over <= 0:
        raise ValueError("c_under and c_over must be positive")
    return c_over / (c_under + c_over)


def target_coverage(c_under: float, c_over: float = 1.0) -> float:
    """1 - tau*: the probability that the unit outlives the planned replacement date, P(R >= d), of the optimal policy."""
    return 1.0 - critical_level(c_under, c_over)


def asymmetric_cost(lead_time, true_rul, c_under: float, c_over: float = 1.0):
    """c_under * (d - R)^+ + c_over * (R - d)^+  (vectorised)."""
    d = np.asarray(lead_time, dtype=float)
    r = np.asarray(true_rul, dtype=float)
    return c_under * np.maximum(d - r, 0.0) + c_over * np.maximum(r - d, 0.0)


def conformal_quantile(ratios, tau: float, min_pool: int = 5, fallback: float = 1.0) -> float:
    """Order statistic rho_(k) with k = ceil((n + 1) * tau), clipped to [1, n] (the conformal quantile of the article).

    With fewer than ``min_pool`` labels the multiplier ``fallback`` (default 1, i.e. replace at the point prediction) is returned.
    """
    v = np.asarray(ratios, dtype=float)
    n = v.size
    if n < min_pool:
        return float(fallback)
    k = min(max(int(math.ceil((n + 1) * tau)), 1), n)
    return float(np.sort(v)[k - 1])


def gaussian_quantile(ratios, tau: float, min_pool: int = 5, fallback: float = 1.0) -> float:
    """Log-normal rule: fit a normal to log(rho) and return its tau-quantile (the parametric baseline of the article).

    It has no coverage guarantee; use it only as a comparison.
    """
    v = np.asarray(ratios, dtype=float)
    if v.size < min_pool:
        return float(fallback)
    lv = np.log(v)
    return float(np.exp(lv.mean() + lv.std() * NormalDist().inv_cdf(tau)))


def risk_floor(n: int, guard="loglog") -> int:
    """Minimum number of units still at risk that a Kaplan-Meier quantile must rest on, for a pool of ``n`` labels.

    ``"loglog"``: ceil(log n * log log n); ``"log2"``: ceil(log(n)**2).  Theorem 6 of the article holds for any floor that is
    larger than every multiple of log n from some n on and smaller than n/4, and these two are examples (the first is the milder one).
    ``None`` or ``0``: no floor (the unguarded rule of Theorem 5); an integer ``k``: a fixed floor ``k``.
    """
    if guard is None:
        return 0
    L = math.log(max(n, 3))
    if guard == "loglog":
        return max(1, int(math.ceil(L * math.log(max(L, 1.0001)))))
    if guard == "log2":
        return int(math.ceil(L ** 2))
    return int(guard)


def kaplan_meier_quantile(values, exact, tau: float, min_pool: int = 5, fallback: float = 1.0, min_risk: int = 0) -> float:
    """tau-quantile of rho from right-censored labels (Kaplan-Meier), at the same finite-sample level as :func:`conformal_quantile`.

    ``values[i]`` is the observed ratio ``R / p`` when ``exact[i]`` is true, and a *lower bound* of it (the ratio ``d / p`` at which the
    unit was replaced preventively) when ``exact[i]`` is false.  If the quantile is not identified by the data (it lies beyond the
    support covered by the censoring times) the ``fallback`` multiplier is returned.

    ``min_risk`` is the risk-set guard of Theorem 6: a quantile that rests on fewer than ``min_risk`` units still at risk (units whose
    observed value is at least the quantile) is not trusted and ``fallback`` is returned instead.  ``0`` (default) switches the guard off.
    """
    v = np.asarray(values, dtype=float)
    d = np.asarray(exact, dtype=float)
    n = v.size
    if n < min_pool:
        return float(fallback)
    o = np.argsort(v, kind="stable")
    vs, ds = v[o], d[o]
    surv = np.cumprod(1.0 - ds / (n - np.arange(n)))
    cdf = 1.0 - surv
    t = min(tau * (n + 1) / n, 1.0)
    hit = np.where(cdf >= t)[0]
    if not hit.size:
        return float(fallback)
    q = float(vs[hit[0]])
    if min_risk and n - int(np.searchsorted(vs, q, side="left")) < min_risk:
        return float(fallback)
    return q
