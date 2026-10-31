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

__all__ = ["critical_level", "target_coverage", "asymmetric_cost", "conformal_quantile", "gaussian_quantile", "kaplan_meier_quantile"]


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


def kaplan_meier_quantile(values, exact, tau: float, min_pool: int = 5, fallback: float = 1.0) -> float:
    """tau-quantile of rho from right-censored labels (Kaplan-Meier), at the same finite-sample level as :func:`conformal_quantile`.

    ``values[i]`` is the observed ratio ``R / p`` when ``exact[i]`` is true, and a *lower bound* of it (the ratio ``d / p`` at which the
    unit was replaced preventively) when ``exact[i]`` is false.  If the quantile is not identified by the data (it lies beyond the
    support covered by the censoring times) the ``fallback`` multiplier is returned.
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
    return float(vs[hit[0]]) if hit.size else float(fallback)
