"""Planning tools from the finite-sample theory of the article (Theorem 3, Proposition 2)."""
from __future__ import annotations

import math

import numpy as np

from .core import critical_level

__all__ = ["expected_coverage", "excess_cost_constant", "required_calibration_size", "density_at_quantile"]


def expected_coverage(n: int, tau: float) -> float:
    """Expected P(R >= d) of the conformal policy with n exact labels: 1 - k/(n+1), k = ceil((n+1) tau) (at most 1/(n+1) below 1 - tau)."""
    k = min(max(int(math.ceil((n + 1) * tau)), 1), n)
    return 1.0 - k / (n + 1.0)


def excess_cost_constant(c_under: float, c_over: float, density: float) -> float:
    """C = kappa * tau* (1 - tau*) / (2 g(q*)): the expected excess cost of the conformal policy is ~ C / n (in units of the point prediction).

    ``density`` is the density g of the ratio R/p at its tau*-quantile q*.
    """
    tau = critical_level(c_under, c_over)
    if density <= 0:
        raise ValueError("density must be positive")
    return (c_under + c_over) * tau * (1.0 - tau) / (2.0 * density)


def required_calibration_size(c_under: float, c_over: float, density: float, tolerance: float) -> int:
    """Smallest n with C / n <= tolerance, i.e. an expected excess cost below ``tolerance`` (in units of the point prediction)."""
    return int(math.ceil(excess_cost_constant(c_under, c_over, density) / tolerance))


def density_at_quantile(ratios, tau: float, bandwidth: float | None = None) -> float:
    """Rough Gaussian-kernel estimate of the density of the ratio at its tau-quantile (kernel in log space, Silverman bandwidth).

    Only meant for sizing the calibration sample; with few labels it is noisy.
    """
    v = np.asarray(ratios, dtype=float)
    if v.size < 10:
        raise ValueError("need at least 10 ratios")
    lv = np.log(v)
    h = bandwidth if bandwidth else 1.06 * lv.std(ddof=1) * v.size ** (-0.2)
    q = float(np.quantile(v, tau))
    z = (np.log(q) - lv) / h
    f_log = np.exp(-0.5 * z * z).sum() / (v.size * h * math.sqrt(2.0 * math.pi))
    return float(f_log / q)
