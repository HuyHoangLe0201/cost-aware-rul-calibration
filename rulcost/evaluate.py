"""Compare replacement policies, and the handling of censored labels, on your own alarm data.

Both functions replay a stream of units in random order (``n_orders`` random permutations): the first ``n_init`` units supply the initial
labels (e.g. units tested to failure), every later unit is decided with the labels collected so far and then, once it is complete, its label is added.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .calibrator import CostAwareCalibrator
from .core import conformal_quantile, critical_level, gaussian_quantile

__all__ = ["evaluate_policies", "PolicyReport", "simulate_closed_loop"]

POLICY_LABELS = {"point": "point RUL", "safety": "safety factor", "fixed90": "fixed 90%", "gauss": "Gauss quantile (log-normal)",
                 "static": "tau* static (first n_init)", "cum": "tau* cumulative (proposed)", "roll": "tau* rolling"}


@dataclass
class PolicyReport:
    """Result of :func:`evaluate_policies`; ``cost`` is the mean over units of the per-unit mean cost, divided by ``normalizer``."""
    c_under: float
    c_over: float
    normalizer: float
    cost: Dict[str, float]
    coverage: Dict[str, float]
    per_unit_cost: Dict[str, np.ndarray] = field(repr=False)

    def table(self) -> str:
        rows = [f"{'policy':<30}{'cost / normalizer':>20}{'coverage':>12}"]
        for k in self.cost:
            name = POLICY_LABELS.get(k, f"safety factor {k[6:]}" if k.startswith("safety") else k)
            rows.append(f"{name:<30}{self.cost[k]:>20.3f}{self.coverage[k]:>12.3f}")
        tau = critical_level(self.c_under, self.c_over)
        rows.append(f"(target coverage of the tau* policies: {1 - tau:.3f}; c_under/c_over = {self.c_under / self.c_over:g})")
        return "\n".join(rows)

    __str__ = table


def evaluate_policies(predicted_rul, true_rul, c_under: float, c_over: float = 1.0, *, n_init: int = 20, window: int = 40, n_orders: int = 150,
                      seed: int = 11, normalizer: Optional[float] = None, safety_factors: Sequence[float] = (0.5, 0.7, 0.85),
                      fixed_level: float = 0.90, min_pool: int = 5) -> PolicyReport:
    """Replay the alarm stream under every policy of the article and report cost and coverage.

    ``predicted_rul[i]`` and ``true_rul[i]`` are, for unit ``i``, the point prediction at the alarm and the RUL actually observed from that moment.
    Policies: ``point`` (q = 1), ``safety@a`` (q = a), ``fixed90`` (conformal quantile at tau = 1 - fixed_level over the last ``window`` units,
    whatever the costs), ``gauss`` (log-normal fit, level tau*), ``static`` (conformal, first ``n_init`` units only), ``cum`` (conformal, all units
    so far; proposed) and ``roll`` (conformal, last ``window`` units).  Cost is ``c_under * overrun + c_over * waste`` divided by ``normalizer``
    (default: mean predicted RUL).  ``coverage`` is the fraction of decisions with ``R >= d``.
    """
    p = np.asarray(predicted_rul, dtype=float)
    r = np.asarray(true_rul, dtype=float)
    n = p.size
    if p.shape != r.shape or p.ndim != 1:
        raise ValueError("predicted_rul and true_rul must be one-dimensional and of equal length")
    if np.any(p <= 0):
        raise ValueError("predicted RUL values must be positive")
    if n <= n_init + 1:
        raise ValueError(f"need more than n_init + 1 = {n_init + 1} units, got {n}")
    norm = float(normalizer) if normalizer else float(p.mean())
    tau = critical_level(c_under, c_over)
    ratio = r / p
    names = ["point"] + [f"safety@{a:g}" for a in safety_factors] + ["fixed90", "gauss", "static", "cum", "roll"]
    tot = {k: np.zeros(n) for k in names}
    cnt = np.zeros(n)
    cov = {k: 0.0 for k in names}
    ndec = 0
    rng = np.random.default_rng(seed)
    for _ in range(n_orders):
        order = rng.permutation(n)
        init, test = order[:n_init], order[n_init:]
        done = list(init)
        srt_cum = sorted(ratio[init])
        srt_init = np.array(srt_cum)
        for e in test:
            arr_cum = np.array(srt_cum)
            roll = np.sort(ratio[done[-window:]])
            q = {"point": 1.0, "fixed90": conformal_quantile(roll, 1.0 - fixed_level, min_pool), "gauss": gaussian_quantile(arr_cum, tau, min_pool),
                 "static": conformal_quantile(srt_init, tau, min_pool), "cum": conformal_quantile(arr_cum, tau, min_pool),
                 "roll": conformal_quantile(roll, tau, min_pool)}
            for a in safety_factors:
                q[f"safety@{a:g}"] = float(a)
            for k in names:
                d = p[e] * q[k]
                tot[k][e] += (c_under * max(d - r[e], 0.0) + c_over * max(r[e] - d, 0.0)) / norm
                cov[k] += float(r[e] >= d)
            cnt[e] += 1
            ndec += 1
            bisect.insort(srt_cum, ratio[e])
            done.append(e)
    seen = cnt > 0
    per_unit = {k: np.where(seen, tot[k] / np.maximum(cnt, 1), np.nan) for k in names}
    return PolicyReport(c_under=float(c_under), c_over=float(c_over), normalizer=norm,
                        cost={k: float(np.nanmean(v)) for k, v in per_unit.items()}, coverage={k: cov[k] / ndec for k in names},
                        per_unit_cost=per_unit)


def simulate_closed_loop(predicted_rul, true_rul, c_under: float, c_over: float = 1.0, *, n_init: int = 20, n_orders: int = 60, seed: int = 5,
                         normalizer: Optional[float] = None, methods: Sequence[str] = ("oracle", "naive", "drop", "kaplan-meier"), risk_guard=None) -> Dict[str, Dict[str, float]]:
    """Closed-loop calibration when preventive replacement censors the labels (Table 3 of the article).

    A unit replaced at ``d = p * q`` before it fails reveals only that its RUL exceeds ``d``.  ``oracle`` keeps the exact label anyway (the
    benchmark), ``naive`` records the censored value as exact, ``drop`` discards the unit, ``kaplan-meier`` keeps it as censored.  Returns, per method,
    the mean cost (divided by ``normalizer``, default: mean predicted RUL) and the mean coverage ``P(R >= d)``.
    """
    p = np.asarray(predicted_rul, dtype=float)
    r = np.asarray(true_rul, dtype=float)
    n = p.size
    norm = float(normalizer) if normalizer else float(p.mean())
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(n) for _ in range(n_orders)]
    out = {}
    for m in methods:
        cal_method = {"oracle": "conformal"}.get(m, m)
        costs, covs = [], []
        for order in orders:
            cal = CostAwareCalibrator(c_under, c_over, method=cal_method, risk_guard=risk_guard)
            for e in order[:n_init]:
                cal.update(p[e], r[e])
            c_list, v_list = [], []
            for e in order[n_init:]:
                d = p[e] * cal.multiplier()
                fail = d >= r[e]
                c_list.append((c_under * max(d - r[e], 0.0) + c_over * max(r[e] - d, 0.0)) / norm)
                v_list.append(r[e] >= d)
                if m == "oracle" or fail:
                    cal.update(p[e], r[e])
                else:
                    cal.update(p[e], d, censored=True)
            costs.append(np.mean(c_list))
            covs.append(np.mean(v_list))
        out[m] = {"cost": float(np.mean(costs)), "coverage": float(np.mean(covs))}
    return out
