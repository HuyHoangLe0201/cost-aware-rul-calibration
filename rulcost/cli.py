"""Command line interface:  python -m rulcost  {calibrate, compare, plan} ..."""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from . import __version__
from .calibrator import CostAwareCalibrator
from .evaluate import evaluate_policies, simulate_closed_loop
from .theory import density_at_quantile, excess_cost_constant, expected_coverage, required_calibration_size


def _read(path, predicted, observed, censored=None, group=None):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    for col in (predicted, observed) + tuple(c for c in (censored, group) if c):
        if col not in rows[0]:
            raise SystemExit(f"{path}: no column {col!r} (columns: {', '.join(rows[0])})")
    p = np.array([float(r[predicted]) for r in rows])
    o = np.array([float(r[observed]) for r in rows])
    c = np.array([r[censored].strip().lower() in ("1", "true", "yes", "t") for r in rows]) if censored else np.zeros(len(rows), bool)
    g = [r[group] for r in rows] if group else [None] * len(rows)
    return p, o, c, g


def _common(sp):
    sp.add_argument("--c-under", type=float, required=True, help="cost per unit time of running past failure")
    sp.add_argument("--c-over", type=float, default=1.0, help="cost per unit time of useful life wasted by early replacement (default 1)")


def _columns(sp):
    sp.add_argument("--predicted", default="predicted_rul", help="column with the point RUL prediction at the alarm")
    sp.add_argument("--observed", default="true_rul", help="column with the RUL observed after the alarm (or the replacement lead time if censored)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rulcost", description=__doc__)
    ap.add_argument("--version", action="version", version=f"rulcost {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("calibrate", help="learn the replacement multiplier from completed units and plan new ones")
    s.add_argument("csv"); _common(s); _columns(s)
    s.add_argument("--censored", help="optional 0/1 column: 1 if the unit was replaced preventively (observed is then a lower bound)")
    s.add_argument("--group", help="optional column defining strata (a separate calibration per value)")
    s.add_argument("--window", type=int, help="use only the last W units (rolling calibration)")
    s.add_argument("--method", default="auto", choices=["auto", "conformal", "kaplan-meier", "gaussian"])
    s.add_argument("--predict", type=float, nargs="*", default=[], help="point predictions for which to print the replacement lead time")
    s.add_argument("--predict-group", help="stratum of the --predict values")

    s = sub.add_parser("compare", help="replay the alarm stream under every policy (point, safety factors, fixed 90%%, Gauss, tau* ...)")
    s.add_argument("csv"); _common(s); _columns(s)
    s.add_argument("--n-init", type=int, default=20); s.add_argument("--window", type=int, default=40)
    s.add_argument("--n-orders", type=int, default=150); s.add_argument("--normalizer", type=float, help="divide costs by this (default: mean prediction)")
    s.add_argument("--censoring", action="store_true", help="also run the closed-loop censoring comparison (oracle, naive, drop, Kaplan-Meier)")

    s = sub.add_parser("plan", help="how many completed units are needed for an expected excess cost below a tolerance")
    _common(s)
    s.add_argument("--tolerance", type=float, required=True, help="acceptable expected excess cost, in units of the point prediction")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--density", type=float, help="density of the ratio R/p at its tau*-quantile")
    g.add_argument("--csv", help="estimate the density from completed units in this file")
    s.add_argument("--predicted", default="predicted_rul"); s.add_argument("--observed", default="true_rul")

    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        p, o, c, g = _read(a.csv, a.predicted, a.observed, a.censored, a.group)
        cal = CostAwareCalibrator(a.c_under, a.c_over, window=a.window, method=a.method).fit(p, o, c, g)
        print(cal)
        print(f"completed units: {len(p)} ({int(c.sum())} censored); tau* = {cal.tau:.4f}; target coverage 1 - tau* = {cal.target_coverage:.4f}")
        for grp in sorted({x for x in g}, key=str):
            print(f"multiplier q{'' if grp is None else f' [{grp}]'} = {cal.multiplier(grp):.4f}  (labels used: {cal.n_labels(grp)})")
        for v in a.predict:
            grp = a.predict_group
            print(f"point prediction {v:g} -> replace after {cal.replacement_lead_time(v, grp):.2f}")
    elif a.cmd == "compare":
        p, o, _, _ = _read(a.csv, a.predicted, a.observed)
        print(evaluate_policies(p, o, a.c_under, a.c_over, n_init=a.n_init, window=a.window, n_orders=a.n_orders, normalizer=a.normalizer))
        if a.censoring:
            res = simulate_closed_loop(p, o, a.c_under, a.c_over, n_init=a.n_init, normalizer=a.normalizer)
            print("\nclosed loop with preventive-replacement censoring:")
            for m, v in res.items():
                print(f"  {m:<14} cost {v['cost']:.3f}  coverage {v['coverage']:.3f}")
    else:
        if a.csv:
            p, o, _, _ = _read(a.csv, a.predicted, a.observed)
            tau = a.c_over / (a.c_under + a.c_over)
            dens = density_at_quantile(o / p, tau)
            print(f"estimated density of R/p at its tau*-quantile: {dens:.3f} (from {len(p)} units)")
        else:
            dens = a.density
        n = required_calibration_size(a.c_under, a.c_over, dens, a.tolerance)
        print(f"excess cost ~ {excess_cost_constant(a.c_under, a.c_over, dens):.4f} / n; "
              f"n >= {n} completed units for an expected excess cost <= {a.tolerance:g} (in units of the point prediction)")
        tau = a.c_over / (a.c_under + a.c_over)
        print(f"expected coverage with n = {n}: {expected_coverage(n, tau):.4f} (target {1 - tau:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
