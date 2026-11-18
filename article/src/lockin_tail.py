"""Empirical tail of the time of the last early step (companion of Theorem 2(e)): P(q_{n+1} < x for some n >= N) from adversarial initial labels.
Log-normal(0, 0.35) ratio, c_u/c_o = 10 (tau* = 1/11, q* = 0.627), x = q* - 0.10, fallback M = 1; the first N0 units are run to failure and their labels are all equal to a value far below q*.
usage: python lockin_tail.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np
from rulcost import kaplan_meier_quantile
TAU, N, REPS = 1 / 11, 800, 1000
rng = np.random.default_rng(77); qs = float(np.exp(0.35 * -1.3352)); x = qs - 0.10
for N0, lab in ((5, 0.40), (5, 0.20), (20, 0.40)):
    last = []
    for _ in range(REPS):
        rho = rng.lognormal(0, 0.35, N); Y = np.empty(N); d = np.zeros(N, bool); T = 0
        for s in range(N):
            if s < N0: Y[s], d[s] = lab, True
            else:
                q = kaplan_meier_quantile(Y[:s], d[:s], TAU); Y[s], d[s] = min(rho[s], q), rho[s] <= q
                if q < x: T = s
        last.append(T)
    last = np.array(last)
    print(f"N0 = {N0:2d}, initial labels all {lab:.2f}: P(early step at some n >= N) for N = 25, 50, 100, 200, 400:", "  ".join(f"{np.mean(last >= n):.3f}" for n in (25, 50, 100, 200, 400)), f"| median last early step {np.median(last):.0f}, max {last.max()}")
