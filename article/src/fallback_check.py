"""When is the closed-loop Kaplan-Meier policy safe?  Fallback multiplier M, censored history, and q* >= 1.

Definition 5 replaces at the fallback multiplier M (M = 1: the point prediction) whenever the Kaplan-Meier estimate has not reached the level t_n.  Theorem 2 holds for any M with G(M) > tau*
(for M = 1 this is G(1) > tau*, i.e. q* < 1).  Here the policy is simulated on log-normal ratios:
  A. well-predicted fleet  rho ~ LN(0, 0.35),   c_u/c_o = 10 (tau* = 1/11, q* = 0.627, G(1) = 0.50);
  B. as A, but the initial records are the outcomes of earlier preventive replacements at the multiplier 0.40, so that almost all are censored;
  C. under-predicted fleet rho ~ LN(0.6, 0.35), c_u/c_o = 10 (q* = 1.14 > 1, G(1) = 0.043 < tau*), 20 exact initial labels;
  D. as C with the censored history of B.
Each case is run with M = 1 and M = 3.  Reported over 300 runs of 1500 units: the mean multiplier over units 1000-1500, the share of units replaced at the fallback, and the mean excess cost per unit
over units 21-1500, in units of c_o times the predicted lead time (exact, from Proposition 1).
usage: python fallback_check.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math
import numpy as np
from rulcost import kaplan_meier_quantile

CU = 10; TAU = 1 / (1 + CU); KAP = CU + 1.0; N, N0, REPS = 1500, 20, 300
erf = np.vectorize(math.erf)
GRID = np.linspace(1e-3, 8.0, 80001)


def make(mu, sig=0.35):
    G = 0.5 * (1 + erf((np.log(GRID) - mu) / (sig * math.sqrt(2))))
    IG = np.concatenate([[0], np.cumsum((G[1:] + G[:-1]) / 2 * np.diff(GRID))])
    qs = float(np.interp(TAU, G, GRID)); I = lambda q: np.interp(q, GRID, IG) - TAU * q
    return qs, (lambda q: KAP * (I(q) - I(qs))), float(np.interp(1.0, GRID, G))


def run(mu, sig, M, hist_q, rng, exc):
    rho = rng.lognormal(mu, sig, N); Y = np.empty(N); d = np.zeros(N, bool); q = np.zeros(N)
    for s in range(N):
        if s < N0:
            if hist_q is None: Y[s], d[s] = rho[s], True                      # exact initial label
            else: Y[s], d[s] = min(rho[s], hist_q), rho[s] <= hist_q           # earlier preventive replacement at hist_q
            q[s] = np.nan
        else:
            q[s] = kaplan_meier_quantile(Y[:s], d[:s], TAU, fallback=M)
            Y[s], d[s] = min(rho[s], q[s]), rho[s] <= q[s]
    qq = q[N0:]; return qq[-500:].mean(), float(np.mean(qq == M)), float(np.mean(exc(qq)))


if __name__ == "__main__":
    cases = [("A  well predicted, exact initial labels", 0.0, None), ("B  well predicted, censored history (0.40)", 0.0, 0.40),
             ("C  under-predicted (q*>1), exact initial labels", 0.6, None), ("D  under-predicted (q*>1), censored history (0.40)", 0.6, 0.40)]
    for name, mu, hq in cases:
        qs, exc, g1 = make(mu)
        print(f"\n{name}: q* = {qs:.3f}, G(1) = {g1:.3f}, tau* = {TAU:.3f}")
        for M in (1.0, 3.0):
            rng = np.random.default_rng(2024); r = np.array([run(mu, 0.35, M, hq, rng, exc) for _ in range(REPS)])
            print(f"   M = {M:.0f}: mean multiplier (units 1000-1500) {r[:, 0].mean():.3f} (sd {r[:, 0].std():.3f}), share at fallback {r[:, 1].mean():.3f}, mean excess cost {r[:, 2].mean():.4f}")
