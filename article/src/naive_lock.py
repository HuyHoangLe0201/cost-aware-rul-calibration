"""Absorbing lock-in of the naive treatments (companion of Proposition 3(c)) against the Kaplan-Meier policy.
Log-normal(0, 0.35) ratio, c_u/c_o = 10 (tau* = 1/11, q* = 0.627), N0 = 20 exact initial labels, 600 units per run, 4000 runs.
For a level qbar the policy is 'locked at qbar' if every multiplier q_s, s > N0, is <= qbar.  Proposition 3(c): for the two naive treatments
P(locked at qbar) = P(Bin(N0, G(qbar)) >= ceil((N0 + 1) tau*)); more precisely every multiplier of either naive treatment is at most the initial conformal
quantile q0 = rho_(k) of the N0 exact labels, which is checked run by run; for the Kaplan-Meier policy the probability of a late early step vanishes (Theorem 2).
usage: python naive_lock.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math, numpy as np
from statistics import NormalDist
from rulcost import kaplan_meier_quantile, conformal_quantile
SIG, TAU, N0, N, REPS = 0.35, 1 / 11, 20, 600, 4000
rng = np.random.default_rng(31); nd = NormalDist(); G = lambda x: nd.cdf(math.log(x) / SIG)
k0 = math.ceil((N0 + 1) * TAU)
def binom_tail(n, p, k): return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))
QB = (0.30, 0.40, 0.50, 0.55)
lock = {m: np.zeros(len(QB)) for m in ("naive (record as exact)", "drop censored", "Kaplan-Meier")}
viol = {m: 0 for m in lock}; below = 0; qs_star = math.exp(SIG * nd.inv_cdf(TAU)); tail = {m: [] for m in lock}
for _ in range(REPS):
    rho = rng.lognormal(0, SIG, N); q0 = float(np.sort(rho[:N0])[k0 - 1]); below += q0 < qs_star
    for m in lock:
        Y = list(rho[:N0]); d = [True] * N0; mx = 0.0; late_mx = 0.0
        for s in range(N0, N):
            if m == "naive (record as exact)": q = conformal_quantile(np.array(Y), TAU)
            elif m == "drop censored": q = conformal_quantile(np.array([y for y, e in zip(Y, d) if e]), TAU)
            else: q = kaplan_meier_quantile(np.array(Y), np.array(d), TAU)
            mx = max(mx, q)
            if s >= N - 100: tail[m].append(q)
            if rho[s] <= q: Y.append(rho[s]); d.append(True)
            else:
                if m == "naive (record as exact)": Y.append(q); d.append(True)
                elif m == "Kaplan-Meier": Y.append(q); d.append(False)
        for i, qb in enumerate(QB): lock[m][i] += mx <= qb
        if m != "Kaplan-Meier": viol[m] += mx > q0 + 1e-12
print("level qbar:                       " + "   ".join(f"{qb:5.2f}" for qb in QB))
print("lower bound P(Bin(20,G(qbar))>=k): " + "   ".join(f"{binom_tail(N0, G(qb), k0):5.3f}" for qb in QB) + f"   (k = {k0})")
for m, v in lock.items(): print(f"{m:34s} " + "   ".join(f"{x / REPS:5.3f}" for x in v) + "   share of runs with all multipliers <= qbar")
print(f"runs in which some multiplier of a naive loop exceeds the initial conformal quantile q0: {viol['naive (record as exact)']} (record), {viol['drop censored']} (drop) of {REPS}")
print(f"share of runs with q0 < q* = {qs_star:.3f}: {below / REPS:.3f}   (exact P(Bin(N0, tau*) >= k) = {binom_tail(N0, TAU, k0):.3f})")
print("mean multiplier over the last 100 units: " + ", ".join(f"{m} {np.mean(v):.3f}" for m, v in tail.items()) + f"   (q* = {qs_star:.3f})")
