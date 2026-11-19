"""Randomised checks on piecewise-uniform ratio distributions (with flat parts): Proposition 2(a),(b); Theorem 1: exact form against Monte Carlo and the stated bound."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math, numpy as np
rng = np.random.default_rng(12345)

def make_dist(flat_prob=0.2, K=None):
    K = K or int(rng.integers(3, 9)); xs = np.concatenate([[0.0], np.sort(rng.uniform(0.05, 3.0, K - 1)), [rng.uniform(3.0, 4.0)]]); xs = np.unique(xs)
    K = len(xs) - 1; m = rng.dirichlet(np.ones(K)); flat = rng.random(K) < flat_prob; flat[0] = False; flat[-1] = False
    m = np.where(flat, 0.0, m); m = m / m.sum(); w = np.diff(xs); dens = m / w
    G = np.concatenate([[0], np.cumsum(m)]); return xs, w, m, dens, G

def Gfun(d, x):
    xs, w, m, dens, G = d; return np.interp(x, xs, G)
def Qfun(d, v):                       # Q(v) = min{x: G(x) >= v}, left-continuous (vectorised)
    xs, w, m, dens, G = d; v = np.atleast_1d(np.asarray(v, float)); i = np.searchsorted(G, v, side="left"); i = np.clip(i, 1, len(G) - 1)
    dd = np.where(m[i - 1] > 0, dens[i - 1], 1.0); return np.where(m[i - 1] > 0, xs[i - 1] + (v - G[i - 1]) / dd, xs[i - 1])
def intG(d, q):                      # int_0^q G(x) dx (exact for piecewise linear G), vectorised in q
    xs, w, m, dens, G = d; q = np.atleast_1d(np.asarray(q, float)); tot = np.zeros_like(q)
    for i in range(len(w)):
        b = np.clip(q, xs[i], xs[i + 1]); h = b - xs[i]; tot += (G[i] + (G[i] + dens[i] * h)) / 2 * h
        tot += np.where(q > xs[i + 1], 0.0, 0.0)
    return tot if tot.size > 1 else float(tot[0])
def mean(d): xs, w, m, dens, G = d; return float(np.sum(m * (xs[:-1] + xs[1:]) / 2))
def m2(d): xs, w, m, dens, G = d; return float(np.sum(m * (xs[:-1] ** 2 + xs[:-1] * xs[1:] + xs[1:] ** 2) / 3))
def Phi(d, q, cu, co=1.0):          # c_u E(q-rho)^+ + c_o E(rho-q)^+, with E(q-rho)^+ = int_0^q G
    I = intG(d, q); return cu * I + co * (mean(d) - q + I)

nA = nB = nBad = 0; worstA = 0.0; worst_ratio_lo = 1e9; worst_ratio_hi = 0.0
for trial in range(150):
    d = make_dist(); cu = float(rng.uniform(1, 30)); co = 1.0; kap = cu + co; tau = co / kap; qs = float(Qfun(d, tau)[0])
    for _ in range(10):
        q = float(rng.uniform(0.0, d[0][-1])); u = float(Gfun(d, q))
        vs = np.linspace(tau, u, 400001); vm = (vs[1:] + vs[:-1]) / 2; integ = kap * float(np.sum(q - Qfun(d, vm)) * (vs[1] - vs[0])) if abs(u - tau) > 1e-12 else 0.0
        ex = Phi(d, q, cu) - Phi(d, qs, cu); worstA = max(worstA, abs(integ - ex)); nA += 1
        # (b): density on the interval between q* and q
        lo, hi = min(q, qs), max(q, qs)
        if hi - lo < 1e-9: continue
        idx = [i for i in range(len(d[2])) if d[0][i + 1] > lo + 1e-12 and d[0][i] < hi - 1e-12]; dd = d[3][idx]
        if len(dd) == 0 or dd.min() <= 0: continue
        gmin, gmax = dd.min(), dd.max(); D = abs(u - tau); nB += 1
        if not (kap * D ** 2 / (2 * gmax) - 1e-9 <= ex <= kap * D ** 2 / (2 * gmin) + 1e-9): nBad += 1
        if D > 1e-6: worst_ratio_lo = min(worst_ratio_lo, ex / (kap * D ** 2 / (2 * gmax))); worst_ratio_hi = max(worst_ratio_hi, ex / (kap * D ** 2 / (2 * gmin)))
print(f"Proposition 2(a): {nA} random (distribution, q) pairs with flat parts; largest |oriented integral - exact excess| = {worstA:.2e}")
print(f"Proposition 2(b): {nB} pairs with positive density on the interval; violations: {nBad}; min excess/lower bound {worst_ratio_lo:.3f} (>= 1), max excess/upper bound {worst_ratio_hi:.3f} (<= 1)")

def bcdf(kk, n, pr):
    pr = np.clip(np.asarray(pr, float), 1e-300, 1 - 1e-16); tot = np.zeros_like(pr)
    for j in range(kk + 1): tot += np.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * np.log(pr) + (n - j) * np.log1p(-pr))
    return np.minimum(tot, 1.0)
def Dkl(a, b):
    a = min(max(a, 1e-15), 1 - 1e-15); return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
def exact_excess(d, cu, n):
    xs, w, m, dens, G = d; co = 1.0; kap = cu + co; tau = co / kap; k = int(math.ceil((n + 1) * tau)); tot = 0.0
    T = lambda v: np.where(v > tau, bcdf(k - 1, n, v), 1 - bcdf(k - 1, n, v))
    for i in range(len(w)):
        if m[i] > 0:
            v = np.linspace(G[i], G[i + 1], 4001); tot += np.trapezoid(np.abs(v - tau) * T(v), v) / dens[i]
        else:                        # flat part of G: atom of dQ of mass w[i] at v = G[i]
            vv = G[i]; tot += abs(vv - tau) * float(T(np.array([vv]))[0]) * w[i]
    return kap * tot
def mc_excess(d, cu, n, reps=20000):
    xs, w, m, dens, G = d; k = int(math.ceil((n + 1) / (cu + 1.0))); co = 1.0; kap = cu + co; tau = co / kap; k = int(math.ceil((n + 1) * tau))
    seg = rng.choice(len(w), size=(reps, n), p=m); x = xs[seg] + rng.random((reps, n)) * w[seg]; q = np.sort(x, axis=1)[:, k - 1]
    qs = float(Qfun(d, tau)[0]); I = intG(d, q); ph = cu * I + co * (mean(d) - q + I); e = ph - Phi(d, qs, cu); return float(e.mean()), float(e.std() / math.sqrt(len(e)))
worst_mc = 0.0; nT = nTbad = 0; ratios = []
for trial in range(40):
    d = make_dist(flat_prob=0.25); cu = float(rng.choice([2.0, 5.0, 10.0])); n = int(rng.choice([30, 60, 120])); ex = exact_excess(d, cu, n); mc, se = mc_excess(d, cu, n, reps=40000)
    worst_mc = max(worst_mc, abs(ex - mc) / max(se, 1e-12))
print(f"Theorem 1, exact form with atoms of dQ (flat parts of G): 40 random cases; largest |exact - Monte Carlo| in standard errors (40000 samples) {worst_mc:.2f}")
for trial in range(300):
    d = make_dist(flat_prob=0.1); cu = float(rng.choice([2.0, 5.0, 10.0, 20.0])); co = 1.0; kap = cu + co; tau = co / kap; n = int(rng.choice([60, 150, 400, 1000]))
    k = int(math.ceil((n + 1) * tau)); pb = k / (n + 1); qstar = float(Qfun(d, tau)[0]); mm2 = m2(d); ex = exact_excess(d, cu, n); best = np.inf
    for eta in np.linspace(0.002, min(tau, 1 - tau) * 0.999, 300):
        if n <= 2 / eta: continue
        qm, qp = float(Qfun(d, tau - eta)[0]), float(Qfun(d, tau + eta)[0]); idx = [i for i in range(len(d[2])) if d[0][i + 1] > qm + 1e-12 and d[0][i] < qp - 1e-12]
        if not idx or d[3][idx].min() <= 0: continue
        gmn = d[3][idx].min(); e_lo = math.exp(-n * Dkl(tau, tau - eta)); e_hi = math.exp(-n * Dkl(tau + tau / n, tau + eta))
        best = min(best, kap / (2 * gmn) * (pb * (1 - pb) / (n + 2) + 1 / (n + 1) ** 2) + kap * tau * qstar * e_lo + kap * math.sqrt(2 * (1 - tau) * mm2) * math.sqrt(e_hi))
    if np.isfinite(best):
        nT += 1; ratios.append(best / ex)
        if ex > best * (1 + 1e-9): nTbad += 1
print(f"Theorem 1 bound: {nT} random cases with an admissible window; violations: {nTbad}; bound / exact excess between {min(ratios):.2f} and {max(ratios):.1f} (median {np.median(ratios):.1f})")
