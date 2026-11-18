"""Numerical audit of the statements proved in Appendix A (a second, independent check besides the line-by-line reading of the proofs).
 1. Proposition 1: excess cost by direct expectation against the integral form, for q below and above q*.
 2. Proposition 2: the quadratic sandwich.
 3. Theorem 1: Monte Carlo expected excess of the conformal quantile against the stated bound, over admissible eta, and against the asymptotic constant.
 4. Proposition 3: quantiles of the two naive treatments against q*.
 5. Proposition 4: the stratification threshold against Monte Carlo for two strata.
 6. Lemma 1: conditional variance of the increment xi_s(u, v) against b (G(v) - G(u)) 1{q > u}, for random (u, v, q).
 7. Lemma 2(i): J_n^+(x) <= (2 / p_x) (R_n^+(x) + C1 * 5 log n) along closed-loop runs; smallest C1 that works.
 8. Lemma 3: sup over levels with R_n >= c log n of |G_hat - G| (c = 5, 20).
 9. Lemma 4: E #{A_j(K)} against K / (1 - G(x)).
usage: python math_audit.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math
import numpy as np
from rulcost import kaplan_meier_quantile

rng = np.random.default_rng(2026)
erf = np.vectorize(math.erf)
SIG = 0.35
G = lambda x: 0.5 * (1 + erf(np.log(np.maximum(x, 1e-300)) / (SIG * math.sqrt(2)))) * (np.asarray(x) > 0)
GRID = np.linspace(1e-3, 6, 120001); GG = G(GRID); IG = np.concatenate([[0], np.cumsum((GG[1:] + GG[:-1]) / 2 * np.diff(GRID))])
dens = lambda x: np.exp(-0.5 * (np.log(x) / SIG) ** 2) / (x * SIG * math.sqrt(2 * math.pi))
Ginv = lambda p: float(np.interp(p, GG, GRID))
M2 = math.exp(2 * SIG ** 2)                                       # E rho^2 of LN(0, SIG)


def Phi(q, cu):                                                    # exact: cu E(q-rho)^+ + E(rho-q)^+
    iq = np.interp(q, GRID, IG)
    EQ = iq                                                        # E (q - rho)^+ = int_0^q G
    ER = math.exp(SIG ** 2 / 2) - q + EQ                           # E (rho - q)^+ = E rho - q + E (q - rho)^+
    return cu * EQ + ER


print("1. Proposition 1")
for cu in (2, 20):
    tau = 1 / (1 + cu); kap = cu + 1; qs = Ginv(tau)
    for q in (0.4 * qs, 0.9 * qs, 1.2 * qs, 2.0 * qs):
        direct = Phi(q, cu) - Phi(qs, cu); xs = np.linspace(qs, q, 20001); integ = kap * np.trapezoid(G(xs) - tau, xs)
        print(f"   cu/co={cu:2d} q/q*={q / qs:4.2f}: direct {direct:.5f}  signed integral {integ:.5f}  (non-negative: {direct >= -1e-9})")

print("2. Proposition 2 (sandwich), cu/co = 10")
cu = 10; tau = 1 / 11; kap = 11; qs = Ginv(tau)
for q in (0.8 * qs, 0.95 * qs, 1.1 * qs, 1.3 * qs):
    lo_hi = np.linspace(min(q, qs), max(q, qs), 2001); gmin, gmax = dens(lo_hi).min(), dens(lo_hi).max(); D = abs(float(G(q)) - tau)
    ex = Phi(q, cu) - Phi(qs, cu)
    print(f"   q/q*={q / qs:4.2f}: lower {kap * gmin * D ** 2 / (2 * gmax ** 2):.5f} <= excess {ex:.5f} <= upper {kap * gmax * D ** 2 / (2 * gmin ** 2):.5f}: {kap * gmin * D ** 2 / (2 * gmax ** 2) - 1e-12 <= ex <= kap * gmax * D ** 2 / (2 * gmin ** 2) + 1e-12}")

print("3. Theorem 1 (log-normal 0.35): Monte Carlo against the stated bound (best admissible eta) and the asymptotic constant")
def Dkl(a, b):
    a = min(max(a, 1e-15), 1 - 1e-15); return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
for cu in (2, 5, 10, 20):
    tau = 1 / (1 + cu); kap = cu + 1; qs = Ginv(tau); const = kap * tau * (1 - tau) / (2 * float(dens(qs)))
    for n in (50, 100, 400, 1000):
        k = min(max(int(math.ceil((n + 1) * tau)), 1), n); pb = k / (n + 1); X = np.sort(rng.lognormal(0, SIG, (20000, n)), axis=1)[:, k - 1]
        ex = float((kap * (np.interp(X, GRID, IG) - tau * X - (np.interp(qs, GRID, IG) - tau * qs))).mean()); best = np.inf
        for eta in np.linspace(0.001, min(tau, 1 - tau) * 0.999, 600):
            if n <= 2 / eta: continue
            qm, qp = Ginv(tau - eta), Ginv(tau + eta); xs = np.linspace(qm, qp, 2001); gmn, gmx = dens(xs).min(), dens(xs).max()
            first = kap * gmx / (2 * gmn ** 2) * (pb * (1 - pb) / (n + 2) + 1 / (n + 1) ** 2)
            pe = math.exp(-n * Dkl(tau, tau - eta)) + math.exp(-n * Dkl(tau + tau / n, tau + eta))
            best = min(best, first + kap * math.sqrt(4 * M2 / (1 - tau) + 2 * qs ** 2) * math.sqrt(pe))
        print(f"   cu/co={cu:2d} n={n:4d}: MC excess {ex:.5f}  bound {best:.5f} (ratio {best / ex:7.1f})  holds: {ex <= best}   n*excess {n * ex:.3f} vs asymptotic {const:.3f}")

print("4. Proposition 3: tau*-quantile after naive treatment (cu/co = 10, q_bar = 1.3 q*)")
tau = 1 / 11; qs = Ginv(tau); qb = 1.3 * qs; Gb = float(G(qb))
q_drop = Ginv(tau * Gb); print(f"   q* = {qs:.4f}; dropped censored: quantile of G/G(q_bar) at tau* = {q_drop:.4f} (<= q*: {q_drop <= qs}); censored value recorded as exact: quantile = {min(qs, qb):.4f} (<= q*: True by construction)")

print("5. Proposition 4: two strata (log-normal mu = -0.25 and +0.25, sigma 0.2), cu/co = 10; n = 200, 1000")
cu = 10; tau = 1 / 11; kap = 11; s2 = 0.2
def cdf(mu): return lambda x: 0.5 * (1 + erf((np.log(np.maximum(x, 1e-300)) - mu) / (s2 * math.sqrt(2))))
mus = (-0.25, 0.25); w = (0.5, 0.5)
xg = np.linspace(1e-3, 5, 200001); Gj = [cdf(m)(xg) for m in mus]; Gp = sum(wi * g for wi, g in zip(w, Gj))
cum = lambda F: np.concatenate([[0], np.cumsum((F[1:] + F[:-1]) / 2 * np.diff(xg))])
IJ = [cum(F) - tau * xg for F in Gj]; QJ = [float(np.interp(tau, F, xg)) for F in Gj]
def mean_cost_exc(q, F_list=None):                                 # excess over each stratum's own optimum, weighted
    return sum(wi * kap * (np.interp(q, xg, I) - np.interp(qj, xg, I)) for wi, I, qj in zip(w, IJ, QJ))
V = min(mean_cost_exc(q) for q in np.linspace(0.3, 2.5, 4401)); pdf = lambda mu: (lambda x: np.exp(-0.5 * ((np.log(x) - mu) / s2) ** 2) / (x * s2 * math.sqrt(2 * math.pi)))
qj = [float(np.interp(tau, F, xg)) for F in Gj]; qp = float(np.interp(tau, Gp, xg)); gj = [float(pdf(m)(q)) for m, q in zip(mus, qj)]; gp = sum(wi * float(pdf(m)(qp)) for wi, m in zip(w, mus))
for n in (200, 1000, 5000):
    thr = kap * tau * (1 - tau) / (2 * n) * (sum(1 / g for g in gj) - 1 / gp)
    # Monte Carlo of the two policies' total excess
    reps = 4000; exs, exp_ = [], []
    for _ in range(reps):
        sam = [np.exp(rng.normal(m, s2, int(wi * n))) for m, wi in zip(mus, w)]
        qh = [np.sort(x)[min(max(int(math.ceil((len(x) + 1) * tau)), 1), len(x)) - 1] for x in sam]
        exs.append(sum(wi * kap * (np.interp(q, xg, I) - np.interp(qj, xg, I)) for wi, q, I, qj in zip(w, qh, IJ, QJ)))
        pooled = np.sort(np.concatenate(sam)); qpool = pooled[min(max(int(math.ceil((len(pooled) + 1) * tau)), 1), len(pooled)) - 1]; exp_.append(mean_cost_exc(qpool))
        if len(exs) >= 400: break
    print(f"   n={n:4d}: V = {V:.4f}; threshold {thr:.5f}; stratified excess {np.mean(exs):.5f}; pooled excess {np.mean(exp_):.5f}; stratifying pays: {np.mean(exs) < np.mean(exp_)} (rule V > threshold: {V > thr})")

print("6. Lemma 1: conditional variance of xi(u, v) against b (G(v) - G(u)) 1{q > u}, x = 1.2")
x = 1.2; Lam = lambda z: -np.log(1 - G(z)); b = 1 + float(Lam(x)); worst = 0
for _ in range(60):
    u = rng.uniform(0, 1.0); v = rng.uniform(u, x); q = rng.uniform(0.05, 2.0); rho = rng.lognormal(0, SIG, 200000)
    Y = np.minimum(rho, q); d = rho <= q; A = d & (Y > u) & (Y <= v)
    # int_{(u,v]} 1{Y >= w} dLambda(w) = Lam(min(Y, v)) - Lam(u) when Y > u (closed at Y), else 0
    B = np.where(Y > u, Lam(np.minimum(Y, v)) - float(Lam(u)), 0.0); xi = A.astype(float) - B
    var = xi.var(); bound = b * float(G(v) - G(u)) * (q > u)
    worst = max(worst, var / bound if bound > 0 else (0 if var < 1e-12 else np.inf))
print(f"   largest ratio variance / bound over 60 random (u, v, q): {worst:.3f}  (must be <= 1)")

print("7-9. Lemmas 2(i), 3, 4 along closed-loop runs (tau* = 1/11, N0 = 20, n up to 3000, 40 runs)")
tau = 1 / 11; qs = Ginv(tau); xlev = 0.95 * qs; px = 1 - float(G(xlev)); c1_needed = 0; err = {5: 0, 20: 0}; lemma4 = {5: [], 20: []}
for run in range(40):
    N = 3000; N0 = 20; rho = rng.lognormal(0, SIG, N); Y = np.empty(N); d = np.zeros(N, bool); q = np.full(N, np.inf)
    for s in range(N):
        if s >= N0: q[s] = kaplan_meier_quantile(Y[:s], d[:s], tau)
        Y[s] = min(rho[s], q[s]); d[s] = rho[s] <= q[s]
        if s in (200, 500, 1000, 2000, 2999):
            n = s + 1; ell = 5 * math.log(n); Jp = np.sum(q[:n] > xlev); Rp = np.sum(Y[:n] > xlev)
            c1_needed = max(c1_needed, (Jp * px / 2 - Rp) / ell)
            # Lemma 3: Nelson-Aalen / KM error at levels with R_n(y) >= c log n
            ex = np.sort(Y[:n][d[:n]]); Rn = np.array([np.sum(Y[:n] >= y) for y in ex]); Ghat = 1 - np.cumprod(1 - 1 / Rn)
            for c in (5, 20):
                ok = (Rn >= c * math.log(n)) & (ex <= xlev)
                if ok.any(): err[c] = max(err[c], float(np.max(np.abs(Ghat[ok] - G(ex[ok])))))
    # Lemma 4: labels created at low rank
    for K in (5, 20):
        cnt = 0; top = []
        for j in range(N):
            if j >= N0 and d[j]:
                kth = sorted(top, reverse=True)[K - 1] if len(top) >= K else 0.0
                if kth < rho[j] <= min(q[j], xlev): cnt += 1
            top.append(Y[j]); top = sorted(top, reverse=True)[:K]
        lemma4[K].append(cnt)
print(f"   Lemma 2(i): smallest C1 with J^+ <= (2/p_x)(R^+ + C1 * 5 log n) needed over all checks: {max(c1_needed, 0):.3f} (negative values mean the inequality held with C1 = 0)")
print(f"   Lemma 3: largest |G_hat - G| at levels with R_n >= 5 log n: {err[5]:.3f}; >= 20 log n: {err[20]:.3f}")
print(f"   Lemma 4: mean count {np.mean(lemma4[5]):.2f} (K=5, bound {5 / px:.1f}), {np.mean(lemma4[20]):.2f} (K=20, bound {20 / px:.1f})")
