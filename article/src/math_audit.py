"""Numerical audit of the statements proved in Appendix A (a second, independent check besides the line-by-line reading of the proofs).
 1. Proposition 1: excess cost by direct expectation against the integral form, for q below and above q*.
 2. Proposition 2: the exact form kappa int_tau^u (q - Q(v)) dv and the sharpened sandwich kappa Delta^2/(2 g_max) <= excess <= kappa Delta^2/(2 g_min).
 3. Theorem 1: Monte Carlo expected excess of the conformal quantile against the stated bound, over admissible eta, and against the asymptotic constant; the exact form kappa int |v - tau| T_n(v) dQ(v).
 4. Proposition 3: quantiles of the two naive treatments against q*; part (c), the cap of every later multiplier by the initial conformal quantile and the exact lock-in probability.
 5. Proposition 4: the stratification threshold against Monte Carlo for two strata.
 6. Lemma 1: conditional variance of the increment chi_s(u, v) against b (G(v) - G(u)) 1{q > u}, for random (u, v, q).
 7. Lemma 2(i): J_n^+(x) <= (2 / p_x) (R_n^+(x) + C1 * 5 log n) along closed-loop runs; smallest C1 that works.
 8. Lemma 3: sup over levels with R_n >= c log n of |G_hat - G| (c = 5, 20).
 9. Lemma 4: E #{A_j(K)} against K / (1 - G(x)).
usage: python math_audit.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math
import numpy as np
from rulcost import kaplan_meier_quantile, conformal_quantile

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

print("2. Proposition 2 (sharpened sandwich kappa Delta^2/(2 g_max) <= excess <= kappa Delta^2/(2 g_min)), cu/co = 10")
cu = 10; tau = 1 / 11; kap = 11; qs = Ginv(tau)
for q in (0.8 * qs, 0.95 * qs, 1.1 * qs, 1.3 * qs):
    lo_hi = np.linspace(min(q, qs), max(q, qs), 2001); gmin, gmax = dens(lo_hi).min(), dens(lo_hi).max(); D = abs(float(G(q)) - tau)
    ex = Phi(q, cu) - Phi(qs, cu)
    print(f"   q/q*={q / qs:4.2f}: lower {kap * D ** 2 / (2 * gmax):.5f} <= excess {ex:.5f} <= upper {kap * D ** 2 / (2 * gmin):.5f}: {kap * D ** 2 / (2 * gmax) - 1e-12 <= ex <= kap * D ** 2 / (2 * gmin) + 1e-12}")

print("   Proposition 2(a): excess = kappa int_tau^u (q - Q(v)) dv (oriented), cu/co = 10")
for q in (0.6 * qs, 0.9 * qs, 1.2 * qs, 1.6 * qs):
    u = float(G(q)); vs = np.linspace(tau, u, 40001); Qv = np.interp(vs, GG, GRID); integ = kap * np.trapezoid(q - Qv, vs)
    print(f"   q/q*={q / qs:4.2f}: oriented integral {integ:.6f}  direct {Phi(q, cu) - Phi(qs, cu):.6f}")

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
            first = kap / (2 * gmn) * (pb * (1 - pb) / (n + 2) + 1 / (n + 1) ** 2)
            e_lo, e_hi = math.exp(-n * Dkl(tau, tau - eta)), math.exp(-n * Dkl(tau + tau / n, tau + eta))
            best = min(best, first + kap * tau * qs * e_lo + kap * math.sqrt(2 * (1 - tau) * M2) * math.sqrt(e_hi))
        print(f"   cu/co={cu:2d} n={n:4d}: MC excess {ex:.5f}  bound {best:.5f} (ratio {best / ex:7.1f})  holds: {ex <= best}   n*excess {n * ex:.3f} vs asymptotic {const:.3f}")

print("   Theorem 1, exact form: E excess = kappa int |v - tau| T_n(v) Q'(v) dv, T_n = P(Bin(n, v) <= k-1) (v > tau), P(Bin(n, v) >= k) (v <= tau); cu/co = 10")
def bcdf(kk, n, pr):                                              # P(Bin(n, pr) <= kk), vectorised in pr
    pr = np.clip(np.asarray(pr, float), 1e-300, 1 - 1e-16); tot = np.zeros_like(pr)
    for j in range(kk + 1): tot += np.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * np.log(pr) + (n - j) * np.log1p(-pr))
    return np.minimum(tot, 1.0)
cu = 10; tau = 1 / 11; kap = 11; qs = Ginv(tau)
vv = np.linspace(1e-6, 1 - 1e-6, 400001); Qv = np.interp(vv, GG, GRID); Qp = 1 / dens(Qv)
for n in (50, 100, 400):
    k = int(math.ceil((n + 1) * tau)); T = np.where(vv > tau, bcdf(k - 1, n, vv), 1 - bcdf(k - 1, n, vv))
    exact = kap * np.trapezoid(np.abs(vv - tau) * T * Qp, vv)
    X = np.sort(rng.lognormal(0, SIG, (20000, n)), axis=1)[:, k - 1]; mc = float((kap * (np.interp(X, GRID, IG) - tau * X - (np.interp(qs, GRID, IG) - tau * qs))).mean())
    print(f"   n={n:4d}: exact form {exact:.6f}   Monte Carlo {mc:.6f}")

print("4. Proposition 3 (corrected): pooled quantiles after the two naive treatments, cu/co = 10, multipliers q_s drawn around q*")
tau = 1 / 11; qs = Ginv(tau)
for name, qlaw in (("single multiplier 1.3 q*", lambda n: np.full(n, 1.3 * qs)), ("single multiplier 0.8 q*", lambda n: np.full(n, 0.8 * qs)), ("multipliers uniform in [0.7 q*, 1.3 q*]", lambda n: rng.uniform(0.7 * qs, 1.3 * qs, n)), ("multipliers all >= q* (uniform in [q*, 1.5 q*])", lambda n: rng.uniform(qs, 1.5 * qs, n))):
    n = 400000; rho = rng.lognormal(0, SIG, n); q = qlaw(n); keep = rho <= q
    q_drop = np.quantile(rho[keep], tau); q_rec = np.quantile(np.minimum(rho, q), tau)
    print(f"   {name:48s}: q* = {qs:.4f}; discard: {q_drop:.4f} (< q*: {q_drop < qs - 1e-3}); record as exact: {q_rec:.4f} ({'below q*' if q_rec < qs - 1e-3 else 'equal to q*'})")

print("   Proposition 3(c): every multiplier of a naive closed loop is at most the initial conformal quantile; lock-in probability P(Bin(N0, G(qbar)) >= k), cu/co = 10, N0 = 20")
N0, NN, RUNS = 20, 150, 1500; k0 = int(math.ceil((N0 + 1) * tau)); viol = {"record": 0, "drop": 0}; lock = {"record": 0, "drop": 0}; QB = 0.55
for _ in range(RUNS):
    rho = rng.lognormal(0, SIG, NN); q0 = float(np.sort(rho[:N0])[k0 - 1])
    for m in ("record", "drop"):
        Y = list(rho[:N0]); mx = 0.0
        for t in range(N0, NN):
            q = conformal_quantile(np.array(Y), tau); mx = max(mx, q)
            if rho[t] <= q: Y.append(rho[t])
            elif m == "record": Y.append(q)
        viol[m] += mx > q0 + 1e-12; lock[m] += mx <= QB
Gq = float(G(QB)); print(f"   runs with a multiplier above the initial quantile: record {viol['record']}, drop {viol['drop']} of {RUNS}; share of runs with all multipliers <= {QB}: record {lock['record'] / RUNS:.3f}, drop {lock['drop'] / RUNS:.3f}, exact {1 - float(bcdf(k0 - 1, N0, np.array([Gq]))[0]):.3f}")

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

print("6. Lemma 1: conditional variance of chi(u, v) against b (G(v) - G(u)) 1{q > u}, x = 1.2")
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

print("10. Proposition 5: coverage error against eps_w = sqrt(log(2w/delta)/(2w)) + 3/w + s(w+1)/2 (delta = 0.1), drifting log-normal location")
tau = 1 / 11
for rate in (0.0, 0.0005, 0.002):                                      # location drift per step
    # Kolmogorov distance between consecutive log-normal distributions: sup_x |Phi((ln x - mu)/sig) - Phi((ln x - mu - rate)/sig)| = 2 Phi(rate / (2 sig)) - 1
    s_ = 2 * 0.5 * (1 + math.erf(rate / (2 * SIG) / math.sqrt(2))) - 1
    for w in (50, 200):
        eps = math.sqrt(math.log(2 * w / 0.1) / (2 * w)) + 3 / w + s_ * (w + 1) / 2; bad = 0; reps = 4000; t0 = 500
        k = min(max(int(math.ceil((w + 1) * tau)), 1), w)
        for _ in range(reps):
            mus = rate * (t0 - np.arange(w, 0, -1)); X = np.sort(rng.lognormal(mus, SIG)); qh = X[k - 1]
            cov_err = abs(0.5 * (1 + math.erf((math.log(qh) - rate * t0) / (SIG * math.sqrt(2)))) - tau); bad += cov_err > eps
        print(f"   drift {rate:.4f} (s = {s_:.5f}), w = {w:3d}: eps_w = {eps:.4f}; share of windows with error above eps_w: {bad / reps:.4f} (must be <= 0.10)")

print("11. Structure of the proof of Theorem 2(e) along adversarial runs (N0 = 5 labels at 0.40, tau* = 1/11, x = q* - 0.10, 300 runs of 800 units)")
tau = 1 / 11; qs = Ginv(tau); x = qs - 0.10; px = 1 - float(G(x)); maxR = 0; maxJ = 0; maxuse = 0; nearly = 0
for run in range(300):
    N = 800; N0 = 5; rho = rng.lognormal(0, SIG, N); Y = np.empty(N); d = np.zeros(N, bool); q = np.full(N, np.inf); uses = {}
    for s in range(N):
        if s < N0: Y[s], d[s] = 0.40, True; continue
        q[s] = kaplan_meier_quantile(Y[:s], d[:s], tau); Y[s], d[s] = min(rho[s], q[s]), rho[s] <= q[s]
        if q[s] < x and s >= 30:
            n = s; Rn = np.sum(Y[:n] >= q[s]); Jn = np.sum(q[:n] >= x); maxR = max(maxR, Rn / math.log(n)); maxJ = max(maxJ, Jn / math.log(n)); uses[q[s]] = uses.get(q[s], 0) + 1
    if uses: maxuse = max(maxuse, max(uses.values()) / math.log(N))
print(f"   Claim 1: over all early steps with n >= 30, max R_n(y) / log n = {maxR:.2f} and max J_n(x) / log n = {maxJ:.2f} (bounded by constants C and C')")
print(f"   Lemma 5 / Step 4: max number of uses of one label before N = 800, divided by log N: {maxuse:.2f}")

print("12. Explicit constants of Lemmas 2-5 and of Theorem 2(e) for the log-normal example (sigma 0.35, cu/co = 10, x = q* - 0.10, N0 = 5)")
SIGc = 0.35; tau = 1 / 11; N0 = 5
from statistics import NormalDist
qs = math.exp(SIGc * NormalDist().inv_cdf(tau)); x = qs - 0.10
Gx = NormalDist().cdf(math.log(x) / SIGc); p = 1 - Gx; Lam = -math.log(p); b = 1 + Lam; c = max(1, Lam)
C1 = 2 / p + 3; A = 2 * math.sqrt(2 * b * C1 / p); B0 = c + 6; th = (tau - Gx) / 2
C0 = max(5, 1920 * A * A / th ** 2, 80 * B0 / th); n1 = max(5, math.ceil(b), math.ceil(math.exp(N0 * b / 5)))
n2 = max(n1, math.ceil(math.exp(2 * N0 * max(1, (Lam + th / 2) / th) / C0)))
Cp = 2 / p * (C0 + 5 * C1); C4 = 1 + 2 / p * (C0 + 3 / (4 * p)); Gp = Cp + N0 * C4 + 1 + C4 * (3.2 * C0 + 16) / p
def okN(N):
    L = math.log(N); Amax = (3.2 * C0 + 16) * L / p
    return math.ceil(N / 2) >= n2 and N >= Cp and 0.5 * (N + 1) + (N0 + Amax) * C4 * L + Cp * L < N
Nn = 3.0
while not okN(int(Nn)): Nn *= 1.5
lo, hi = Nn / 1.5, Nn
for _ in range(80):
    mid = (lo + hi) / 2
    if okN(int(mid)): hi = mid
    else: lo = mid
print(f"   p = {p:.4f}, Lambda = {Lam:.4f}, b = {b:.4f}, C1 = {C1:.3f}, A = {A:.3f}, B0 = {B0:.2f}, theta = {th:.5f}")
print(f"   C0 = {C0:.3e}, n2 = {n2}, C' = {Cp:.3e}, C4 = {C4:.3e}, Gamma' = {Gp:.3e}; threshold N1 = {hi:.3e}; failure constant 64 b^2 + 3 = {64 * b * b + 3:.1f}")
Lam_f = lambda z: -np.log(1 - G(z)); worst_i = 0; worst_ii = 0
for run in range(40):
    Nn_ = 2500; rho = rng.lognormal(0, SIGc, Nn_); Y = np.empty(Nn_); d = np.zeros(Nn_, bool); qv = np.full(Nn_, np.inf)
    for s_ in range(Nn_):
        if s_ < N0: Y[s_], d[s_] = 0.40, True; continue
        qv[s_] = kaplan_meier_quantile(Y[:s_], d[:s_], tau); Y[s_], d[s_] = min(rho[s_], qv[s_]), rho[s_] <= qv[s_]
        n = s_ + 1
        if n in (100, 300, 1000, 2500):
            ell = 5 * math.log(n); Jp = np.sum(qv[:n] > x); Rp = np.sum(Y[:n] > x); worst_i = max(worst_i, Jp / ((2 / p) * (Rp + C1 * ell)))
            for _ in range(40):
                u = rng.uniform(0, x * 0.98); v = rng.uniform(u + 1e-3, x); Yv = Y[:n]; msk = Yv > u
                Mn = np.sum((Yv > u) & (Yv <= v) & d[:n]) - (np.sum(Lam_f(np.minimum(Yv[msk], v)) - float(Lam_f(u))) if msk.any() else 0.0)
                rhs = A * math.sqrt(ell * float(G(v) - G(u)) * (np.sum(Yv > u) + ell)) + B0 * ell; worst_ii = max(worst_ii, abs(Mn) / rhs)
print(f"   Lemma 2 with the explicit constants, 40 closed-loop runs of 2500 units: largest LHS/RHS = {worst_i:.2f} for (i) and {worst_ii:.2f} for (ii) (must be <= 1)")
# Lemma 4(ii) and Lemma 5 along adversarial runs
bad4 = 0; bad5 = 0; tot5 = 0; N = 800
for run in range(150):
    rho = rng.lognormal(0, SIGc, N); Y = np.empty(N); d = np.zeros(N, bool); qv = np.full(N, np.inf)
    for s in range(N):
        if s < N0: Y[s], d[s] = 0.40, True; continue
        qv[s] = kaplan_meier_quantile(Y[:s], d[:s], tau); Y[s], d[s] = min(rho[s], qv[s]), rho[s] <= qv[s]
    for K in (5, 20):
        cnt = 0; top = []
        for j in range(N):
            if j >= N0 and d[j]:
                kth = top[K - 1] if len(top) >= K else 0.0
                if kth < rho[j] <= min(qv[j], x): cnt += 1
            top.append(Y[j]); top = sorted(top, reverse=True)[:K]
        bad4 += cnt > (3.2 * K + 9.6 * math.log(N)) / p + 3 * math.log(N)
    # Lemma 5: uses of each label below x at steps > n0 = 30: S_u > p u / 2 - 3/(4 p) log N
    n0 = 30; uses = {}
    for s in range(n0, N):
        if qv[s] < x:
            u = uses.setdefault(qv[s], [0, 0]); u[0] += 1; u[1] += (rho[s] > qv[s])
            tot5 += 1; bad5 += (u[1] <= p * u[0] / 2 - 3 / (4 * p) * math.log(N))
print(f"   Lemma 4(ii): {bad4} violations in {150 * 2} (run, K) pairs; Lemma 5: {bad5} violations among {tot5} uses")

print("13. A bounded, non-log-normal ratio rho = 2 Beta(2, 3), cu/co = 10 (tau* = 1/11): Theorem 1 (bound and exact form against Monte Carlo) and Theorem 2 (closed-loop Kaplan-Meier, five unfavourable initial labels)")
_a, _b = 2.0, 3.0; _norm = math.gamma(_a + _b) / (math.gamma(_a) * math.gamma(_b))
_x = np.linspace(0, 2, 400001); _pdf = np.where((_x > 0) & (_x < 2), 0.5 * _norm * (_x / 2) ** (_a - 1) * (1 - _x / 2) ** (_b - 1), 0.0)
_G = np.concatenate([[0], np.cumsum((_pdf[1:] + _pdf[:-1]) / 2 * np.diff(_x))]); _G /= _G[-1]
_IG = np.concatenate([[0], np.cumsum((_G[1:] + _G[:-1]) / 2 * np.diff(_x))]); _m1 = float(np.trapezoid(_x * _pdf, _x)); _m2 = float(np.trapezoid(_x ** 2 * _pdf, _x))
cu = 10; tau = 1 / 11; kap = 11; _Q = lambda p: float(np.interp(p, _G, _x)); qs = _Q(tau)
_Phi = lambda q: cu * np.interp(q, _x, _IG) + (_m1 - q + np.interp(q, _x, _IG))
for n in (50, 100, 400):
    k = int(math.ceil((n + 1) * tau)); pb = k / (n + 1)
    X = np.sort(2 * rng.beta(_a, _b, (20000, n)), axis=1)[:, k - 1]; mc = float((_Phi(X) - _Phi(qs)).mean())
    vv = np.linspace(1e-6, 1 - 1e-6, 200001); Qv = np.interp(vv, _G, _x); dens = lambda y: np.interp(y, _x, _pdf)
    T = np.where(vv > tau, bcdf(k - 1, n, vv), 1 - bcdf(k - 1, n, vv)); exact = kap * np.trapezoid(np.abs(vv - tau) * T / np.maximum(dens(Qv), 1e-12), vv)
    best = np.inf
    for eta in np.linspace(0.001, min(tau, 1 - tau) * 0.999, 400):
        if n <= 2 / eta: continue
        xs = np.linspace(_Q(tau - eta), _Q(tau + eta), 2001); gmn = dens(xs).min()
        e_lo, e_hi = math.exp(-n * Dkl(tau, tau - eta)), math.exp(-n * Dkl(tau + tau / n, tau + eta))
        best = min(best, kap / (2 * gmn) * (pb * (1 - pb) / (n + 2) + 1 / (n + 1) ** 2) + kap * tau * qs * e_lo + kap * math.sqrt(2 * (1 - tau) * _m2) * math.sqrt(e_hi))
    print(f"   n={n:4d}: Monte Carlo excess {mc:.5f}, exact form {exact:.5f}, Theorem 1 bound {best:.5f} (holds: {mc <= best}); n*excess {n * mc:.3f}")
_kc = {"cu": cu}
def _run(adv, N=500):
    rho = 2 * rng.beta(_a, _b, N)
    Y = list(np.full(5, 0.5 * qs)) if adv else list(rho[:20]); d = [True] * len(Y); out = []
    for s in range(len(Y), N):
        q = kaplan_meier_quantile(np.array(Y), np.array(d), tau, fallback=1.0); out.append(q)
        if rho[s] <= q: Y.append(rho[s]); d.append(True)
        else: Y.append(q); d.append(False)
    return np.array(out)
for adv in (False, True):
    R = [_run(adv) for _ in range(100)]; last = [np.max(np.where(r < qs - 0.10)[0]) if np.any(r < qs - 0.10) else -1 for r in R]
    print(f"   Kaplan-Meier closed loop, {'five unfavourable labels at 0.5 q*' if adv else '20 exact labels'}: mean multiplier over the last 100 steps {np.mean([r[-100:].mean() for r in R]):.4f} (q* = {qs:.4f}); last step below q* - 0.10 at a median of {np.median(last):.0f} and at most {max(last)} (100 runs of 500 units)")
