"""Proposition 4: difference of the expected costs of the pooled and the stratified conformal policies against V - kappa tau(1-tau)/(2n) [sum_j 1/g_j(q_j*) - 1/g(q*)] (two log-normal strata, small V)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math, numpy as np
rng = np.random.default_rng(99); erf = np.vectorize(math.erf); SIG = 0.2; CU = 10.0; TAU = 1 / (1 + CU); KAP = CU + 1.0; W = (0.5, 0.5)
Nc = lambda z: 0.5 * (1 + erf(np.asarray(z) / math.sqrt(2)))
def Phi_j(q, mu):                                    # c_u E(q - rho)^+ + c_o E(rho - q)^+ for rho ~ LN(mu, SIG)
    q = np.maximum(np.asarray(q, float), 1e-12); z = (np.log(q) - mu) / SIG; Eqr = q * Nc(z) - math.exp(mu + SIG ** 2 / 2) * Nc(z - SIG); Er = math.exp(mu + SIG ** 2 / 2) - q + Eqr
    return CU * Eqr + Er
def qstar_j(mu): z = NormInv(TAU); return math.exp(mu + SIG * z)
def NormInv(p):
    from statistics import NormalDist; return NormalDist().inv_cdf(p)
def dens(x, mu): return math.exp(-0.5 * ((math.log(x) - mu) / SIG) ** 2) / (x * SIG * math.sqrt(2 * math.pi))
NS = (60, 100, 200, 800, 3200)
for m0 in (0.03,):
    mus = (-m0, m0); qj = [qstar_j(m) for m in mus]; gj = [dens(q, m) for q, m in zip(qj, mus)]
    # mixture quantile and density
    xs = np.linspace(0.2, 3, 200001); Gm = sum(w * Nc((np.log(xs) - m) / SIG) for w, m in zip(W, mus)); qm = float(np.interp(TAU, Gm, xs)); gm = sum(w * dens(qm, m) for w, m in zip(W, mus))
    Phibar = lambda q: sum(w * Phi_j(q, m) for w, m in zip(W, mus)); V = float(Phibar(qm) - sum(w * Phi_j(q, m) for w, q, m in zip(W, qj, mus)))
    for n in NS:
        reps = 6000; k = int(math.ceil((n + 1) * TAU)); pooled = np.empty(reps); strat = np.empty(reps)
        for r in range(reps):
            lab = rng.random(n) < W[0]; x = np.where(lab, np.exp(rng.normal(mus[0], SIG, n)), np.exp(rng.normal(mus[1], SIG, n)))
            pooled[r] = Phibar(np.sort(x)[k - 1]) - sum(w * Phi_j(q, m) for w, q, m in zip(W, qj, mus))
            tot = 0.0
            for w, mu, q_star, sel in ((W[0], mus[0], qj[0], lab), (W[1], mus[1], qj[1], ~lab)):
                xj = np.sort(x[sel]); nj = len(xj); kj = int(math.ceil((nj + 1) * TAU)); qh = xj[kj - 1] if kj <= nj else xj[-1]; tot += w * (float(Phi_j(qh, mu)) - float(Phi_j(q_star, mu)))
            strat[r] = tot
        diff = pooled.mean() - strat.mean(); se = math.sqrt(pooled.var() / reps + strat.var() / reps)
        pred = V - KAP * TAU * (1 - TAU) / (2 * n) * (sum(1 / g for g in gj) - 1 / gm)
        print(f"shift +-{m0}: V = {V:.5f}; n = {n:5d}: pooled - stratified expected cost = {diff:.5f} (s.e. {se:.5f}); prediction V - C/n[..] = {pred:.5f}; stratifying pays: {diff > 0}, rule: {pred > 0}")
