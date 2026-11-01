"""Conformal quantile against the log-normal (Gauss) quantile when the ratio is NOT log-normal.

The article's experiments (Section 5) use C-MAPSS fleets whose ratios are close to log-normal, where a parametric rule is expected to do well.
Here the true distribution G of the ratio rho = R / p is known, the calibration sample is complete and i.i.d. (open loop, as in Theorem 1), and the
expected excess cost of a multiplier q is exact (Proposition 1):  Phi(q) - Phi(q*) = kappa * int_{q*}^{q} (G(x) - tau*) dx,  kappa = c_u + c_o.
Units: c_o = 1 and p = 1, so a cost is a multiple of c_o times the predicted lead time; "% of optimal" divides by Phi(q*).

Six ratio distributions:  lognormal (control); log-t with 3 degrees of freedom (heavy tails); two failure modes (70% / 30%, the second one early);
early-failure contamination (7% of units fail at 5-50% of the prediction); two operating regimes with opposite bias; Weibull with shape 5 (light lower tail).

usage: python misspecification_check.py   (about 1 minute)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math
import numpy as np

CU = [2, 5, 10, 20]                  # c_u / c_o
NS = [20, 50, 100, 400]
REPS = 6000
GRID = np.linspace(0.0, 25.0, 250001)
erf = np.vectorize(math.erf)
Phi_n = lambda z: 0.5 * (1 + erf(z / math.sqrt(2)))


def ln_cdf(mu, sig):
    return lambda x: np.where(x > 0, Phi_n((np.log(np.maximum(x, 1e-300)) - mu) / sig), 0.0)


def t3_cdf(t):
    return 0.5 + (t / (math.sqrt(3) * (1 + t * t / 3)) + np.arctan(t / math.sqrt(3))) / math.pi


DISTS = {}


def add(name, sampler, cdf):
    DISTS[name] = (sampler, cdf)


add("lognormal (control)", lambda r, n: r.lognormal(0.0, 0.35, n), ln_cdf(0.0, 0.35))
add("heavy tails (log-t, 3 df)", lambda r, n: np.exp(0.25 * r.standard_t(3, n)), lambda x: t3_cdf(np.log(np.maximum(x, 1e-300)) / 0.25) * (x > 0))


def mix_sampler(parts):
    def s(r, n):
        w = np.array([p[0] for p in parts]); k = r.choice(len(parts), size=n, p=w / w.sum()); out = np.empty(n)
        for i, p in enumerate(parts):
            m = k == i; out[m] = p[1](r, int(m.sum()))
        return out
    return s


def mix_cdf(parts):
    w = np.array([p[0] for p in parts]); w = w / w.sum()
    return lambda x: sum(wi * p[2](x) for wi, p in zip(w, parts))


lnp = lambda w, mu, sig: (w, lambda r, n: r.lognormal(mu, sig, n), ln_cdf(mu, sig))
unp = lambda w, a, b: (w, lambda r, n: r.uniform(a, b, n), lambda x: np.clip((x - a) / (b - a), 0, 1))
P2 = [lnp(0.7, 0.0, 0.15), lnp(0.3, -0.7, 0.20)]
P3 = [lnp(0.93, 0.0, 0.30), unp(0.07, 0.05, 0.50)]
P4 = [lnp(0.5, -0.25, 0.20), lnp(0.5, 0.25, 0.20)]
add("two failure modes (70/30)", mix_sampler(P2), mix_cdf(P2))
add("early-failure contamination (7%)", mix_sampler(P3), mix_cdf(P3))
add("two operating regimes (50/50)", mix_sampler(P4), mix_cdf(P4))
add("Weibull, shape 5", lambda r, n: 1.1 * r.weibull(5.0, n), lambda x: np.where(x > 0, 1 - np.exp(-(np.maximum(x, 0) / 1.1) ** 5), 0.0))


def prepare(cdf):
    G = cdf(GRID); dx = GRID[1] - GRID[0]
    IG = np.concatenate([[0], np.cumsum((G[1:] + G[:-1]) / 2 * dx)])                 # int_0^x G
    S = 1 - G; tail = np.concatenate([np.cumsum(((S[1:] + S[:-1]) / 2 * dx)[::-1])[::-1], [0]])   # int_x^25 (1 - G)
    return G, IG, tail


def run():
    from statistics import NormalDist
    rng = np.random.default_rng(12345)
    out = {}
    for name, (sampler, cdf) in DISTS.items():
        G, IG, tail = prepare(cdf)
        big = np.log(sampler(rng, 2_000_000)); mu_inf, sd_inf = big.mean(), big.std()          # limit of the log-normal fit
        samples = {}
        for n in NS:
            X = sampler(rng, n * REPS).reshape(REPS, n)                                         # REPS independent calibration sets
            samples[n] = (np.sort(X, axis=1), np.log(X).mean(axis=1), np.log(X).std(axis=1))
        for cu in CU:
            tau = 1 / (1 + cu); kap = cu + 1.0; zt = NormalDist().inv_cdf(tau)
            qs = float(np.interp(tau, G, GRID))
            Gq = lambda q: np.interp(q, GRID, G)
            I = lambda q: np.interp(q, GRID, IG) - tau * np.asarray(q)
            phi_star = cu * np.interp(qs, GRID, IG) + np.interp(qs, GRID, tail)
            exc = lambda q: kap * (I(q) - I(qs))
            qG_inf = math.exp(mu_inf + sd_inf * zt)
            row = {"q*": qs, "opt": float(phi_star), "gauss_inf": float(exc(qG_inf)), "cov_gauss_inf": float(Gq(qG_inf)), "tau": tau}
            for n in NS:
                Xs, m, sd = samples[n]
                k = min(max(int(math.ceil((n + 1) * tau)), 1), n)
                qc = Xs[:, k - 1]; qg = np.exp(m + sd * zt)
                ec, eg = exc(qc), exc(qg)
                row[n] = {"conf": float(ec.mean()), "gauss": float(eg.mean()), "cov_conf": float(Gq(qc).mean()), "cov_gauss": float(Gq(qg).mean()),
                          "conf_se": float(ec.std() / math.sqrt(REPS)), "gauss_se": float(eg.std() / math.sqrt(REPS))}
            out[(name, cu)] = row
    return out


if __name__ == "__main__":
    res = run()
    print(f"{'distribution':36s} cu/co  q*     opt   | n    conf    gauss  (% of optimal)  cov_c  cov_g  | Gauss n->inf: excess  % opt   coverage (target)")
    for (name, cu), row in res.items():
        for n in NS:
            r = row[n]
            tail = f"| {row['gauss_inf']:.4f}  {100 * row['gauss_inf'] / row['opt']:5.1f}%   {row['cov_gauss_inf']:.3f} ({row['tau']:.3f})" if n == NS[-1] else ""
            print(f"{name:36s} {cu:5d} {row['q*']:6.3f} {row['opt']:6.3f} | {n:4d} {r['conf']:.4f}  {r['gauss']:.4f}  ({100 * r['conf'] / row['opt']:4.1f}% {100 * r['gauss'] / row['opt']:4.1f}%)  {r['cov_conf']:.3f}  {r['cov_gauss']:.3f}  {tail}")
