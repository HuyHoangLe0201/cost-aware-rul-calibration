"""Exact early-stream behaviour of the conformal quantile policy (answer to the question: what happens at a start-up sample such as n = 20 when tau* = 1/21?).

For i.i.d. continuous ratios and the conformal index k = ceil((n + 1) tau*), the probability integral G(q_hat) = G(rho_(k)) has the Beta(k, n + 1 - k) law, so that
  - the expected miscoverage is E G(q_hat) = k / (n + 1)  (target tau*; coverage = 1 - G(q_hat)),
  - the expected excess cost E[Phi(q_hat) - Phi(q*)] = E kappa * int_{q*}^{q_hat} (G(x) - tau*) dx is computed by quadrature over the Beta density (no Monte Carlo error).
The ratio is log-normal(0, 0.35) as in the checks of Theorem 2; costs are in units of c_o times the predicted lead time (p = 1).  The last columns give the smallest n for which
Theorem 1 applies (eta < min(tau*, 1 - tau*) and n > 2 / eta, so n > 2 / min(tau*, 1 - tau*) strictly) and the asymptotic value n * excess -> kappa tau* (1 - tau*) / (2 g(q*)).
usage: python early_stream_exact.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math
import numpy as np

SIG = 0.35
erf = np.vectorize(math.erf)
Phi_n = lambda z: 0.5 * (1 + erf(z / math.sqrt(2)))
GRID = np.linspace(1e-4, 4.0, 400001)
G = Phi_n(np.log(GRID) / SIG)
dens = lambda x: np.exp(-0.5 * (np.log(x) / SIG) ** 2) / (x * SIG * math.sqrt(2 * math.pi))
IG = np.concatenate([[0], np.cumsum((G[1:] + G[:-1]) / 2 * np.diff(GRID))])          # int G
CU = [2, 5, 10, 20]
NS = [5, 10, 20, 21, 30, 40, 41, 42, 60, 100, 200, 400]


def expected_excess(k, n, tau, kap):
    qs = float(np.interp(tau, G, GRID)); Iq = lambda q: np.interp(q, GRID, IG) - tau * q
    u = np.linspace(1e-7, 1 - 1e-7, 200001); lb = math.lgamma(n + 1) - math.lgamma(k) - math.lgamma(n + 1 - k)
    w = np.exp(lb + (k - 1) * np.log(u) + (n - k) * np.log1p(-u)); w = w / np.trapezoid(w, u)
    q = np.interp(u, G, GRID)
    return float(np.trapezoid(kap * (Iq(q) - Iq(qs)) * w, u)), qs


if __name__ == "__main__":
    print("log-normal(0, 0.35) ratio; expected coverage 1 - k/(n+1) against target 1 - tau*; expected excess cost in units of c_o * p")
    for cu in CU:
        tau = 1 / (1 + cu); kap = cu + 1.0
        n_min = int(math.floor(2 / min(tau, 1 - tau))) + 1
        qs = float(np.interp(tau, G, GRID)); gq = float(dens(qs)); const = kap * tau * (1 - tau) / (2 * gq)
        print(f"\nc_u/c_o = {cu}: tau* = {tau:.4f}, q* = {qs:.3f}, Theorem 1 needs n >= {n_min}, asymptotic n*excess = {const:.3f}")
        print("    n    k   coverage (target %.4f)   excess cost    n*excess" % (1 - tau))
        for n in NS:
            k = min(max(int(math.ceil((n + 1) * tau)), 1), n); ex, _ = expected_excess(k, n, tau, kap)
            print(f"{n:5d} {k:4d}   {1 - k / (n + 1):.4f}                 {ex:.5f}       {n * ex:.3f}")
