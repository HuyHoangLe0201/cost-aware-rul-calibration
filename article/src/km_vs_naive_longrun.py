"""Long-run behaviour of the closed loop on a log-normal ratio (sigma = 0.35, c_u/c_o = 10): final multiplier and excess cost of the oracle, Kaplan-Meier, naive and
drop treatments of censored labels, for several initial sample sizes N0, from a purely random (unbiased) initial pool.
usage: python km_vs_naive_longrun.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math, numpy as np
from rulcost import CostAwareCalibrator

MU, SIG, CU = 0.0, 0.35, 10.0
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
def cost_q(q):                       # Phi(q) = c_u E(q - rho)^+ + c_o E(rho - q)^+ for lognormal rho
    z1 = (math.log(q) - MU) / SIG; z2 = z1 - SIG; m = math.exp(MU + SIG ** 2 / 2)
    return CU * (q * Phi(z1) - m * Phi(z2)) + (m * (1 - Phi(z2)) - q * (1 - Phi(z1)))
tau = 1 / (1 + CU); qstar = math.exp(MU + SIG * -1.3351)           # normal quantile at 1/11 = -1.3352
for N0 in (20, 100):
    print(f"N0 = {N0}: q* = {qstar:.3f}, Phi(q*) = {cost_q(qstar):.4f}")
    for method in ("oracle", "kaplan-meier", "naive", "drop"):
        rng = np.random.default_rng(3); fin = []; ex = []
        for rep in range(80):
            n = 800; rho = rng.lognormal(MU, SIG, n); cal = CostAwareCalibrator(CU, 1.0, method={"oracle": "conformal"}.get(method, method)); qs = []
            for s in range(n):
                if s < N0: cal.update(1.0, rho[s]); continue
                q = cal.multiplier(); qs.append(q)
                if method == "oracle" or rho[s] <= q: cal.update(1.0, rho[s])
                else: cal.update(1.0, q, censored=True)
            fin.append(np.mean(qs[-100:])); ex.append(np.mean([cost_q(q) - cost_q(qstar) for q in qs]))
        print(f"   {method:<13} final multiplier {np.mean(fin):.3f} (sd {np.std(fin):.3f}); mean excess cost over the stream {np.mean(ex):.4f}")
