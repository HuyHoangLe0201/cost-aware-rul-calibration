"""Exact checks of two consequences of Propositions 1-2 (log-normal ratio, sigma = 0.35).
A. Proposition 2(c) (misjudged cost ratio): the multiplier Q(tau) designed for a wrong level tau has coverage error |tau - tau*|; its exact excess cost against the sandwich of Proposition 2(b).
B. Proposition 7 (ignoring the dependence between the prediction p and the ratio rho): z ~ N(0, 1), rho = exp(sigma z), p = exp(s (phi z + sqrt(1 - phi^2) e)), e ~ N(0, 1) independent;
   the p-weighted law of rho is log-normal(sigma s phi, sigma), so that G_w, q_w, q* and the costs are exact. The excess cost of the ordinary quantile q* against the sandwich
   kappa Delta_w^2 / (2 g_max) <= excess / E p <= kappa Delta_w^2 / (2 g_min) and against the bound kappa tau* (1 - tau*) Var(w) / (2 g_min), with Var(w) = exp(s^2) - 1.
C. Proposition 7 on the C-MAPSS fleet FD001 (MLP, c_u/c_o = 20): the a-priori bound kappa tau* (1 - tau*) CV(p)^2 / (2 gamma), gamma = (p_min / E p) g(q*) with g(q*) a kernel estimate, for the protocol and for thresholds uniform on 30-90.
usage: python robustness_checks.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np, math
from statistics import NormalDist
from scipy.stats import norm
N = NormalDist(); SIG = 0.35

def Phi_ratio(q, cu, mu=0.0):
    """c_u E (q - rho)^+ + E (rho - q)^+ for rho ~ log-normal(mu, SIG)"""
    d = (math.log(q) - mu)/SIG; e = math.exp(mu + SIG**2/2)
    return cu*(q*norm.cdf(d) - e*norm.cdf(d - SIG)) + e*(1 - norm.cdf(d - SIG)) - q*(1 - norm.cdf(d))

def dens(x, mu=0.0): return norm.pdf((math.log(x) - mu)/SIG)/(SIG*x)

print("A. misjudged cost ratio: true c_u/c_o, ratio used, coverage error |tau - tau*|, exact excess (units of c_o p), sandwich of Proposition 2(b)")
for cu_true in (10, 5, 2):
    tau_s = 1/(1 + cu_true); kap = cu_true + 1; qs = math.exp(SIG*N.inv_cdf(tau_s)); opt = Phi_ratio(qs, cu_true)
    for factor in (0.5, 2.0):
        cu_used = cu_true*factor; tau = 1/(1 + cu_used); q = math.exp(SIG*N.inv_cdf(tau)); ex = Phi_ratio(q, cu_true) - opt
        xs = np.linspace(min(q, qs), max(q, qs), 4001); gl = min(dens(v) for v in xs); gh = max(dens(v) for v in xs); D = abs(tau - tau_s)
        lo, hi = kap*D*D/(2*gh), kap*D*D/(2*gl)
        print(f"   c_u/c_o = {cu_true:2d}, ratio used {cu_used:4.1f}: coverage error {D:.4f}, excess {ex:.5f} ({100*ex/opt:.1f}% of the optimal cost {opt:.3f}); sandwich [{lo:.5f}, {hi:.5f}] holds: {lo - 1e-12 <= ex <= hi + 1e-12}")

print("B. dependence between p and rho: excess cost of the ordinary quantile q* under the p-weighted cost, per unit of E p")
for cu in (5, 20):
    tau = 1/(1 + cu); kap = cu + 1; zt = N.inv_cdf(tau)
    for s_, phi in ((0.06, 0.8), (0.30, 0.5), (0.30, 0.9), (0.50, 0.9), (0.50, -0.9)):
        mu_w = SIG*s_*phi; q_ord = math.exp(SIG*zt); q_w = math.exp(mu_w + SIG*zt)
        ex = Phi_ratio(q_ord, cu, mu_w) - Phi_ratio(q_w, cu, mu_w)
        Dw = abs(norm.cdf((math.log(q_ord) - mu_w)/SIG) - tau); varw = math.exp(s_**2) - 1
        xs = np.linspace(min(q_ord, q_w), max(q_ord, q_w), 4001); gl = min(dens(v, mu_w) for v in xs); gh = max(dens(v, mu_w) for v in xs)
        lo, b1, b2 = kap*Dw*Dw/(2*gh), kap*Dw*Dw/(2*gl), kap*tau*(1 - tau)*varw/(2*gl)
        print(f"   c_u/c_o = {cu:2d}, s = {s_:.2f}, phi = {phi:+.1f}: Var(w) = {varw:.4f}, Delta_w = {Dw:.4f} (Cauchy-Schwarz bound {math.sqrt(tau*(1 - tau)*varw):.4f}), excess/E p = {ex:.5f}, sandwich [{lo:.5f}, {b1:.5f}], bound with Var(w) {b2:.5f}; holds: {lo - 1e-12 <= ex <= b1 + 1e-12 and b1 <= b2 + 1e-12 and Dw <= math.sqrt(tau*(1 - tau)*varw) + 1e-12}")

print("C. Proposition 7 on FD001 (MLP, c_u/c_o = 20): a-priori bound as a fraction of the p-weighted cost level 0.3665")
import pickle
from scipy.stats import gaussian_kde
d = pickle.load(open(f"{WORK}/suite/mlp_FD001.pkl", "rb")); r = d["ratios"]; p = d["p_a"]; tau = 1/21; kap = 21.0
qs = np.quantile(r, tau); g = float(gaussian_kde(r)(qs)[0])
for name, cv, pm in (("protocol", p.std()/p.mean(), p.min()/p.mean()), ("thresholds uniform on 30-90", 0.2887, 0.5)):
    B = kap*tau*(1 - tau)*cv**2/(2*pm*g)
    print(f"   {name:28s}: CV(p) = {cv:.4f}, p_min/E p = {pm:.3f}, g(q*) = {g:.3f}: bound {B:.4f} per unit of E p = {100*B/0.3665:.1f}% of the cost level")
