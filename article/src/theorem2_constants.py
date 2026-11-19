"""Independent recomputation of the explicit constants of Theorem 2 from the formulas as printed in the paper (log-normal example sigma = 0.35, cu/co = 10, N0 = 5)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math
from statistics import NormalDist
SIG, TAU, N0, M = 0.35, 1/11, 5, 1.0
nd = NormalDist(); G = lambda x: nd.cdf(math.log(x)/SIG); g = lambda x: math.exp(-0.5*(math.log(x)/SIG)**2)/(x*SIG*math.sqrt(2*math.pi))
qs = math.exp(SIG*nd.inv_cdf(TAU)); print("q* =", round(qs, 4), " G(M) =", round(G(M), 4), "> tau* :", G(M) > TAU)

# ---- Theorem 2(e): level x = q* - 0.10 ----
x = qs - 0.10; p = 1 - G(x); Lam = -math.log(p); b = 1 + Lam; c = max(1, Lam); C1 = 2/p + 3; A = 2*math.sqrt(2*b*C1/p); B0 = c + 6
theta = (TAU - G(x))/2; C0 = max(5, 1920*A**2/theta**2, 80*B0/theta)
n1 = max(5, math.ceil(b), math.ceil(math.exp(N0*b/5))); n2 = max(n1, math.ceil(math.exp(2*N0*max(1, (Lam + theta/2)/theta)/C0)))
Cp = 2/p*(C0 + 5*C1); C4 = 1 + 2/p*(C0 + 3/(4*p)); Gp = Cp + N0*C4 + 1 + C4*(3.2*C0 + 16)/p
bound_N1 = max(3, 2*n2, math.ceil(Cp), math.ceil(8*Gp*math.log(8*Gp)**2) + 1)
print(f"(e): p={p:.4f} b={b:.4f} C1={C1:.3f} A={A:.3f} B0={B0:.1f} theta={theta:.5f} C0={C0:.4g} n2={n2} C'={Cp:.4g} C4={C4:.4g} Gamma'={Gp:.4g}  N1 <= {bound_N1:.4g}")
# exact least N >= 3 such that (alpha), (beta) hold for all larger N (conditions are monotone beyond the first solution)
def beta_ok(N):
    Amax = (3.2*C0 + 16)*math.log(N)/p
    return 0.5*(N + 1) + (N0 + Amax)*C4*math.log(N) + Cp*math.log(N) < N
lo, hi = 3.0, 1e40
for _ in range(400):
    mid = math.sqrt(lo*hi)
    if beta_ok(mid) and math.ceil(mid/2) >= n2 and mid >= Cp: hi = mid
    else: lo = mid
print(f"     least N satisfying (alpha), (beta): {hi:.4g}   (the closed-form bound above is {bound_N1:.4g}; the audit gave 3.116e+20)")

# ---- Theorem 2(f): thresholds for several window half-widths eps0 ----
for eps0 in (0.05, 0.1, 0.15, 0.2):
    if not (0 < eps0 < qs and qs + eps0 < M and G(qs + eps0) < 1): continue
    xs = [qs - eps0 + 2*eps0*i/2000 for i in range(2001)]; glo = min(g(v) for v in xs); ghi = max(g(v) for v in xs)
    p0 = 1 - G(qs + eps0); L0 = -math.log(p0); b0 = 1 + L0; C1b = 2/p0 + 3; Ab = 2*math.sqrt(2*b0*C1b/p0)
    U0 = max(5*eps0**2, 7680*Ab**2/glo**2, 160*(b0 + 6)*eps0/glo); U1 = 2/p0*(U0 + 5*C1b*eps0**2); U2 = eps0**2 + 2/p0*(U0 + 3*eps0**2/(4*p0))
    U3 = U1*eps0**2 + N0*U2*eps0**2 + eps0**4 + U2*(3.2*U0 + 16*eps0**2)/p0
    n2b = max(5, math.ceil(b0), math.ceil(math.exp(N0*b0/5)), math.ceil(math.exp(N0*(L0 + 1)/1e4)))
    def ok(n):
        if n < 8*U3 or n < 3: return False
        e = n**(-1/6)*math.log(n)**(1/3); Nn = math.ceil(23*U3*n**(2/3)*math.log(n)**(2/3))
        return (2*e <= eps0 and U3*n**(2/3)*math.log(n)**(2/3) >= 3 and max(2*n2b, 4/(glo*e) + 3, U1/e**2 + 1) <= 23*U3*n**(2/3)*math.log(n)**(2/3)
                and Nn + 1 <= n and (128*b0**2 + 3)*Nn**(-2)*math.log(Nn) <= n**(-1/3)*math.log(n)**(2/3))
    lo, hi = 3.0, 1e200
    for _ in range(600):
        mid = math.sqrt(lo*hi)
        if ok(mid): hi = mid
        else: lo = mid
    print(f"(f): eps0 = {eps0}: g_min = {glo:.3f}, g_max = {ghi:.3f}, Upsilon_0 = {U0:.3g}, Upsilon_3 = {U3:.3g}; smallest n satisfying the conditions of n_f: {hi:.3g}")
