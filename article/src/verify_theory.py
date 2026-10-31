"""Numerical checks of the theory in theory.md.
(A) exact identity  C(d)-C(d*) = kappa * int_{d*}^{d}(F-tau*)  and quadratic sandwich in coverage error.
(B) excess cost of the finite-sample conformal quantile scales ~ 1/n  (ratios from C-MAPSS FD002 alarms, smoothed).
(C) drift: optimal rolling window w* ~ drift^(-2/3)  (lognormal ratios with linear drift in log-location)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np
rng = np.random.default_rng(3)

# ---------- (A) identity & sandwich, lognormal R ----------
sig, cu, co = 0.4, 5.0, 1.0; kappa = cu + co; tau = co / kappa
M = 2_000_000; R = np.sort(np.exp(rng.normal(0, sig, M)))
cs = np.concatenate([[0], np.cumsum(R)])
def C(d):                                    # exact-ish expected cost from the sample
    j = np.searchsorted(R, d); return (cu * (d * j - cs[j]) + co * ((cs[-1] - cs[j]) - d * (M - j))) / M
from scipy.stats import norm
dstar = np.exp(sig * norm.ppf(tau)); print("(A) d*=%.4f  C(d*)=%.5f" % (dstar, C(dstar)))
f = lambda x: norm.pdf(np.log(x) / sig) / (sig * x)
for dh in (0.7, 0.85, 1.2, 1.5):
    d = dstar * dh; Fd = norm.cdf(np.log(d) / sig); delta = abs(Fd - tau)
    xs = np.linspace(min(d, dstar), max(d, dstar), 2001); fm, fM = f(xs).min(), f(xs).max()
    ex = C(d) - C(dstar); lo = kappa * fm * delta**2 / (2 * fM**2); hi = kappa * fM * delta**2 / (2 * fm**2)
    xi = np.linspace(min(d, dstar), max(d, dstar), 20001); ident = kappa * abs(np.trapezoid(norm.cdf(np.log(xi) / sig) - tau, xi))
    print("   d/d*=%.2f  excess=%.5f  identity=%.5f  sandwich [%.5f, %.5f]  ok=%s" % (dh, ex, ident, lo, hi, lo <= ex <= hi))

# ---------- (B) 1/n scaling with real ratios ----------
Zf = np.load(WORK + "/oof_cmapss_FD002.npz"); ids = sorted({k.split("__")[0] for k in Zf.files}); rho = []
for e in ids:
    r = Zf[f"{e}__rul"]; p = np.exp(Zf[f"{e}__data"]); T = len(r)
    ps = np.array([np.median(p[max(0, i - 4):i + 1]) for i in range(T)]); ix = np.where((np.arange(T) >= 10) & (ps <= 60))[0]
    if len(ix): rho.append(r[ix[0]] / ps[ix[0]])
rho = np.array(rho); print("\n(B) %d ratios, median %.2f, sd(log) %.2f" % (len(rho), np.median(rho), np.std(np.log(rho))))
def sampler(n, h=0.08): return rho[rng.integers(0, len(rho), n)] * np.exp(h * rng.normal(size=n))
Pop = np.sort(sampler(1_500_000)); cp = np.concatenate([[0], np.cumsum(Pop)]); MP = len(Pop)
for cu_ in (2.0, 10.0):
    kap = cu_ + 1; tau_ = 1 / kap
    Cq = lambda q: (cu_ * (q * (j := np.searchsorted(Pop, q)) - cp[j]) + ((cp[-1] - cp[j]) - q * (MP - j))) / MP
    qs = Pop[int(tau_ * MP)]; c0 = Cq(qs); print("   c_u/c_o=%g, tau*=%.3f" % (cu_, tau_))
    prev = None
    for n in (5, 10, 20, 40, 80, 160, 320):
        k = min(max(int(np.ceil((n + 1) * tau_)), 1), n); ex = []
        for _ in range(4000):
            q = np.sort(sampler(n))[k - 1]; ex.append(Cq(q) - c0)
        m = np.mean(ex); print("     n=%3d  E[excess]=%.5f   n*E[excess]=%.4f%s" % (n, m, n * m, "" if prev is None else "   local slope %.2f" % (np.log(m / prev[1]) / np.log(n / prev[0])))); prev = (n, m)

# ---------- (C) window exponent under drift ----------
sg, tau_c = 0.3, 0.1; Zs = np.sort(rng.normal(size=200_000)); cz = np.concatenate([[0], np.cumsum(np.exp(sg * Zs))]); Rr = np.exp(sg * Zs)
def Cpop(q, mu):                           # expected cost at location mu (lognormal), cu=9, co=1 (tau*=0.1)
    qq = q / np.exp(mu); j = np.searchsorted(Rr, qq); return np.exp(mu) * (9 * (qq * j - cz[j]) + ((cz[-1] - cz[j]) - qq * (len(Rr) - j))) / len(Rr)
print("\n(C) optimal rolling window vs drift speed a (log-location per step)")
ws = np.unique(np.round(np.geomspace(4, 400, 22)).astype(int)); res = []
for a in (0.0003, 0.001, 0.003, 0.01):
    best = []
    for w in ws:
        k = min(max(int(np.ceil((w + 1) * tau_c)), 1), w); T0 = 400; ex = []
        for _ in range(1500):
            mu_hist = a * (np.arange(T0 - w, T0)); obs = np.exp(mu_hist + sg * rng.normal(size=w))
            q = np.sort(obs)[k - 1]; mu_now = a * T0; qstar = np.exp(mu_now + sg * norm.ppf(tau_c))
            ex.append(Cpop(q, mu_now) - Cpop(qstar, mu_now))
        best.append(np.mean(ex))
    wstar = ws[int(np.argmin(best))]; res.append((a, wstar)); print("   a=%.4f  w*=%d  min excess %.5f" % (a, wstar, min(best)))
a_, w_ = np.array(res).T; print("   fitted log-log slope w* vs a: %.2f  (theory -2/3 = -0.67)" % np.polyfit(np.log(a_), np.log(w_), 1)[0])
