"""Brute-force numerical verification of the elementary inequalities used in the proofs of Appendix A (Lemmas 2-5, Theorem 2(e),(f), Corollary 1, Proposition 5, Proposition 6).
Each check scans a fine grid and reports the worst margin (a non-negative margin means the inequality holds on the grid).
usage: python elementary_checks.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math, numpy as np
E = math.e
rows = []
def report(name, margin, note=""):
    rows.append((name, margin)); print(f"{'OK ' if margin >= -1e-12 else 'FAIL'} {name}: smallest margin {margin:.3e} {note}")

x = np.linspace(0, 1, 100001); report("e^{-x} <= 1-(1-e^{-1})x on [0,1]  (Lemma 4)", float(np.min(1 - (1 - 1/E)*x - np.exp(-x))))
t = np.linspace(0, 0.5, 100001); report("t <= -log(1-t)  (Lemma 3)", float(np.min(-np.log1p(-t) - t))); report("-log(1-t) <= t + t^2 on [0,1/2]  (Lemma 3)", float(np.min(t + t**2 + np.log1p(-t))))
report("1/(1-e^{-1}) < 1.6  (Lemma 4)", 1.6 - 1/(1 - 1/E))
l = np.linspace(E**2, 200, 100001); report("(e/l)^l <= e^{-l} for l >= e^2  (Lemma 2, Step 3)", float(np.min(-l - l*(1 - np.log(l)))))
b = np.linspace(1, 1000, 100001); report("5b^2 + 2.13b <= 8b^2 for b >= 1  (Lemma 2, Step 5)", float(np.min(3*b**2 - 2.13*b)))
nn = np.arange(5, 2_000_001); w = 1.0
worst = 1e9
for bb in (1.0, 1.5, 3.0, 5.0, 10.0, 50.0, 1000.0, 1e5):
    n = nn[nn >= bb]; worst = min(worst, float(np.min(5*np.log(n) - (np.ceil(np.log2(bb*n)) + 1))))
report("ceil(log2(bn)) + 1 <= 5 log n for n >= max(5, b)  (Lemma 2, Step 5)", worst)
n = nn.astype(float); report("10 b n^{-4} log n <= 2 b n^{-3} log n for n >= 5, and b n^{-4} <= 0.13 b n^{-3} log n for n >= 5  (Lemma 2, Step 5)", float(min(np.min(2 - 10/n), np.min(0.13*np.log(n) - 1/n))))
worst = 1e9
for n0 in list(range(2, 400)) + [1000, 10**4, 10**5, 10**6]:
    k = np.arange(n0, n0 + 4_000_000 if n0 < 10**4 else n0 + 20_000_000, dtype=float); s = float(np.sum(np.log(k)/k**3))
    tail = ((math.log(n0 + len(k)) + 0.5)/(2*(n0 + len(k))**2))      # integral tail beyond the truncation
    worst = min(worst, (math.log(n0) + 1)/n0**2 - (s + tail))
report("sum_{n >= n0} n^{-3} log n <= (log n0 + 1)/n0^2 for n0 >= 2  (Theorem 2(e))", worst)
N = np.arange(3, 10**6, dtype=float); n0 = np.ceil(N/2)
report("8 b^2 (log n0 + 1)/n0^2 <= 64 b^2 N^{-2} log N for N >= 3  (Theorem 2(e))", float(np.min(64*np.log(N)/N**2 - 8*(np.log(n0) + 1)/n0**2)))
report("2 N^{-3} + N^{-2} <= 3 N^{-2} log N for N >= 3  (Theorem 2(e))", float(np.min(3*np.log(N)/N**2 - 2/N**3 - 1/N**2)))
# (E): y >= 4c log^2(4c)  =>  y > c log^2 y
worst = 1e9
for c in [1, 1.5, 2, 3, 5, 10, 100, 1e3, 1e6, 1e10, 1e16]:
    y0 = 4*c*math.log(4*c)**2; y = y0*np.exp(np.linspace(0, 40, 400001)); worst = min(worst, float(np.min((y - c*np.log(y)**2)/y)))
report("(E): y >= 4c log^2(4c) implies y > c log^2 y, c >= 1  (Theorem 2(e),(f))", worst)
# Hoeffding's lemma as used in Lemma 5: for zeta in {0,1} with mean pi >= beta, E exp{4 beta (beta - zeta)} <= exp{2 beta^2}
beta = np.linspace(0.001, 1, 1000)[:, None]; pi = np.linspace(0, 1, 1001)[None, :]; ok = pi >= beta
lhs = pi*np.exp(4*beta*(beta - 1)) + (1 - pi)*np.exp(4*beta*beta); m = np.where(ok, 2*beta**2 - np.log(lhs), 1.0)
report("Lemma 5: E exp{4 beta (beta - zeta)} <= exp{2 beta^2} for Bernoulli(pi), pi >= beta", float(np.min(m)))
# Lemma 3: constants 1920 = (8 sqrt30)^2 and 2 sqrt(30) A / sqrt(C0) + 20 B0 / C0 <= theta/2 when C0 = max(1920 A^2/theta^2, 80 B0/theta)
rng = np.random.default_rng(1); A = rng.uniform(5, 50, 100000); B0 = rng.uniform(7, 30, 100000); th = rng.uniform(1e-3, 0.5, 100000); C0 = np.maximum(np.maximum(5, 1920*A**2/th**2), 80*B0/th)
report("Lemma 3: 2 sqrt30 A/sqrt(C0) + 20 B0/C0 <= theta/2 (random A, B0, theta)", float(np.min(th/2 - (2*math.sqrt(30)*A/np.sqrt(C0) + 20*B0/C0))))
# Lemma 3: Cauchy-Schwarz sums, sum_j 1/T_j <= 2/T0 for T_j = 2^j T0
report("sum_j 1/T_j <= 2/T0 (geometric)", 2 - sum(2.0**-j for j in range(60)))
# Theorem 2(f), Step 1: with eps_s^4 = 23 Upsilon_3 log^2 s / s, Gbar = Upsilon_3 / eps_s^4 = s / (23 log^2 s); then 8 Gbar <= s and ceil(8 Gbar log^2(8 Gbar)) + 1 <= s
worst = 1e9
for s_ in list(range(8, 20000)) + [int(10**e) for e in np.linspace(4.3, 9, 400)]:
    Gb = s_ / (23*math.log(s_)**2); lg = math.log(8*Gb)
    worst = min(worst, s_ - (math.ceil(8*Gb*lg**2) + 1), s_ - 8*Gb)
report("Theorem 2(f): 8 Gbar <= s and ceil(8 Gbar log^2(8 Gbar)) + 1 <= s when Gbar = s/(23 log^2 s), s >= 8", worst)
# Theorem 2(f), Step 1: sum_{s <= n} log s / sqrt(s) <= 2 sqrt(n) log n  (n >= 2)
cs = np.cumsum(np.log(np.arange(1, 2_000_001))/np.sqrt(np.arange(1, 2_000_001))); nn = np.arange(1, 2_000_001)
report("Theorem 2(f): sum_{s<=n} log s / sqrt(s) <= 2 sqrt(n) log n for n >= 2", float(np.min((2*np.sqrt(nn)*np.log(nn) - cs)[1:])))
# Theorem 2(f), Step 2: (8 b^2 / log 2) sum_{m>=2} (log m)^2 / m^2 <= 23 b^2, i.e. sum (log m)^2/m^2 <= 1.9877 * ... (tail by the integral)
mm = np.arange(2, 5_000_001, dtype=float); tot = float(np.sum(np.log(mm)**2/mm**2)); Nn_ = 5_000_000.0
tail = (math.log(Nn_)**2 + 2*math.log(Nn_) + 2)/Nn_                 # int_N^inf (log x)^2 / x^2 dx
report("Theorem 2(f): (8/log 2) sum_{m>=2} (log m)^2/m^2 <= 23", 23 - 8/math.log(2)*(tot + tail))
# Theorem 2(f), Step 2: sum_k eta_k^2 <= 4/3 eta_top^2, sum_k eta_k <= 2 eta_top (geometric); K <= (1/2) log_2 n when eta_min = sqrt(5 log n / n) and eta_top <= 1
report("geometric sums sum 4^-k <= 4/3 and sum 2^-k <= 2", min(4/3 - sum(4.0**-k for k in range(80)), 2 - sum(2.0**-k for k in range(80))))
# Theorem 2(f), Step 3: (1 + log^2 n) / (sqrt(n) log n) is nonincreasing for n >= e^2 and 1 + log^2 n <= 2 log^2 n for n >= 3
nn2 = np.arange(8, 3_000_000, dtype=float); f = (1 + np.log(nn2)**2)/(np.sqrt(nn2)*np.log(nn2))
report("Theorem 2(f): (1 + log^2 n)/(sqrt(n) log n) nonincreasing for n >= 8", float(np.min(f[:-1] - f[1:])))
# Corollary 1: sum_{m >= 1} 1/((m+1)(m+2)) = 1/2, sum_{m >= 1} (m+1)^-2 < 0.65, sum_{m = m0}^{n-1} 1/(m+2) <= log(n+2), tau_m (1 - tau_m) <= tau (1 - tau) + 1/(m+1)
worst = min(0.5 - 1/2, 0.65 - (math.pi**2/6 - 1))
for n_ in (3, 10, 100, 10**4, 10**6):
    for m0 in (1, 2, 5):
        if m0 < n_: worst = min(worst, math.log(n_ + 2) - sum(1/(m + 2) for m in range(m0, n_)))
rg = np.random.default_rng(5)
for _ in range(20000):
    tau = rg.uniform(0.01, 0.99); m = int(rg.integers(5, 5000)); k = math.ceil((m + 1)*tau); tm = k/(m + 1)
    worst = min(worst, tau*(1 - tau) + 1/(m + 1) - tm*(1 - tm))
report("Corollary 1: the elementary sums and tau_m (1 - tau_m) <= tau* (1 - tau*) + 1/(m+1)", worst)
# Proposition 6: Fisher information of the prior cos^2(pi theta / 2 delta)/delta on [-delta, delta] is pi^2 / delta^2
th = np.linspace(-1, 1, 2_000_001); lam = np.cos(np.pi*th/2)**2; lam1 = -np.pi*np.cos(np.pi*th/2)*np.sin(np.pi*th/2)
inner = np.where(lam > 1e-12, lam1**2/np.maximum(lam, 1e-12), 0.0)
report("Proposition 6: Fisher information of the prior cos^2(pi theta/2) on [-1, 1] equals pi^2 (relative error)", 1e-6 - abs(float(np.trapezoid(inner, th)) - math.pi**2)/math.pi**2)
# Proposition 5: sum over i of i*lambda / w = lambda (w+1)/2 ; k/w - tau in [0, 2/w)
ws = np.arange(5, 2000); taus = np.linspace(0.01, 0.99, 99); worst = 1e9; worst0 = 1e9
for w in ws[::37]:
    for tau in taus:
        k = math.ceil((w + 1)*tau); 
        if k > w: continue
        d = k/w - tau; worst = min(worst, 2/w - d); worst0 = min(worst0, d)
report("Proposition 5: 0 <= k/w - tau* < 2/w for k = ceil((w+1) tau*) with k <= w", min(worst, worst0))
print("all checks passed" if all(m >= -1e-12 for _, m in rows) else "SOME CHECK FAILED")
