"""Brute-force numerical verification of the elementary inequalities used in the proofs of Appendix A (Lemmas 2-5, Theorem 2(e),(f), Proposition 5).
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
# Theorem 2(f): 8 G log^2(8 G) <= (200/9) Upsilon n^{2/3} (log n)^{2/3} when G = Upsilon n^{2/3} (log n)^{-4/3} and 8G <= n^{5/3}
worst = 1e9
for lu in (0, 1, 3, 6, 10, 15, 20):
    Ups = 10.0**lu
    for ln in np.linspace(math.log(max(8*Ups, 3)), 700, 4000):
        n = math.exp(ln); Gb = Ups*n**(2/3)*ln**(-4/3); lg = math.log(8*Gb)
        if lg <= (5/3)*ln: worst = min(worst, (200/9)*Ups*n**(2/3)*ln**(2/3) - 8*Gb*lg**2)
report("Theorem 2(f): 8 Gbar log^2(8 Gbar) <= (200/9) Upsilon_3 n^{2/3}(log n)^{2/3} (granted log(8 Gbar) <= (5/3) log n)", worst)
# Proposition 5: sum over i of i*lambda / w = lambda (w+1)/2 ; k/w - tau in [0, 2/w)
ws = np.arange(5, 2000); taus = np.linspace(0.01, 0.99, 99); worst = 1e9; worst0 = 1e9
for w in ws[::37]:
    for tau in taus:
        k = math.ceil((w + 1)*tau); 
        if k > w: continue
        d = k/w - tau; worst = min(worst, 2/w - d); worst0 = min(worst0, d)
report("Proposition 5: 0 <= k/w - tau* < 2/w for k = ceil((w+1) tau*) with k <= w", min(worst, worst0))
print("all checks passed" if all(m >= -1e-12 for _, m in rows) else "SOME CHECK FAILED")
