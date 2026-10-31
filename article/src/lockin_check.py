"""Numerical checks of Theorem 2 for the Kaplan-Meier policy as run (no guard), log-normal ratio rho ~ LogNormal(0, 0.35), tau* = 1/11, q* = 0.627.
 (1) accuracy of the Nelson-Aalen estimate at levels that rest on at least c log n units at risk (Lemma 3);
 (2) adversarial initial pools (five identical labels at 0.50): last step with a multiplier below q* - 0.10;
 (3) Lemma 4: the number of labels created at rank below K, i.e. exact labels rho_j with Y_(K)(j-1) < rho_j <= min(q_j, x) (x = q* - 0.10), against its bound K / (1 - G(x));
 (4) number of steps above q* + 0.10 against log n.
usage: python lockin_check.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math, numpy as np
import rulcost as rc

MU, SIG, TAU = 0.0, 0.35, 1 / 11
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
G = lambda y: Phi((math.log(y) - MU) / SIG) if y > 0 else 0.0
Lam = lambda y: -math.log(1 - G(y))
def Ginv(p):
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi); lo, hi = (mid, hi) if G(mid) < p else (lo, mid)
    return lo
QS = Ginv(TAU); X = QS - 0.10; PX = 1 - G(X)

def run(n, rng, n0, init=None, Ks=()):
    """Closed loop of the policy of Theorem 2; returns the multipliers and, for each K in Ks, the number of labels created at rank below K."""
    rho = rng.lognormal(MU, SIG, n)
    if init is not None: rho[:len(init)] = init
    q = np.full(n, np.inf); Y = rho.copy(); d = np.ones(n, bool); low = {K: 0 for K in Ks}
    for s in range(n0, n):
        q[s] = rc.kaplan_meier_quantile(Y[:s], d[:s], TAU)
        Y[s] = min(rho[s], q[s]); d[s] = rho[s] <= q[s]
        if Ks and d[s]:
            top = np.sort(Y[:s])[::-1]
            for K in Ks:
                yk = top[K - 1] if s >= K else 0.0
                if yk < rho[s] <= min(q[s], X): low[K] += 1
    return rho, q, Y, d, low

def na_error_supported(Y, d, x, rmin):
    pts = np.sort(Y[d & (Y <= x)]); err = cum = 0.0
    for t in pts:
        r = np.sum(Y >= t)
        if r < rmin: break
        err = max(err, abs(cum - Lam(t))); cum += 1.0 / r; err = max(err, abs(cum - Lam(t)))
    return err

if __name__ == "__main__":
    rng = np.random.default_rng(11); x = Ginv(0.6)
    print(f"tau* = {TAU:.4f}, q* = {QS:.4f}, x = q* - 0.10 = {X:.4f}, 1 - G(x) = {PX:.3f}")
    print("(1) sup |Lambda_hat - Lambda| over levels with at least c*log n units at risk (N0 = 20)")
    for c in (2, 5, 10, 20):
        errs = []
        for n in (200, 400, 800):
            for _ in range(8):
                rho, q, Y, d, _ = run(n, rng, 20); errs.append(na_error_supported(Y, d, x, c * math.log(n)))
        print(f"   c = {c:>2}: median {np.median(errs):.3f}, max {np.max(errs):.3f}")
    print("(2) five identical initial labels at 0.50 (300 runs of 600 units): last step with a multiplier below q* - 0.10")
    last = []
    for _ in range(300):
        rho, q, Y, d, _ = run(600, rng, 5, init=np.full(5, 0.50))
        bad = np.nonzero(q < X)[0]; last.append(bad.max() if len(bad) else -1)
    last = np.array(last)
    print(f"   median {np.median(last):.0f}, 99th percentile {np.percentile(last, 99):.0f}, max {last.max()}; runs with a step beyond 300: {np.mean(last >= 300):.3f}")
    print("(3) labels created at rank below K (5 identical initial labels at 0.50, 600 units, 200 runs): mean and maximum, against K / (1 - G(x))")
    Ks = (5, 10, 20, 40); cnt = {K: [] for K in Ks}
    for _ in range(200):
        rho, q, Y, d, low = run(600, rng, 5, init=np.full(5, 0.50), Ks=Ks)
        for K in Ks: cnt[K].append(low[K])
    for K in Ks: print(f"   K = {K:>2}: mean {np.mean(cnt[K]):.2f}, max {np.max(cnt[K])}, bound {K / PX:.1f}")
    print("(4) steps above q* + 0.10 (single runs, N0 = 20)")
    for n in (500, 1000, 2000):
        rho, q, Y, d, _ = run(n, rng, 20)
        print(f"   n = {n:>4}: {int(np.sum(q > QS + 0.10)):>3} steps; log n = {math.log(n):.1f}; mean of the last 200 multipliers {np.mean(q[-200:]):.3f}")
