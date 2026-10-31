"""Numerical checks of Theorem 5 (Kaplan-Meier under policy-induced censoring) on a log-normal ratio rho ~ LogNormal(0, 0.35).
 (1) the one-band form of the integration-by-parts bound behind Lemma A3,  sup_{y<=x} |Lambda_hat_n(y) - Lambda(y)| <= 2 sup_{u<=x} |M_n(u)| / R_n(x)   (M_n = N_n - int R_n dLambda), on censoring sequences that
     depend on the past (Kaplan-Meier policy, a random-walk policy, a policy that tracks the running minimum), and the size of sup|M_n|/sqrt(n log n);
 (2) part (a): frequency of ever replacing below x = q* - Delta against the Chernoff bound sum_{m>=N0} exp(-m D(tau || G(x)));
 (3) part (b): number of units with multiplier >= x+ = q* + Delta (Theorem 5(b): O(log n)).
The guarded policy of Theorem 6 is checked in guard_check.py.
usage: python km_theory_check.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math, numpy as np
from rulcost import kaplan_meier_quantile

MU, SIG = 0.0, 0.35
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
G = lambda y: Phi((math.log(y) - MU) / SIG) if y > 0 else 0.0
Lam = lambda y: -math.log(1 - G(y))
def Ginv(p):
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if G(mid) < p else (lo, mid)
    return lo

def run_policy(n, rng, kind, tau=1 / 11, N0=20):
    rho = rng.lognormal(MU, SIG, n); q = np.full(n, np.inf); Y = np.empty(n); d = np.zeros(n, bool)
    for s in range(n):
        if s >= N0:
            if kind == "km":
                v, e = Y[:s], d[:s]; q[s] = kaplan_meier_quantile(v, e, tau)
            elif kind == "walk": q[s] = float(np.exp(rng.normal(-0.3, 0.5)))                      # censoring level independent of the data
            elif kind == "minrun": q[s] = 1.2 * Y[:s][d[:s]].min() if d[:s].any() else 1.0       # depends on the smallest exact label so far
        Y[s] = min(rho[s], q[s]); d[s] = rho[s] <= q[s]
    return rho, q, Y, d

def na_and_bound(Y, d, x):
    """exact sup|M_n(u)| over u<=x, Nelson-Aalen error, and R_n(x)"""
    n = len(Y); order = np.argsort(Y, kind="stable"); Ys, ds = Y[order], d[order]
    R = n - np.arange(n)                                    # risk set size at each ordered observation (ties: at-risk includes tied observations)
    # exact-event times up to x
    ev = [(Ys[i], R[i]) for i in range(n) if ds[i] and Ys[i] <= x]
    # R_n(u) = #{Y >= u} is a left-continuous step function; A_n(u) = int_0^u R_n dLambda
    grid = np.concatenate([[0.0], np.unique(np.minimum(Ys[Ys <= x], x)), [x]])
    grid = np.unique(grid)
    A = {0.0: 0.0}; acc = 0.0
    for a, b in zip(grid[:-1], grid[1:]):
        Rab = np.sum(Y >= b)                                # R_n is constant (= #{Y >= b}) on (a, b]
        acc += Rab * (Lam(b) - Lam(a)); A[b] = acc
    Nn = lambda u: int(np.sum(d & (Y <= u)))
    S = 0.0
    for u in grid:
        S = max(S, abs(Nn(u) - A[u]))                       # right limit at u
        if u > 0: S = max(S, abs(Nn(u) - np.sum(d & (Y == u)) - A[u]))     # left limit at u
    # NA error uniformly on y<=x: evaluate at event times (right and left limits) and x
    err = 0.0; cum = 0.0
    for t, r in ev:
        err = max(err, abs(cum - Lam(t)))                   # just before the jump
        cum += 1.0 / r; err = max(err, abs(cum - Lam(t)))   # just after
    err = max(err, abs(cum - Lam(x)))
    return S, err, int(np.sum(Y >= x))

if __name__ == "__main__":
    rng = np.random.default_rng(1); tau = 1 / 11; qstar = Ginv(tau)
    print(f"tau = {tau:.4f}, q* = {qstar:.4f}")
    print("(1) pathwise bound  err <= 2 S / R_n(x):")
    for kind in ("km", "walk", "minrun"):
        worst = 0.0; ratios = []
        for rep in range(40):
            n = int(rng.integers(150, 500)); rho, q, Y, d = run_policy(n, rng, kind)
            x = Ginv(0.6)
            S, err, Rx = na_and_bound(Y, d, x)
            if Rx > 0: worst = max(worst, err / (2 * S / Rx)); ratios.append(S / math.sqrt(n * math.log(n)))
        print(f"   {kind:<7} max err / bound = {worst:.3f} (must be <= 1);  sup|M_n| / sqrt(n log n) in [{min(ratios):.2f}, {max(ratios):.2f}]")
    print("(2) part (a): frequency of ever replacing below x = q* - Delta within 300 units, against the bound")
    for N0 in (20, 50, 100):
        for Delta in (0.10, 0.15, 0.20):
            x = qstar - Delta; Gx = G(x); D = lambda a, p: a * math.log(a / p) + (1 - a) * math.log((1 - a) / (1 - p))
            bound = sum(math.exp(-m * D(tau, Gx)) for m in range(N0, 100000)); hits = 0; reps = 400
            for _ in range(reps):
                rho, q, Y, d = run_policy(300, rng, "km", tau, N0)
                hits += bool(np.any(q[N0:] < x))
            print(f"   N0={N0:>3} Delta={Delta:.2f} (G(x)={Gx:.3f}): empirical {hits / reps:.3f}   bound {min(bound, 1.0):.3f}")
    print("(3) part (b): number of units with multiplier >= x+ = q* + 0.15 (single long runs)")
    for n in (500, 1000, 2000):
        cnt = []
        for _ in range(6):
            rho, q, Y, d = run_policy(n, rng, "km"); cnt.append(int(np.sum(q[20:] >= qstar + 0.15)))
        print(f"   n={n:>4}: counts {cnt}, mean {np.mean(cnt):.1f}, log n = {math.log(n):.1f}")
