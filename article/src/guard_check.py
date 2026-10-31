"""Numerical checks of Theorem 6 (Kaplan-Meier with a risk-set guard): no lock-in from any initial pool, and the guard is inactive in the article's experiments.
 (1) accuracy of the Nelson-Aalen estimate at levels that rest on at least c log n units at risk (Lemma A4), on closed-loop data of the unguarded policy;
 (2) adversarial initial pools (all initial labels far below q*): last step below q* - 0.10, unguarded against guarded;
 (3) number of steps above q* + 0.10 against log n log log n and log^2 n for the guarded policy;
 (4) the closed-loop experiments of the article (C-MAPSS FD001-FD004): how often the guard changes the multiplier, and the cost with and without it.
usage: python guard_check.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math, os, sys, numpy as np
import rulcost as rc
from rulcost.core import risk_floor

MU, SIG, TAU = 0.0, 0.35, 1 / 11
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
G = lambda y: Phi((math.log(y) - MU) / SIG) if y > 0 else 0.0
Lam = lambda y: -math.log(1 - G(y))
def Ginv(p):
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi); lo, hi = (mid, hi) if G(mid) < p else (lo, mid)
    return lo
QS = Ginv(TAU)

def run(n, rng, n0, guard, init=None, tau=TAU):
    rho = rng.lognormal(MU, SIG, n)
    if init is not None: rho[:len(init)] = init
    q = np.full(n, np.inf); Y = rho.copy(); d = np.ones(n, bool)
    for s in range(n0, n):
        q[s] = rc.kaplan_meier_quantile(Y[:s], d[:s], tau, 5, 1.0, risk_floor(s, guard))
        Y[s] = min(rho[s], q[s]); d[s] = rho[s] <= q[s]
    return rho, q, Y, d

def na_error_supported(Y, d, x, rmin):
    """sup of |Lambda_hat(y) - Lambda(y)| over y <= x with R_n(y) >= rmin (Nelson-Aalen)."""
    n = len(Y); pts = np.sort(Y[d & (Y <= x)]); err = 0.0; cum = 0.0
    for t in pts:
        r = np.sum(Y >= t)
        if r < rmin: break
        err = max(err, abs(cum - Lam(t))); cum += 1.0 / r; err = max(err, abs(cum - Lam(t)))
    return err

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    x = Ginv(0.6)
    print(f"tau* = {TAU:.4f}, q* = {QS:.4f}")
    print("(1) sup |Lambda_hat - Lambda| over levels with at least c*log n units at risk (unguarded policy, N0 = 20)")
    for c in (2, 5, 10, 20):
        errs = []
        for n in (200, 400, 800):
            for _ in range(8):
                rho, q, Y, d = run(n, rng, 20, None); errs.append(na_error_supported(Y, d, x, c * math.log(n)))
        print(f"   c = {c:>2}: median {np.median(errs):.3f}, max {np.max(errs):.3f}  (the bound of Lemma A4 is C*sqrt(1/c))")
    print("(2) adversarial pools: last step with multiplier below q* - 0.10 (300 runs of 600 units, N0 = 5, all initial labels = 0.50)")
    for guard in (None, "loglog"):
        last = []
        for _ in range(300):
            rho, q, Y, d = run(600, rng, 5, guard, init=np.full(5, 0.50))
            bad = np.nonzero(q < QS - 0.10)[0]; last.append(bad.max() if len(bad) else -1)
        last = np.array(last)
        print(f"   guard={str(guard):<5}: any step below: {np.mean(last >= 0):.2f}; last such step: median {np.median(last[last>=0]):.0f}, 99th pct {np.percentile(last[last>=0], 99):.0f}, max {last.max()}; steps >= 300: {np.mean(last >= 300):.3f}")
    print("(3) guarded policy: number of steps with multiplier above q* + 0.10 (single runs) against log^2 n")
    for n in (500, 1000, 2000, 4000):
        rho, q, Y, d = run(n, rng, 20, "loglog")
        print(f"   n = {n:>5}: {int(np.sum(q > QS + 0.10)):>3} steps above q*+0.10; log^2 n = {math.log(n)**2:.0f}; multiplier at n: {q[-1]:.3f}; mean of the last 200: {np.mean(q[-200:]):.3f}")

    # (4) the closed-loop experiments of the article
    print("(4) article closed loop (C-MAPSS, Kaplan-Meier arm of Table 3; N0 = 20, 60 random orders): steps at which the guard ceil(log n log log n) changes the multiplier")
    P0, NPERM, TAU_EXP = 60, 60, 0.8
    for DS in ("FD001", "FD002", "FD003", "FD004"):
        Z = np.load(f"{WORK}/oof_cmapss_{DS}.npz"); ids = sorted({k.split("__")[0] for k in Z.files}); A = {}
        for e in ids:
            r = Z[f"{e}__rul"]; p = np.exp(Z[f"{e}__data"]); T = len(r)
            ps = np.array([np.median(p[max(0, i - 4):i + 1]) for i in range(T)]); ix = np.where((np.arange(T) >= 10) & (ps <= P0))[0]
            if len(ix): A[e] = (ps[ix[0]], r[ix[0]])
        E = [e for e in ids if e in A]; ra = {e: A[e][1] / A[e][0] for e in E}
        rng0 = np.random.default_rng(5); orders = [list(rng0.permutation(E)) for _ in range(NPERM)]
        for eps in (0.0, 0.1, 0.2):
            tot = binds = 0; dcost = []
            for cu in (2, 5, 10, 20):
                tau = 1.0 / (1.0 + cu)
                for guard in (None, "loglog"):
                    rngs = np.random.default_rng(100 + int(cu * 10 + eps * 100))
                    costs = []
                    for order in orders:
                        init, test = order[:20], order[20:]; vals = [ra[e] for e in init]; dl = [1.0] * 20; c = []
                        for e in test:
                            p, r = A[e]; v = np.array(vals); d_ = np.array(dl)
                            explore = rngs.random() < eps; tt = TAU_EXP if explore else tau
                            q = rc.kaplan_meier_quantile(v, d_, tt, 5, 1.0, 0 if explore else risk_floor(len(v), guard))        # the floor protects the calibration quantile, not the exploration arm
                            if guard is None:
                                q2 = rc.kaplan_meier_quantile(v, d_, tt, 5, 1.0, 0 if explore else risk_floor(len(v), "loglog")); tot += 1; binds += int(q2 != q)
                            d = p * q; fail = d >= r; c.append((cu * max(d - r, 0) + max(r - d, 0)) / P0)
                            if fail: vals.append(ra[e]); dl.append(1.0)
                            else: vals.append(q); dl.append(0.0)
                        costs.append(np.mean(c))
                    if guard is None: c0 = np.mean(costs)
                    else: dcost.append(np.mean(costs) - c0)
            print(f"   {DS} eps={eps:.1f}: guard changed {binds} of {tot} multipliers; largest change in mean cost {max(abs(x) for x in dcost):.4f}")
