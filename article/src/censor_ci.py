"""Engine-level bootstrap intervals for the closed-loop censoring experiment.
For each fleet, c_u/c_o and method m in {naive, drop, km}: per-engine mean cost (over random stream orders) minus the oracle-label cost for the SAME engine and the SAME orders;
bootstrap over engines (4000 resamples). Also coverage difference. N0=20, P0=60, MLP, 60 orders (same as censor_eval.py)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
DS = sys.argv[1]; sys.argv = ["x", DS]
import censor_eval as C
A, E, N0, P0 = C.A, C.E, C.N0, C.P0

def run_units(method, cu, order, co=1.0):
    tau = co / (co + cu); ra = {e: A[e][1] / A[e][0] for e in E}; init, test = order[:N0], order[N0:]
    vals = [ra[e] for e in init]; dl = [1.0] * N0; out = {}
    for e in test:
        if e not in A: continue
        p, r = A[e]; rho = ra[e]; v = np.array(vals); d_ = np.array(dl)
        q = C.q_km(v, d_, tau) if method == "km" else C.q_exact(v, tau)
        d = p * q; fail = d >= r
        out[e] = ((cu * max(d - r, 0) + co * max(r - d, 0)) / P0, float(r >= d))
        if method == "oracle" or fail: vals.append(rho); dl.append(1.0)
        elif method == "naive": vals.append(q); dl.append(1.0)
        elif method == "km": vals.append(q); dl.append(0.0)
    return out

def boot(x, B=4000):
    r = np.random.default_rng(0); idx = r.integers(0, len(x), (B, len(x))); s = x[idx].mean(1); return x.mean(), *np.percentile(s, [2.5, 97.5])

rng0 = np.random.default_rng(5); orders = [list(rng0.permutation(E)) for _ in range(C.NPERM)]
print(f"{DS}: engines {len(E)}  (difference to oracle labels; engine-level 95% bootstrap; cost then coverage)")
for cu in (2, 5, 10, 20):
    per = {}
    for m in ("oracle", "naive", "drop", "km"):
        acc = {e: [] for e in E}
        for o in orders:
            for e, (c, cv) in run_units(m, cu, o).items(): acc[e].append((c, cv))
        per[m] = {e: np.mean(v, 0) for e, v in acc.items() if v}
    common = sorted(set(per["oracle"]) & set(per["km"]) & set(per["naive"]) & set(per["drop"]))
    line = f" c_u/c_o={cu:>2}: "
    for m in ("km", "naive", "drop"):
        dc = np.array([per[m][e][0] - per["oracle"][e][0] for e in common]); dv = np.array([per[m][e][1] - per["oracle"][e][1] for e in common])
        a, lo, hi = boot(dc); b, lo2, hi2 = boot(dv)
        line += f"{m}: cost {a:+.3f} [{lo:+.3f},{hi:+.3f}] cov {b:+.3f} [{lo2:+.3f},{hi2:+.3f}] | "
    print(line)
