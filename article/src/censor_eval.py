"""Censoring from preventive replacement (closed-loop calibration).
After an engine is deployed with replacement date d = p*q, its label rho = r/p is observed EXACTLY only if it fails first (rho <= q);
if replaced preventively (rho > q) we only learn rho > q (right-censored at q).
The first N0 engines supply exact labels (offline run-to-failure testing). Afterwards each method evolves its own label pool.
Methods: oracle (exact labels always, = open-loop benchmark), naive (censored value treated as exact), drop (censored discarded),
km (Kaplan-Meier censoring-aware quantile). Exploration: with prob eps an engine is replaced at the 0.8-level quantile instead (less censoring).
usage: python censor_eval.py FD002"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
DS = sys.argv[1] if len(sys.argv) > 1 else "FD002"
Z = np.load(f"{WORK}/oof_cmapss_{DS}.npz"); ids = sorted({k.split("__")[0] for k in Z.files})
import os
P0, N0, NPERM, MINPOOL, TAU_EXP = 60, int(os.environ.get("N0", 20)), 60, 5, 0.8
A = {}
for e in ids:
    r = Z[f"{e}__rul"]; p = np.exp(Z[f"{e}__data"]); T = len(r)
    ps = np.array([np.median(p[max(0, i - 4):i + 1]) for i in range(T)]); ix = np.where((np.arange(T) >= 10) & (ps <= P0))[0]
    if len(ix): A[e] = (ps[ix[0]], r[ix[0]])
E = [e for e in ids if e in A]

def q_exact(v, tau):
    n = len(v)
    if n < MINPOOL: return 1.0
    return np.sort(v)[min(max(int(np.ceil((n + 1) * tau)), 1), n) - 1]

def q_km(v, d, tau):
    n = len(v)
    if n < MINPOOL: return 1.0
    o = np.argsort(v, kind="stable"); vs, ds = v[o], d[o]
    S = np.cumprod(1.0 - ds / (n - np.arange(n))); G = 1.0 - S
    t = min(tau * (n + 1) / n, 1.0); hit = np.where(G >= t)[0]
    return vs[hit[0]] if len(hit) else 1.0                      # not identified -> fall back to point prediction

def run(method, eps, cu, order, rng, co=1.0):
    tau = co / (co + cu); ra = {e: A[e][1] / A[e][0] for e in E}
    init, test = order[:N0], order[N0:]
    vals = [ra[e] for e in init]; dl = [1.0] * N0
    cost, cov = [], []
    for e in test:
        if e not in A: continue
        p, r = A[e]; rho = ra[e]; v = np.array(vals); d_ = np.array(dl)
        explore = (method != "oracle") and (rng.random() < eps)
        tt = TAU_EXP if explore else tau
        if method in ("oracle", "naive", "drop"): q = q_exact(v if method != "drop" else v, tt)
        else: q = q_km(v, d_, tt)
        d = p * q; fail = d >= r
        cost.append((cu * max(d - r, 0) + co * max(r - d, 0)) / P0)
        if not explore: cov.append(r >= d)
        if method == "oracle": vals.append(rho); dl.append(1.0)
        elif fail: vals.append(rho); dl.append(1.0)
        elif method == "naive": vals.append(q); dl.append(1.0)           # censored at q treated as exact
        elif method == "km": vals.append(q); dl.append(0.0)              # censored
        # drop: discard censored label
    return np.mean(cost), np.mean(cov)

if __name__ == "__main__":
    rng0 = np.random.default_rng(5)
    orders = [list(rng0.permutation(E)) for _ in range(NPERM)]
    print(f"{DS}: {len(E)} engines, P0={P0}, N0={N0}, perms={NPERM}")
    cfg = [("oracle", 0), ("naive", 0), ("drop", 0), ("km", 0), ("km", 0.1), ("km", 0.2), ("naive", 0.1), ("drop", 0.1)]
    for cu in (2, 5, 10, 20):
        tau = 1 / (1 + cu); line = f"cu/co={cu:>2} target cover {1-tau:.2f} | "
        for m, eps in cfg:
            rng = np.random.default_rng(99); res = [run(m, eps, cu, o, rng) for o in orders]
            line += f"{m}{'' if eps == 0 else '+e' + str(eps)}: cost {np.mean([r[0] for r in res]):.2f} cov {np.mean([r[1] for r in res]):.2f} | "
        print(line)
