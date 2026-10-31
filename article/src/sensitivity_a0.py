"""Sensitivity of the policy to assumption A0 (the ratio rho = r/p is independent of the point prediction p).
Without A0 the best multiplier minimises E[p (c_u (q-rho)^+ + c_o (rho-q)^+)], i.e. it is the tau*-quantile of rho under weights proportional to p
(E[p 1{rho<=q}] / E[p] = tau*).  Two checks on the MLP out-of-fold predictions:
 (1) the alarm threshold of the article, P0 = 60: spread of p, rank correlation of p and rho, ordinary against p-weighted quantile;
 (2) a deliberately heterogeneous fleet in which every unit has its own alarm threshold P0_i ~ U(30, 90) (p varies by a factor of three).
usage: python sensitivity_a0.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np
from scipy.stats import spearmanr
from rulcost import conformal_quantile, first_alarm

CUS = (2, 5, 10, 20); NORD, N0 = 60, 20

def load(ds, thr):
    Z = np.load(f"{WORK}/oof_cmapss_{ds}.npz"); ids = sorted({k.split("__")[0] for k in Z.files}); P, R = [], []
    for j, e in enumerate(ids):
        a = first_alarm(np.exp(Z[f"{e}__data"]), thr[j] if hasattr(thr, "__len__") else thr)
        if a: P.append(a[1]); R.append(float(Z[f"{e}__rul"][a[0]]))
    return np.array(P), np.array(R)

def wquantile(rho, w, tau):
    o = np.argsort(rho); cw = np.cumsum(w[o]) / w.sum(); return rho[o][min(np.searchsorted(cw, tau), len(o) - 1)]

def cost_total(q, p, rho, cu): return float((p * (cu * np.maximum(q - rho, 0) + np.maximum(rho - q, 0))).sum() / p.sum())

def static_report(label, P, R):
    rho = R / P; print(f"  {label}: n={len(P)}, p quantiles 0/5/50/95/100% = {np.round(np.quantile(P, [0, .05, .5, .95, 1]), 1).tolist()}, Spearman(p, rho) = {spearmanr(P, rho)[0]:+.2f}")
    for cu in CUS:
        tau = 1 / (1 + cu); qo = conformal_quantile(rho, tau); qw = wquantile(rho, P, tau)
        print(f"     c_u/c_o={cu:>2}: q ordinary {qo:.3f}, q p-weighted {qw:.3f}; p-weighted cost of ordinary vs weighted: {cost_total(qo, P, rho, cu):.4f} vs {cost_total(qw, P, rho, cu):.4f} ({100 * (cost_total(qo, P, rho, cu) / cost_total(qw, P, rho, cu) - 1):+.2f}%)")

def stream(P, R, cu, seed=11):
    rho = R / P; tau = 1 / (1 + cu); rng = np.random.default_rng(seed); n = len(P); ca = {"ord": 0.0, "wt": 0.0, "pt": 0.0}; sp = 0.0
    for _ in range(NORD):
        order = rng.permutation(n); init, test = list(order[:N0]), order[N0:]; done = list(init)
        for e in test:
            d = np.array(done); qo = conformal_quantile(rho[d], tau); qw = wquantile(rho[d], P[d], tau) if len(d) >= 5 else 1.0
            for k, q in (("ord", qo), ("wt", qw)): ca[k] += P[e] * (cu * max(q - rho[e], 0) + max(rho[e] - q, 0))
            sp += P[e]; done.append(e)
    return {k: v / sp for k, v in ca.items() if k != "pt"}

if __name__ == "__main__":
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        print(f"=== {ds}")
        P, R = load(ds, 60.0); static_report("article protocol (P0 = 60)", P, R)
        rng = np.random.default_rng(7); Z = np.load(f"{WORK}/oof_cmapss_{ds}.npz"); n = len({k.split('__')[0] for k in Z.files})
        thr = rng.uniform(30, 90, n); P2, R2 = load(ds, thr); static_report("heterogeneous thresholds U(30, 90)", P2, R2)
        for cu in (2, 20):
            r = stream(P2, R2, cu); print(f"     stream, heterogeneous, c_u/c_o={cu:>2}: total cost / total p: ordinary {r['ord']:.4f}, p-weighted {r['wt']:.4f} ({100 * (r['ord'] / r['wt'] - 1):+.2f}%)")
