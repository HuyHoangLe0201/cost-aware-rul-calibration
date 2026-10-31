"""Alarm-triggered decision protocol (observable-only).
Alarm at the first t>=30 where the causal-median (10) of predicted RUL <= P0. At the alarm the policy picks replacement lead time d.
Calibration pool = ratio r/p at the ALARM TIME of already-completed bearings (bearing-level exchangeable; one point per bearing).
Cost = (c_u*(d-r)^+ + c_o*(r-d)^+)/P0.  Conformal quantile index k=ceil((n+1)tau) (finite-sample)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
fn = sys.argv[1] if len(sys.argv) > 1 else "oof2_preds.npz"
Z = np.load(WORK + "/" + fn)
names = sorted({k.split("__")[0] for k in Z.files})
N0, WIN, NPERM, MINPOOL = 5, 8, 600, 3
rng = np.random.default_rng(7)

def alarm(model, P0):
    A = {}
    for n in names:
        r = Z[f"{n}__rul"]; p = np.exp(Z[f"{n}__{model}"]); T = len(r)
        ps = np.array([np.median(p[max(0, i - 9):i + 1]) for i in range(T)])
        idx = np.where((np.arange(T) >= 30) & (ps <= P0))[0]
        if len(idx): i = idx[0]; A[n] = (ps[i], r[i], i, T)
    return A

def cq(pool, tau):
    n = len(pool)
    if n < MINPOOL: return 1.0
    k = int(np.ceil((n + 1) * tau)); k = min(max(k, 1), n)
    return np.sort(pool)[k - 1]

def run(model, P0, cu, co=1.0):
    A = alarm(model, P0); tau = co / (co + cu)
    pols = ["point", "fixed90", "tau_static", "tau_cum", "tau_roll"]
    per = {k: {n: [] for n in A} for k in pols}; cov = {k: [] for k in pols}
    for _ in range(NPERM):
        order = list(rng.permutation(names)); init, test = order[:N0], order[N0:]
        for j, n in enumerate(test):
            if n not in A: continue
            done = init + test[:j]; p, r = A[n][0], A[n][1]
            ratios = lambda ns: np.array([A[m][1] / A[m][0] for m in ns if m in A])
            ds = {"point": p, "fixed90": p * cq(ratios(done[-WIN:]), 0.10), "tau_static": p * cq(ratios(init), tau),
                  "tau_cum": p * cq(ratios(done), tau), "tau_roll": p * cq(ratios(done[-WIN:]), tau)}
            for k, d in ds.items():
                per[k][n].append((cu * max(d - r, 0) + co * max(r - d, 0)) / P0); cov[k].append(r >= d)
    bm = {k: np.array([np.mean(v) for v in per[k].values() if v]) for k in pols}   # bearing-level mean cost
    return A, bm, {k: np.mean(v) for k, v in cov.items()}

def boot(a, b=None, B=4000):
    r = np.random.default_rng(0); n = len(a); idx = r.integers(0, n, (B, n))
    s = a[idx].mean(1) if b is None else (a - b)[idx].mean(1)
    return np.percentile(s, [2.5, 97.5])

if __name__ == "__main__":
    print("file", fn)
    for P0 in (100, 200, 400):
        for model in ("data", "hybrid"):
            A = alarm(model, P0)
            err = np.median([abs(np.log(A[n][0] / A[n][1])) for n in A])
            print(f"\n--- P0={P0} {model}: alarmed {len(A)}/17, median|log(p/r)| at alarm {err:.2f}")
            for cu in (2, 5, 10, 20):
                _, bm, cv = run(model, P0, cu)
                tau = 1 / (1 + cu)
                s = f"  cu/co={cu:>2} (target cover {1-tau:.2f}) cost: " + " ".join(f"{k}={bm[k].mean():.2f}" for k in bm)
                s += " | cov: " + " ".join(f"{k}={cv[k]:.2f}" for k in cv if k != "point")
                d = bm["fixed90"] - bm["tau_cum"]; lo, hi = boot(d)
                s += f" | fixed90-tau_cum {d.mean():+.2f} [{lo:+.2f},{hi:+.2f}]"
                print(s)
