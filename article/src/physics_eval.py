"""Physics-consistency-stratified (Mondrian) conformal calibration.
Observable at alarm: I = std_j log(max(RULhat(t-j)-j,1)), j<K (kinematic inconsistency of the forecast path under dRUL/dt=-1).
Strata: NB bins by I; bin edges from the initial-calibration engines' I (quantiles). Pool calibration ratios within stratum (cum), conformal quantile.
Compare tau_cum (pooled) vs tau_mond. Same stream orders, paired bootstrap over engines.
usage: python physics_eval.py FD002 [cnn_]"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, os, numpy as np
DS = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else ""; K = int(os.environ.get("K", 10)); NB = int(os.environ.get("NB", 2))
Z = np.load(f"{WORK}/oof_cmapss_{PFX}{DS}.npz"); ids = sorted({k.split("__")[0] for k in Z.files})
P0, N0, MINPOOL, NPERM = 60, 20, 5, 100
def at_alarm(e):
    r = Z[f"{e}__rul"]; p = np.exp(Z[f"{e}__data"]); T = len(r)
    ps = np.array([np.median(p[max(0, i - 4):i + 1]) for i in range(T)]); ix = np.where((np.arange(T) >= 10) & (ps <= P0))[0]
    if not len(ix): return None
    t = ix[0]; j = np.arange(min(K, t + 1)); s = np.maximum(p[t - j] - j, 1.0)
    return ps[t], r[t], float(np.std(np.log(s)))
A = {e: a for e in ids if (a := at_alarm(e)) is not None}; E = list(A)
def cq(pool, tau):
    n = len(pool)
    if n < MINPOOL: return None
    return np.sort(pool)[min(max(int(np.ceil((n + 1) * tau)), 1), n) - 1]
def boot(x, B=4000):
    r = np.random.default_rng(0); idx = r.integers(0, len(x), (B, len(x))); s = x[idx].mean(1); return x.mean(), *np.percentile(s, [2.5, 97.5])
# value of information (descriptive): spread of log rho by inconsistency tercile
I = np.array([A[e][2] for e in E]); L = np.log([A[e][1] / A[e][0] for e in E]); q3 = np.quantile(I, [1/3, 2/3]); b3 = np.digitize(I, q3)
print(f"{DS}{PFX} K={K}: corr(I, |log rho - median|) = {np.corrcoef(I, np.abs(L - np.median(L)))[0,1]:+.2f}; sd(log rho) by I-tercile: " + " ".join(f"{L[b3==b].std():.3f}" for b in range(3)) + f" | pooled {L.std():.3f}")
for cu in (2, 5, 10, 20):
    tau = 1 / (1 + cu); rng = np.random.default_rng(11); per = {"pooled": {e: [] for e in E}, "mond": {e: [] for e in E}}; cov = {"pooled": [], "mond": []}
    for _ in range(NPERM):
        order = list(rng.permutation(E)); init, test = order[:N0], order[N0:]
        edges = np.quantile([A[e][2] for e in init], np.linspace(0, 1, NB + 1)[1:-1]) if NB > 1 else []
        bin_of = {e: int(np.digitize(A[e][2], edges)) for e in E}
        for j, e in enumerate(test):
            done = init + test[:j]; p, r, _i = A[e]; ra = lambda ns: np.array([A[m][1] / A[m][0] for m in ns])
            qp = cq(ra(done), tau); same = [m for m in done if bin_of[m] == bin_of[e]]; qm = cq(ra(same), tau)
            qp = 1.0 if qp is None else qp; qm = qp if qm is None else qm
            for k, q in (("pooled", qp), ("mond", qm)):
                d = p * q; per[k][e].append((cu * max(d - r, 0) + max(r - d, 0)) / P0); cov[k].append(r >= d)
    em = {k: np.array([np.mean(v) for v in per[k].values()]) for k in per}
    m, lo, hi = boot(em["pooled"] - em["mond"])
    print(f"  cu/co={cu:>2} (target {1-tau:.2f}) cover pooled {np.mean(cov['pooled']):.2f} mond {np.mean(cov['mond']):.2f} | cost pooled {em['pooled'].mean():.3f} mond {em['mond'].mean():.3f} | gain(pooled-mond) {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
