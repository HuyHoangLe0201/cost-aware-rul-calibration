"""Exploratory: does the kinematic law RUL(t+j)=RUL(t)-j carry usable signal at inference?
 shifted forecasts s_j(t) = RULhat(t-j) - j  (j=0..K-1), fused = median_j s_j, inconsistency = std_j log s_j."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
DS = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else ""
Z = np.load(f"{WORK}/oof_cmapss_{PFX}{DS}.npz"); ids = sorted({k.split("__")[0] for k in Z.files})
def series(e, K):
    r = Z[f"{e}__rul"]; p = np.exp(Z[f"{e}__data"]); T = len(r)
    f = np.empty(T); inc = np.empty(T)
    for t in range(T):
        j = np.arange(min(K, t + 1)); s = np.maximum(p[t - j] - j, 1.0)
        f[t] = np.median(s); inc[t] = np.std(np.log(s)) if len(s) > 1 else 0.0
    return r, p, f, inc
for K in (1, 5, 10, 20):
    errs_raw, errs_f, X, Y = [], [], [], []
    for e in ids:
        r, p, f, inc = series(e, K); T = len(r); m = (r <= 100) & (np.arange(T) >= 25)
        errs_raw.append(np.abs(np.log(p[m] / r[m]))); errs_f.append(np.abs(np.log(f[m] / r[m]))); X.append(inc[m]); Y.append(np.abs(np.log(f[m] / r[m])))
    er, ef = np.concatenate(errs_raw), np.concatenate(errs_f); X, Y = np.concatenate(X), np.concatenate(Y)
    from scipy.stats import spearmanr
    print(f"{DS}{PFX} K={K:>2}: median|log err| raw {np.median(er):.3f} -> kinematic-fused {np.median(ef):.3f} | mean {er.mean():.3f}->{ef.mean():.3f} | spearman(inconsistency, |err|) {spearmanr(X, Y)[0]:+.2f}")
