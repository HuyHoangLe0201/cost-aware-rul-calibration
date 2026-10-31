"""Engine-specific time-scale calibration from the unit-rate law dRUL/dt = -1.
If p = lambda_e * r (multiplicative engine bias) then dp/dt = -lambda_e, so lambda_hat = -slope of p over the last K cycles; r_hat = p/lambda_hat.
Shrunk: lambda_c = lambda_hat**w."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
DS = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else ""
Z = np.load(f"{WORK}/oof_cmapss_{PFX}{DS}.npz"); ids = sorted({k.split("__")[0] for k in Z.files})
def lam(p, t, K):
    j = np.arange(max(0, t - K + 1), t + 1)
    if len(j) < 8 or p[j].max() > 118: return np.nan                  # not enough history / saturated (clipped) region
    s = np.polyfit(j, p[j], 1)[0]; return np.clip(-s, 0.25, 4.0)
for K in (15, 30):
    for w in (0.0, 0.25, 0.5, 1.0):
        E = []; E0 = []
        for e in ids:
            r = Z[f"{e}__rul"]; p = np.exp(Z[f"{e}__data"]); T = len(r)
            for t in range(25, T):
                if r[t] > 100: continue
                l = lam(p, t, K)
                if np.isnan(l): continue
                E.append(abs(np.log(p[t] / (l ** w) / r[t]))); E0.append(abs(np.log(p[t] / r[t])))
        print(f"{DS}{PFX} K={K} w={w}: n={len(E)} median|log err| {np.median(E):.3f} (raw {np.median(E0):.3f}) mean {np.mean(E):.3f} (raw {np.mean(E0):.3f})")
