"""Monte Carlo error from the finite number of random stream orders: per-order mean cost of the cumulative tau* policy (MLP, P0 = 60, N0 = 20),
its standard deviation over the 150 orders and the resulting standard error of the reported average.
usage: python order_variability.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import bisect, numpy as np
from rulcost import first_alarm, conformal_quantile

for ds in ("FD001", "FD002", "FD003", "FD004"):
    Z = np.load(f"{WORK}/oof_cmapss_{ds}.npz"); P, R = [], []
    for e in sorted({k.split("__")[0] for k in Z.files}):
        a = first_alarm(np.exp(Z[f"{e}__data"]), 60.0)
        if a: P.append(a[1]); R.append(float(Z[f"{e}__rul"][a[0]]))
    P, R = np.array(P), np.array(R); rho = R / P; n = len(P)
    for cu in (2, 20):
        tau = 1 / (1 + cu); rng = np.random.default_rng(11); per = []
        for _ in range(150):
            order = rng.permutation(n); srt = sorted(rho[order[:20]]); c = []
            for e in order[20:]:
                q = conformal_quantile(np.array(srt), tau); d = P[e] * q
                c.append((cu * max(d - R[e], 0) + max(R[e] - d, 0)) / 60.0); bisect.insort(srt, rho[e])
            per.append(np.mean(c))
        per = np.array(per); print(f"{ds} c_u/c_o={cu:>2}: mean {per.mean():.3f}, SD over orders {per.std(ddof=1):.3f}, SE of the mean over 150 orders {per.std(ddof=1) / np.sqrt(150):.4f}")
