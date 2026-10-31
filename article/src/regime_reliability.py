"""Calibration by operating regime on FD002 and FD004 (six operating conditions): coverage and cost of pooled cumulative calibration and of calibration
stratified by the regime in force at the alarm cycle (regimes: k-means on the three operating settings, as in the predictor).  Same stream protocol as the
article (N0 = 20, 60 random orders, P0 = 60, MLP out-of-fold predictions).  Fault modes are not labelled in C-MAPSS and cannot be analysed.
usage: python regime_reliability.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np
from sklearn.cluster import KMeans
from rulcost import first_alarm, CostAwareCalibrator

ROOT = CMAPSS_ROOT
N0, NORD, CUS = 20, 60, (5, 20)

for ds in ("FD002", "FD004"):
    raw = np.loadtxt(f"{ROOT}train_{ds}.txt"); eng = raw[:, 0].astype(int); cond = KMeans(6, n_init=5, random_state=0).fit_predict(raw[:, 2:5])
    Z = np.load(f"{WORK}/oof_cmapss_{ds}.npz"); P, R, G = [], [], []
    for e in sorted({int(k.split("__")[0][1:]) for k in Z.files}):
        a = first_alarm(np.exp(Z[f"e{e}__data"]), 60.0)
        if a: P.append(a[1]); R.append(float(Z[f"e{e}__rul"][a[0]])); G.append(int(cond[eng == e][a[0]]))
    P, R, G = np.array(P), np.array(R), np.array(G); n = len(P); print(f"=== {ds}: {n} alarmed units; units per regime at the alarm {np.bincount(G, minlength=6).tolist()}; median ratio by regime {[round(float(np.median(R[G == g] / P[G == g])), 2) for g in range(6)]}")
    for cu in CUS:
        rng = np.random.default_rng(11); hit = {k: np.zeros(6) for k in ("pool", "strat")}; cnt = np.zeros(6); cost = {k: np.zeros(6) for k in hit}
        for _ in range(NORD):
            order = rng.permutation(n); pool = CostAwareCalibrator(cu, 1.0); st = CostAwareCalibrator(cu, 1.0)
            for e in order[:N0]: pool.update(P[e], R[e]); st.update(P[e], R[e], group=G[e])
            for e in order[N0:]:
                for k, cal, grp in (("pool", pool, None), ("strat", st, G[e])):
                    d = P[e] * cal.multiplier(grp); hit[k][G[e]] += R[e] >= d; cost[k][G[e]] += (cu * max(d - R[e], 0) + max(R[e] - d, 0)) / 60.0
                cnt[G[e]] += 1; pool.update(P[e], R[e]); st.update(P[e], R[e], group=G[e])
        tgt = 1 - 1 / (1 + cu)
        print(f" c_u/c_o={cu} (target coverage {tgt:.2f}): coverage by regime pooled " + " ".join(f"{h / c:.2f}" for h, c in zip(hit['pool'], cnt)) + " | stratified " + " ".join(f"{h / c:.2f}" for h, c in zip(hit['strat'], cnt)))
        print(f"        cost by regime pooled " + " ".join(f"{h / c:.2f}" for h, c in zip(cost['pool'], cnt)) + " | stratified " + " ".join(f"{h / c:.2f}" for h, c in zip(cost['strat'], cnt)) + f" | overall pooled {cost['pool'].sum() / cnt.sum():.3f} stratified {cost['strat'].sum() / cnt.sum():.3f}")
