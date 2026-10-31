"""Policy suite: every baseline policy x predictor on the alarm-triggered fleet stream (random stream, P0=60, N0=20, 150 orders, same orders for all predictors).
Policies (multiplier q applied to the point prediction p; replacement date d = p*q):
 point (q=1) | a50, a70, a85 (safety factors q=0.5/0.7/0.85) | f90 (conformal quantile at level 0.10 over the last 40 units, as in the main tables) | gauss (log-normal fit to the pool, level tau*) |
 static (conformal tau*, initial 20 units only) | cum (conformal tau*, all completed units; PROPOSED) | roll (conformal tau*, last 40) | oracle (leave-one-out quantile of all alarm ratios)
Also records: sweeps for the risk-waste frontier and reliability curves (cum, gauss, static, safety factor), per-engine mean costs, alarm ratios.
usage: python policy_suite.py            (runs every predictor x fleet and writes suite/*.pkl)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, os, pickle, numpy as np
from multiprocessing import Pool
from scipy.special import ndtri
P0, N0, WIN, NPERM, MINPOOL = 60, 20, 40, 150, 5
CUS = (2, 5, 10, 20); POL = ["point", "a50", "a70", "a85", "f90", "gauss", "static", "cum", "roll", "oracle"]
TAUS = np.linspace(0.03, 0.62, 21); ALPH = np.linspace(0.30, 1.20, 21)
FILES = {"ridge": ("oof_cmapss_ridge_{}.npz", "data"), "hgb": ("oof_cmapss_hgb_{}.npz", "data"), "rf": ("oof_cmapss_rf_{}.npz", "data"), "mlp": ("oof_cmapss_{}.npz", "data"),
         "gru": ("oof_cmapss_gru_{}.npz", "data"), "cnn": ("oof_cmapss_cnn_{}.npz", "data"), "paris": ("oof_cmapss_{}.npz", "hybrid"), "ssm": ("oof_cmapss_ssmL3_b1.5_{}.npz", "hybrid")}
def alarms(Z, key, ids):
    A = {}
    for e in ids:
        r = Z[f"{e}__rul"]; p = np.exp(Z[f"{e}__{key}"]); T = len(r)
        ps = np.array([np.median(p[max(0, i - 4):i + 1]) for i in range(T)]); ix = np.where((np.arange(T) >= 10) & (ps <= P0))[0]
        if len(ix): i = ix[0]; A[e] = (ps[i], r[i])
    return A
def qconf(srt, tau):
    n = len(srt)
    if n < MINPOOL: return 1.0
    return srt[min(max(int(np.ceil((n + 1) * tau)), 1), n) - 1]
def qconf_vec(srt, taus):
    n = len(srt)
    if n < MINPOOL: return np.ones(len(taus))
    k = np.minimum(np.maximum(np.ceil((n + 1) * taus).astype(int), 1), n) - 1; return srt[k]
def qgauss(pool, taus):
    if len(pool) < MINPOOL: return np.ones(np.size(taus))
    l = np.log(pool); return np.exp(l.mean() + l.std() * ndtri(taus))
def run(args):
    name, ds = args; fn, key = FILES[name]; path = WORK + "/" + fn.format(ds)
    if not os.path.exists(path): return name, ds, None
    Z = np.load(path); ids = sorted({k.split("__")[0] for k in Z.files}); A = alarms(Z, key, ids); ra = {e: A[e][1] / A[e][0] for e in A}
    allsorted = np.sort(np.array(list(ra.values()))); pos = {e: int(np.searchsorted(allsorted, ra[e])) for e in ra}
    rng = np.random.default_rng(11); cost = {p: {c: {e: [0.0, 0] for e in A} for c in CUS} for p in POL}
    sw = {k: np.zeros((len(TAUS), 4)) for k in ("cum", "gauss", "static")}; sw["alpha"] = np.zeros((len(ALPH), 4)); nd = 0
    ovw = {c: {p: np.zeros(3) for p in POL} for c in CUS}                     # overrun, waste, count (all test decisions) per policy / ratio
    for _ in range(NPERM):
        order = list(rng.permutation(ids)); init, test = order[:N0], order[N0:]; done = list(init); srt_cum = sorted(ra[m] for m in init if m in ra); srt_init = np.array(srt_cum)
        for e in test:
            if e in A:
                p, r = A[e]; arr_cum = np.array(srt_cum); roll = np.sort(np.array([ra[m] for m in done[-WIN:] if m in ra])); nd += 1
                # sweeps (cum, gauss, static) and safety factors
                for key2, qs in (("cum", qconf_vec(arr_cum, TAUS)), ("gauss", qgauss(arr_cum, TAUS)), ("static", qconf_vec(srt_init, TAUS))):
                    d = p * qs; sw[key2][:, 0] += np.maximum(d - r, 0) / P0; sw[key2][:, 1] += np.maximum(r - d, 0) / P0; sw[key2][:, 2] += (r >= d); sw[key2][:, 3] += 1
                d = p * ALPH; sw["alpha"][:, 0] += np.maximum(d - r, 0) / P0; sw["alpha"][:, 1] += np.maximum(r - d, 0) / P0; sw["alpha"][:, 2] += (r >= d); sw["alpha"][:, 3] += 1
                # LOO oracle ranks
                for cu in CUS:
                    tau = 1.0 / (1 + cu); qs = {"point": 1.0, "a50": 0.5, "a70": 0.7, "a85": 0.85, "f90": qconf(roll, 0.10), "gauss": float(qgauss(arr_cum, tau)),
                                              "static": qconf(srt_init, tau), "cum": qconf(arr_cum, tau), "roll": qconf(roll, tau)}
                    n1 = len(allsorted) - 1; k = min(max(int(np.ceil((n1 + 1) * tau)), 1), n1) - 1; qs["oracle"] = allsorted[k if k < pos[e] else k + 1]
                    for pol in POL:
                        d = p * qs[pol]; over = max(d - r, 0) / P0; waste = max(r - d, 0) / P0; c = cu * over + waste; cost[pol][cu][e][0] += c; cost[pol][cu][e][1] += 1
                        ovw[cu][pol] += (over, waste, 1)
                import bisect; bisect.insort(srt_cum, ra[e])
            done.append(e)
    per = {p: {c: np.array([v[0] / v[1] if v[1] else np.nan for v in cost[p][c].values()]) for c in CUS} for p in POL}
    sw = {k: dict(over=v[:, 0] / v[:, 3], waste=v[:, 1] / v[:, 3], cov=v[:, 2] / v[:, 3]) for k, v in sw.items()}
    ovw = {c: {p: (v[0] / v[2], v[1] / v[2]) for p, v in ovw[c].items()} for c in CUS}
    out = dict(name=name, ds=ds, n_alarm=len(A), n=len(ids), engines=list(A), per=per, sweeps=sw, ovw=ovw, ratios=np.array([ra[e] for e in A]), p_a=np.array([A[e][0] for e in A]), r_a=np.array([A[e][1] for e in A]), TAUS=TAUS, ALPH=ALPH)
    os.makedirs(WORK + "/suite", exist_ok=True); pickle.dump(out, open(f"{WORK}/suite/{name}_{ds}.pkl", "wb")); return name, ds, len(A)
if __name__ == "__main__":
    names = sys.argv[1:] or list(FILES); jobs = [(n, d) for d in ("FD001", "FD003", "FD002", "FD004") for n in names]
    with Pool(4) as p:
        for name, ds, na in p.imap_unordered(run, jobs): print(name, ds, "alarmed", na, flush=True)
