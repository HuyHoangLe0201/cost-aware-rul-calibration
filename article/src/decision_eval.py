"""Fleet-stream evaluation: calibrated RUL quantile -> replacement time d (Proposition 1).
Cost: c_u*(d-r)^+ + c_o*(r-d)^+ , normalised by true life.  Conditioning for calibration uses OBSERVABLE only
(bin of the point prediction), never true life stage. Decision points are placed at 20/40/60/80% of true life
purely as an evaluation protocol."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
Z = np.load(f"{WORK}/oof_preds_s{seed}.npz")
names = sorted({k.split("__")[0] for k in Z.files})
STAGES = [0.2, 0.4, 0.6, 0.8]
EDGES = [300, 800]                      # bins of point RUL prediction (snapshots)
N0, WIN, NPERM = 5, 8, 400
rng = np.random.default_rng(1)

def prep(model):
    B = {}
    for n in names:
        r = Z[f"{n}__rul"]; p = np.exp(Z[f"{n}__{model}"]); T = len(r)
        sub = np.arange(0, T, 5)                       # calibration snapshots (thinned)
        B[n] = dict(r=r, p=p, T=T, cal_bin=np.digitize(p[sub], EDGES), cal_ratio=(r / p)[sub],
                    idx=[min(int(s * T), T - 1) for s in STAGES])
    return B

def q(pool, tau):
    if len(pool) < 20: return np.nan
    t = min(1.0, tau * (len(pool) + 1) / len(pool))
    return np.quantile(pool, t, method="higher")

def run(model, cu, co):
    B = prep(model); tau = co / (co + cu)
    pol = ["point", "fixed90", "tau_static", "tau_cum", "tau_roll", "oracle"]
    cost = {k: [] for k in pol}; cov = {k: [] for k in pol[1:5]}
    for _ in range(NPERM):
        order = list(rng.permutation(names))
        init, test = order[:N0], order[N0:]
        for j, n in enumerate(test):
            done_all = init + test[:j]; done_win = done_all[-WIN:]
            b = B[n]
            for i in b["idx"]:
                p, r = b["p"][i], b["r"][i]; k = int(np.digitize(p, EDGES))
                def pool(ns): return np.concatenate([B[m]["cal_ratio"][B[m]["cal_bin"] == k] for m in ns])
                ds = {"point": p, "oracle": r}
                for name, ns, t in (("fixed90", done_win, 0.10), ("tau_static", init, tau),
                                    ("tau_cum", done_all, tau), ("tau_roll", done_win, tau)):
                    qq = q(pool(ns), t); ds[name] = p * (qq if np.isfinite(qq) else 1.0)
                for name, d in ds.items():
                    cost[name].append((cu * max(d - r, 0) + co * max(r - d, 0)) / b["T"])
                    if name in cov: cov[name].append(r >= d)
    return {k: np.mean(v) for k, v in cost.items()}, {k: np.mean(v) for k, v in cov.items()}

if __name__ == "__main__":
    for model in ("data", "hybrid"):
        B = prep(model)
        mae = np.mean([np.mean(np.abs(B[n]["p"] - B[n]["r"]) / B[n]["r"]) for n in names])
        print(f"\n===== predictor: {model}   (mean rel. abs. error over all snapshots {mae:.3f}) =====")
        for cu in (2, 5, 10, 20):
            c, v = run(model, cu, 1.0)
            tau = 1 / (1 + cu)
            print(f"c_u/c_o={cu:>2}  tau*={tau:.3f} (target cover {1-tau:.2f}) | " +
                  " ".join(f"{k}={c[k]:.3f}" for k in c) + " | cover: " + " ".join(f"{k}={v[k]:.2f}" for k in v))
