"""C-MAPSS fleet decision evaluation (observable-only alarm protocol; see decision_eval2.py).
Streams: 'random' (exchangeable) and 'drift' (engines ordered by total life + noise: synthetic non-stationarity stress test).
usage: python fleet_eval.py FD001"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
DS = sys.argv[1] if len(sys.argv) > 1 else "FD001"
import os
PFX = os.environ.get("PFX", "")           # e.g. PFX=cnn_ -> oof_cmapss_cnn_FD001.npz (only "data" model)
MODELS = ("data", "hybrid") if (os.environ.get("BOTH") or not PFX) else ("data",)
Z = np.load(f"{WORK}/oof_cmapss_{PFX}{DS}.npz")
ids = sorted({k.split("__")[0] for k in Z.files}); N = len(ids)
N0, WIN, MINPOOL, CUS = 20, 40, 5, (2, 5, 10, 20)

def alarms(model, P0):
    A = {}
    for e in ids:
        r = Z[f"{e}__rul"]; p = np.exp(Z[f"{e}__{model}"]); T = len(r)
        ps = np.array([np.median(p[max(0, i - 4):i + 1]) for i in range(T)])
        ix = np.where((np.arange(T) >= 10) & (ps <= P0))[0]
        if len(ix): i = ix[0]; A[e] = (ps[i], r[i], T)
    return A

def cq(pool, tau):
    n = len(pool)
    if n < MINPOOL: return 1.0
    k = min(max(int(np.ceil((n + 1) * tau)), 1), n); return np.sort(pool)[k - 1]

def order_stream(kind, A, rng):
    es = list(ids)
    if kind == "random": return list(rng.permutation(es))
    life = np.array([len(Z[f"{e}__rul"]) for e in es], float)
    key = (np.argsort(np.argsort(life)) / len(es)) + rng.normal(0, 0.15, len(es))   # rank + noise
    return [es[i] for i in np.argsort(key)]

def run(model, P0, cu, kind, nperm, co=1.0):
    A = alarms(model, P0); tau = co / (co + cu)
    pol = ["point", "fixed90", "tau_static", "tau_cum", "tau_roll"]
    per = {k: {e: [] for e in A} for k in pol}; cov = {k: [] for k in pol}
    rng = np.random.default_rng(11)                      # same stream orders for every model / policy
    for _ in range(nperm):
        order = order_stream(kind, A, rng); init, test = order[:N0], order[N0:]
        ra = {m: A[m][1] / A[m][0] for m in A}
        for j, e in enumerate(test):
            if e not in A: continue
            done = init + test[:j]; p, r = A[e][0], A[e][1]
            R = lambda ns: np.array([ra[m] for m in ns if m in ra])
            ds = {"point": p, "fixed90": p * cq(R(done[-WIN:]), .10), "tau_static": p * cq(R(init), tau),
                  "tau_cum": p * cq(R(done), tau), "tau_roll": p * cq(R(done[-WIN:]), tau)}
            for k, d in ds.items():
                per[k][e].append((cu * max(d - r, 0) + co * max(r - d, 0)) / P0); cov[k].append(r >= d)
    em = {k: np.array([np.mean(v) if v else np.nan for v in per[k].values()]) for k in pol}
    ok = ~np.isnan(em["point"]); em = {k: v[ok] for k, v in em.items()}
    return A, em, {k: np.mean(v) for k, v in cov.items()}

def boot(x, B=4000):
    r = np.random.default_rng(0); idx = r.integers(0, len(x), (B, len(x))); s = x[idx].mean(1)
    return x.mean(), *np.percentile(s, [2.5, 97.5])

if __name__ == "__main__":
    for P0 in (30, 60):
        for kind, nperm in (("random", 150), ("drift", 100)):
            cache = {}
            for model in MODELS:
                A = alarms(model, P0); err = np.median([abs(np.log(A[e][0] / A[e][1])) for e in A])
                print(f"\n=== {DS} P0={P0} stream={kind} model={model}: alarmed {len(A)}/{N}, median|log(p/r)| {err:.2f}")
                for cu in CUS:
                    A_, em, cv = run(model, P0, cu, kind, nperm); cache[(model, cu)] = em
                    tau = 1 / (1 + cu)
                    line = f" cu/co={cu:>2} cover(target {1-tau:.4f}): " + " ".join(f"{k}={cv[k]:.3f}" for k in cv if k != "point")
                    line += " | cost: " + " ".join(f"{k}={em[k].mean():.3f}" for k in em)
                    m, lo, hi = boot(em["fixed90"] - em["tau_cum"]); line += f" | fixed90-tau_cum {m:+.2f}[{lo:+.2f},{hi:+.2f}]"
                    m, lo, hi = boot(em["tau_roll"] - em["tau_cum"]); line += f" | roll-cum {m:+.2f}[{lo:+.2f},{hi:+.2f}]"
                    print(line)
            if len(MODELS) < 2: continue
            print(f"--- paired hybrid-data (tau_cum cost; negative favours hybrid) {DS} P0={P0} {kind}")
            for cu in CUS:
                a, b = cache[("hybrid", cu)]["tau_cum"], cache[("data", cu)]["tau_cum"]
                n = min(len(a), len(b))      # engines alarmed by both models align only if sets equal; guard
                if len(a) == len(b): m, lo, hi = boot(a - b); print(f" cu/co={cu:>2}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
                else: print(f" cu/co={cu:>2}: alarm sets differ ({len(a)} vs {len(b)}), unpaired means {a.mean():.2f} vs {b.mean():.2f}")
