"""Paired decision comparison on PRONOSTIA (alarm protocol of decision_eval2.py): hybrid-file model vs data-file model.
usage: python pron_compare.py oof_pron_A.npz"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
fn = sys.argv[1]; sys.argv = ["x", fn]
import decision_eval2 as D
def boot(x, B=4000):
    r = np.random.default_rng(0); idx = r.integers(0, len(x), (B, len(x))); s = x[idx].mean(1); return x.mean(), *np.percentile(s, [2.5, 97.5])
print("file", fn, "| paired bearing-level cost difference (hybrid - data) with tau_cum; negative favours the hybrid-file feature set")
for P0 in (100, 200, 400):
    Ad, Ah = D.alarm("data", P0), D.alarm("hybrid", P0); common = sorted(set(Ad) & set(Ah))
    ed = np.median([abs(np.log(Ad[n][0] / Ad[n][1])) for n in common]); eh = np.median([abs(np.log(Ah[n][0] / Ah[n][1])) for n in common])
    print(f"P0={P0}: alarmed data {len(Ad)} hybrid {len(Ah)} common {len(common)}; median|log(p/r)| at alarm data {ed:.2f} hybrid {eh:.2f}")
    for cu in (2, 5, 10, 20):
        res = {}
        for model in ("data", "hybrid"):
            _, bm, cv = D.run(model, P0, cu); res[model] = (bm, cv)
        # bearing-level arrays are ordered by the alarmed set; re-run restricted to common bearings for pairing
        def percost(model):
            A = D.alarm(model, P0); tau = 1 / (1 + cu); rng = np.random.default_rng(7); per = {n: [] for n in common}
            for _ in range(D.NPERM):
                order = list(rng.permutation(D.names)); init, test = order[:D.N0], order[D.N0:]
                for j, n in enumerate(test):
                    if n not in per: continue
                    done = init + test[:j]; p, r = A[n][0], A[n][1]; ra = np.array([A[m][1] / A[m][0] for m in done if m in A]); q = D.cq(ra, tau); d = p * q
                    per[n].append((cu * max(d - r, 0) + max(r - d, 0)) / P0)
            return np.array([np.mean(v) for v in per.values() if v])
        a, b = percost("hybrid"), percost("data"); k = min(len(a), len(b)); m, lo, hi = boot(a[:k] - b[:k])
        print(f"   cu/co={cu:>2}: cost data {b.mean():.2f} hybrid {a.mean():.2f} | paired {m:+.2f} [{lo:+.2f},{hi:+.2f}] (n={k} bearings)")
