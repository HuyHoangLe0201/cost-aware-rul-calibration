"""Stress test of closed-loop calibration under policy-induced censoring (the self-confirming-conservatism scenario).
The initial calibration pool is deliberately biased low: of the first 2*N0 units of a random order only the N0 units with the smallest ratios are kept
(the others are discarded), so the first multiplier lies well below q*.  The rest of the stream is then decided with each labelling method:
oracle (exact labels always), naive, drop and Kaplan-Meier.  Reported: cost and coverage over all decisions, and coverage in the first 20 and in the later decisions.
usage: python adversarial_init.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np
from rulcost import CostAwareCalibrator, first_alarm

N0, NORD = 20, 60

def load(ds):
    Z = np.load(f"{WORK}/oof_cmapss_{ds}.npz"); P, R = [], []
    for e in sorted({k.split("__")[0] for k in Z.files}):
        a = first_alarm(np.exp(Z[f"{e}__data"]), 60.0)
        if a: P.append(a[1]); R.append(float(Z[f"{e}__rul"][a[0]]))
    return np.array(P), np.array(R)

def run(P, R, cu, method, biased, seed=5):
    rng = np.random.default_rng(seed); n = len(P); rho = R / P; cov_early, cov_late, costs, q_early, q_late = [], [], [], [], []
    for _ in range(NORD):
        order = rng.permutation(n)
        if biased:
            first = order[:2 * N0]; init = first[np.argsort(rho[first])[:N0]]; stream = order[2 * N0:]
        else:
            init, stream = order[:N0], order[N0:]
        cal = CostAwareCalibrator(cu, 1.0, method={"oracle": "conformal"}.get(method, method))
        for e in init: cal.update(P[e], R[e])
        c = []; cv = []; qs = []
        for e in stream:
            q = cal.multiplier(); d = P[e] * q; fail = d >= R[e]; qs.append(q)
            c.append((cu * max(d - R[e], 0) + max(R[e] - d, 0)) / 60.0); cv.append(R[e] >= d)
            if method == "oracle" or fail: cal.update(P[e], R[e])
            else: cal.update(P[e], d, censored=True)
        costs.append(np.mean(c)); cov_early.append(np.mean(cv[:20])); cov_late.append(np.mean(cv[40:])); q_early.append(np.mean(qs[:5])); q_late.append(np.mean(qs[-20:]))
    return np.mean(costs), np.mean(cov_early), np.mean(cov_late), np.mean(q_early), np.mean(q_late)

if __name__ == "__main__":
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        P, R = load(ds); rho = R / P
        for cu in (5, 20):
            tau = 1 / (1 + cu); qstar = np.quantile(rho, tau)
            print(f"{ds} c_u/c_o={cu:>2}  target coverage {1 - tau:.2f}, population q* ~ {qstar:.3f}, units {len(P)}")
            for init_kind in (False, True):
                line = f"   {'biased-low init ' if init_kind else 'random init     '}"
                for m in ("oracle", "naive", "drop", "kaplan-meier"):
                    c, ce, cl, qe, ql = run(P, R, cu, m, init_kind)
                    line += f"| {m[:6]:<6} cost {c:.3f} cov early {ce:.2f} late {cl:.2f} q {qe:.2f}->{ql:.2f} "
                print(line)
