"""A fixed corrective-maintenance penalty on top of the linear cost: how far is the tau*-quantile policy from the best multiplier, and does a penalty-aware rule recover?

Cost of a unit (divided by P0, c_o = 1):  [c_u (d - r)^+ + c_o (r - d)^+] / P0 + f * 1{r <= d},   f = F / P0, with d = p q, i.e. a failure before the replacement date costs F
in addition to the waiting cost; f = 1 means that a failure costs as much as wasting P0 cycles of life.  With f = 0 this is the cost of the article.

Population level (MLP out-of-fold alarms of the four fleets): the multiplier q_f that minimises the mean cost, the multiplier q* = tau*-quantile of the ratios, the coverage 1 - G(q) of both,
and the excess cost of q* over q_f (in units of P0 and in per cent of the best cost).  In ratio space the first-order condition is  G(q) + (f / kappa) g(q) = tau*  (for p = P0), so q_f < q*: a
fixed penalty calls for earlier replacement than the quantile rule.

Sequential level (random stream orders, first N0 = 20 units run to failure, then decisions with the pool of earlier units):
  tau-oracle : tau*-conformal quantile of exact labels;    f-oracle : multiplier that minimises the empirical penalised cost over exact labels;
  tau-KM     : closed-loop Kaplan-Meier policy of Definition 5 (labels censored by the policy);
  f-KM       : closed-loop policy that minimises  J(q) = kappa * int_0^q (G_hat - tau*) dx + f * G_hat(q)  over the multipliers q covered by the data, with G_hat the Kaplan-Meier estimate
               (J depends on G only through its values on [0, q], like the quantile rule, so it can be computed from right-censored labels).
usage: python fixed_penalty_check.py   (a few minutes)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np

P0, MINPOOL, NPERM, N0 = 60, 5, 100, 20
CUS, FS = [5, 20], [0.0, 0.5, 1.0, 2.0]
QGRID = np.arange(0.20, 1.50 + 1e-9, 0.005)


def load(ds):
    Z = np.load(f"{WORK}/oof_cmapss_{ds}.npz"); ids = sorted({k.split("__")[0] for k in Z.files}); A = {}
    for e in ids:
        r = Z[f"{e}__rul"]; p = np.exp(Z[f"{e}__data"]); T = len(r)
        ps = np.array([np.median(p[max(0, i - 4):i + 1]) for i in range(T)]); ix = np.where((np.arange(T) >= 10) & (ps <= P0))[0]
        if len(ix): A[e] = (ps[ix[0]], r[ix[0]])
    return A


def unit_cost(q, p, r, cu, f, co=1.0):
    d = p * q
    return (cu * np.maximum(d - r, 0) + co * np.maximum(r - d, 0)) / P0 + f * (r <= d)


def km_cdf_on_grid(v, ex, grid):
    """Kaplan-Meier estimate on a grid; v: observed values, ex: 1 if exact, 0 if censored.  Returns G_hat(grid) and the largest value covered by the data."""
    n = len(v); o = np.argsort(v, kind="stable"); vs, ds = v[o], ex[o].astype(float)
    S = np.cumprod(1.0 - ds / (n - np.arange(n)))
    idx = np.searchsorted(vs, grid, side="right")                    # number of observations <= x
    G = np.where(idx > 0, 1.0 - S[np.maximum(idx - 1, 0)], 0.0)
    return G, vs[-1]


def q_tau_exact(v, tau):
    n = len(v)
    return 1.0 if n < MINPOOL else np.sort(v)[min(max(int(np.ceil((n + 1) * tau)), 1), n) - 1]


def q_tau_km(v, ex, tau):
    n = len(v)
    if n < MINPOOL: return 1.0
    G, _ = km_cdf_on_grid(v, ex, np.sort(v)); t = min(tau * (n + 1) / n, 1.0); hit = np.where(G >= t)[0]
    return np.sort(v)[hit[0]] if len(hit) else 1.0


def q_f(v, ex, cu, f, tau):
    """Minimiser over the grid of J(q) = kappa * int_0^q (G_hat - tau) dx + f * G_hat(q); only multipliers up to the largest observed value are admissible."""
    n = len(v)
    if n < MINPOOL: return 1.0
    kap = cu + 1.0; G, top = km_cdf_on_grid(v, ex, QGRID); ok = QGRID <= top
    if not ok.any(): return 1.0
    dx = QGRID[1] - QGRID[0]; J = kap * np.cumsum(G - tau) * dx + f * G
    J = np.where(ok, J, np.inf); return QGRID[int(np.argmin(J))]


def run(method, cu, f, A, order):
    tau = 1.0 / (1.0 + cu); vals = [A[e][1] / A[e][0] for e in order[:N0]]; ex = [1.0] * N0; cost = []
    for e in order[N0:]:
        p, r = A[e]; v, x = np.array(vals), np.array(ex)
        if method == "tau-oracle": q = q_tau_exact(v, tau)
        elif method == "f-oracle": q = q_f(v, np.ones_like(x), cu, f, tau)
        elif method == "tau-KM": q = q_tau_km(v, x, tau)
        else: q = q_f(v, x, cu, f, tau)
        cost.append(float(unit_cost(q, p, r, cu, f)))
        if method.endswith("oracle") or p * q >= r: vals.append(r / p); ex.append(1.0)
        else: vals.append(q); ex.append(0.0)
    return np.mean(cost)


if __name__ == "__main__":
    fleets = ["FD001", "FD002", "FD003", "FD004"]; data = {ds: load(ds) for ds in fleets}
    print("Population level (MLP): best multiplier q_f under the penalty f, the quantile q*, their coverages 1 - G(q), and the excess cost of q* over q_f (units of P0; % of the best cost)")
    for cu in CUS:
        tau = 1 / (1 + cu)
        for f in FS:
            row = []
            for ds in fleets:
                A = data[ds]; p = np.array([A[e][0] for e in A]); r = np.array([A[e][1] for e in A]); rho = r / p
                C = np.array([unit_cost(q, p, r, cu, f).mean() for q in QGRID]); qf = QGRID[int(np.argmin(C))]
                qs = np.quantile(rho, tau, method="higher"); cs = unit_cost(qs, p, r, cu, f).mean()
                row.append((qf, qs, 1 - (rho < qf).mean(), 1 - (rho < qs).mean(), cs - C.min(), 100 * (cs - C.min()) / C.min()))
            m = np.mean(row, axis=0)
            print(f"c_u/c_o={cu:2d} f={f:3.1f}: q_f {m[0]:.3f}  q* {m[1]:.3f}  coverage at q_f {m[2]:.3f}  at q* {m[3]:.3f} (target {1 - tau:.3f})  excess of q* {m[4]:.3f} ({m[5]:.1f}%)   [mean of 4 fleets]")
    print("\nSequential level (mean cost per alarm / P0 including the penalty, mean over the four fleets and %d stream orders)" % NPERM)
    print("c_u/c_o  f  | tau-oracle  f-oracle | tau-KM  f-KM")
    for cu in CUS:
        for f in FS:
            res = {m: [] for m in ("tau-oracle", "f-oracle", "tau-KM", "f-KM")}
            for ds in fleets:
                A = data[ds]; E = sorted(A); rng0 = np.random.default_rng(11); orders = [list(rng0.permutation(E)) for _ in range(NPERM)]
                for m in res: res[m].append(np.mean([run(m, cu, f, A, o) for o in orders]))
            print(f"{cu:5d} {f:5.1f} | {np.mean(res['tau-oracle']):.3f}  {np.mean(res['f-oracle']):.3f}  | {np.mean(res['tau-KM']):.3f}  {np.mean(res['f-KM']):.3f}", flush=True)
