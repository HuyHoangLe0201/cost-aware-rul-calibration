"""Early phase of the closed loop: how much does a small or biased initial pool cost, and how fast does Kaplan-Meier calibration recover?

For every fleet (MLP out-of-fold predictions, P0 = 60) and every random stream order, the first N0 units are run to failure (exact labels) and the
following ones are decided sequentially by (a) the oracle (cumulative conformal quantile of exact labels, every later unit also observed exactly) and
(b) the closed-loop Kaplan-Meier policy of Definition 5 (units replaced before failure are right-censored at the replacement multiplier).
Only the two large fleets (FD002: 260 engines, FD004: 249) are used, so that the streams are longer than m + 100 decisions for every N0; FD001 and FD003 (100 engines) are too short.
Reported, per N0 and cost ratio, as means over the stream orders and the two fleets:
  - excess of Kaplan-Meier over the oracle in the cumulative cost of the first m decisions (m = 25, 50, 100), in units of P0 (cost per alarm divided by P0, summed over decisions);
  - the "learning cost" of the oracle itself: cumulative cost of its first m decisions minus m times its mean cost over the last 100 decisions of the stream;
  - the same excess in per cent of the oracle's cumulative cost over the first m decisions;
  - the share of the first 50 decisions at which the Kaplan-Meier multiplier is more than 0.10 below the long-run multiplier (oracle, last 100 decisions), i.e. early steps.
usage: python early_phase.py        (a few minutes)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np

P0, MINPOOL, NPERM = 60, 5, 150
NS, CUS, MS = [5, 10, 20, 40], [5, 20], [25, 50, 100]


def load(ds):
    Z = np.load(f"{WORK}/oof_cmapss_{ds}.npz"); ids = sorted({k.split("__")[0] for k in Z.files}); A = {}
    for e in ids:
        r = Z[f"{e}__rul"]; p = np.exp(Z[f"{e}__data"]); T = len(r)
        ps = np.array([np.median(p[max(0, i - 4):i + 1]) for i in range(T)]); ix = np.where((np.arange(T) >= 10) & (ps <= P0))[0]
        if len(ix): A[e] = (ps[ix[0]], r[ix[0]])
    return A


def q_exact(v, tau):
    n = len(v)
    if n < MINPOOL: return 1.0
    return np.sort(v)[min(max(int(np.ceil((n + 1) * tau)), 1), n) - 1]


def q_km(v, d, tau):
    n = len(v)
    if n < MINPOOL: return 1.0
    o = np.argsort(v, kind="stable"); vs, ds = v[o], d[o]
    G = 1.0 - np.cumprod(1.0 - ds / (n - np.arange(n)))
    t = min(tau * (n + 1) / n, 1.0); hit = np.where(G >= t)[0]
    return vs[hit[0]] if len(hit) else 1.0


def run(method, cu, A, order, N0, co=1.0):
    tau = co / (co + cu); vals = [A[e][1] / A[e][0] for e in order[:N0]]; dl = [1.0] * N0; cost, qs = [], []
    for e in order[N0:]:
        p, r = A[e]; rho = r / p; v, d_ = np.array(vals), np.array(dl)
        q = q_exact(v, tau) if method == "oracle" else q_km(v, d_, tau)
        d = p * q; cost.append((cu * max(d - r, 0) + co * max(r - d, 0)) / P0); qs.append(q)
        if method == "oracle" or d >= r: vals.append(rho); dl.append(1.0)
        else: vals.append(q); dl.append(0.0)
    return np.array(cost), np.array(qs)


if __name__ == "__main__":
    fleets = ["FD002", "FD004"]; acc = {}
    for ds in fleets:
        A = load(ds); E = sorted(A); rng0 = np.random.default_rng(7); orders = [list(rng0.permutation(E)) for _ in range(NPERM)]
        for N0 in NS:
            for cu in CUS:
                ex = {m: [] for m in MS}; lc = {m: [] for m in MS}; rel = {m: [] for m in MS}; early = []
                for o in orders:
                    co_, qo = run("oracle", cu, A, o, N0); ck, qk = run("km", cu, A, o, N0)
                    steady = co_[-100:].mean(); qlong = qo[-100:].mean()
                    for m in MS:
                        ex[m].append(ck[:m].sum() - co_[:m].sum()); lc[m].append(co_[:m].sum() - m * steady); rel[m].append(100 * (ck[:m].sum() / co_[:m].sum() - 1))
                    early.append(np.mean(qk[:50] < qlong - 0.10))
                acc.setdefault((N0, cu), []).append(({m: np.mean(ex[m]) for m in MS}, {m: np.mean(lc[m]) for m in MS}, float(np.mean(early)),
                                                      {m: np.std(ex[m]) / np.sqrt(NPERM) for m in MS}, {m: np.mean(rel[m]) for m in MS}))
                print(f"{ds} N0={N0:2d} cu/co={cu:2d}  KM-oracle first m: " + "  ".join(f"m={m}: {np.mean(ex[m]):+.3f}" for m in MS) +
                      "  | oracle learning cost: " + "  ".join(f"m={m}: {np.mean(lc[m]):+.2f}" for m in MS) + f"  | share early KM steps in first 50: {np.mean(early):.3f}", flush=True)
    print("\nMean over the two fleets (units of P0):")
    print("N0  cu/co | KM - oracle, first 25 / 50 / 100 decisions | in % of oracle | oracle learning cost, first 25 / 50 / 100 | share of first 50 KM steps below long-run q - 0.10 | max s.e. of KM-oracle")
    for (N0, cu), L in sorted(acc.items()):
        ex = [np.mean([x[0][m] for x in L]) for m in MS]; lc = [np.mean([x[1][m] for x in L]) for m in MS]; ea = np.mean([x[2] for x in L]); se = max(x[3][m] for x in L for m in MS); rl = [np.mean([x[4][m] for x in L]) for m in MS]
        print(f"{N0:3d} {cu:5d} | " + " / ".join(f"{v:+.3f}" for v in ex) + " | " + " / ".join(f"{v:+.1f}%" for v in rl) + " | " + " / ".join(f"{v:+.2f}" for v in lc) + f" | {ea:.3f} | {se:.3f}")
