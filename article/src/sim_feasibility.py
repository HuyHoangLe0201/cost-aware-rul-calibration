"""
Feasibility simulation (synthetic, numpy only) for:
 "Decision-aware prognostics: calibrated RUL intervals -> cost-optimal replacement"

Checks two things before investing in real data / PINN training:
 (A) Proposition 1: replacing at the tau*-quantile of the RUL distribution,
     tau* = c_o/(c_o+c_u), minimises expected cost under linear asymmetric loss.
 (B) Static split-conformal vs stage-conditional ROLLING conformal (delayed
     feedback: labels of a unit become available only after it fails) under
     concept drift of predictor bias across the unit stream.
Predictor is a stand-in: p = r * exp(b_t + u_i + e), r = true RUL.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np

rng = np.random.default_rng(0)

# ---------------- (A) Proposition 1 -----------------
def prop1(c_u=5.0, c_o=1.0, n=200_000):
    r = rng.lognormal(mean=np.log(100), sigma=0.35, size=n)  # RUL dist (any F works)
    d_grid = np.linspace(20, 250, 461)
    cost = np.array([(c_u * np.maximum(d - r, 0) + c_o * np.maximum(r - d, 0)).mean()
                     for d in d_grid])
    d_num = d_grid[cost.argmin()]
    tau = c_o / (c_o + c_u)
    d_theory = np.quantile(r, tau)
    print(f"[A] c_u={c_u}, c_o={c_o}: tau*={tau:.3f}  d_numeric={d_num:.1f}  "
          f"d_quantile={d_theory:.1f}  (should match)")

for cu in (2, 5, 10, 20):
    prop1(c_u=cu)

# ---------------- (B) conformal under drift -----------------
N_UNITS, N_CAL = 400, 100
STAGES = np.array([0.2, 0.4, 0.6, 0.8])      # fraction of life at which decision is made
SIG_U, SIG_E = 0.20, 0.10                    # unit-level / obs noise (multiplicative)

def make_unit(i):
    L = rng.weibull(3.0) * 1000 + 200         # life
    drift = 0.0 if i < N_CAL else 0.35 * (i - N_CAL) / (N_UNITS - N_CAL)  # bias drifts (over-estimation)
    u = rng.normal(0, SIG_U)
    out = []
    for s in STAGES:
        r = (1 - s) * L
        # predictor gets relatively noisier early in life
        noise = rng.normal(0, SIG_E * (1.5 - s))
        p = r * np.exp(drift + u + noise)
        out.append((s, p, r))
    return out

units = [make_unit(i) for i in range(N_UNITS)]
ratio = lambda p, r: r / p          # normalised target; quantile of this scales point RUL

c_u, c_o = 5.0, 1.0
tau = c_o / (c_o + c_u)            # target quantile for the replace-time d

def cost(d, r):
    return c_u * max(d - r, 0) + c_o * max(r - d, 0)

res = {k: [] for k in ("point", "static_conf", "rolling_conf", "oracle")}
cover = {k: {s: [] for s in STAGES} for k in ("static_conf", "rolling_conf")}
cal_static = {s: np.array([ratio(p, r) for (ss, p, r) in sum(units[:N_CAL], []) if ss == s]) for s in STAGES}
WINDOW = 60
for i in range(N_CAL, N_UNITS):
    for (s, p, r) in units[i]:
        # static: calibrated once on first N_CAL units
        q_static = np.quantile(cal_static[s], tau)
        # rolling: most recent WINDOW *completed* units only (delayed feedback)
        hist = [ratio(pp, rr) for u in units[max(0, i - WINDOW):i] for (ss, pp, rr) in u if ss == s]
        q_roll = np.quantile(hist, tau)
        for name, d in (("point", p), ("static_conf", p * q_static),
                        ("rolling_conf", p * q_roll), ("oracle", None)):
            if name == "oracle":
                res[name].append(cost(r * 1.0, r)); continue
            res[name].append(cost(d, r))
        # one-sided (lower) coverage of the tau-quantile bound: P(r >= d)
        cover["static_conf"][s].append(r >= p * q_static)
        cover["rolling_conf"][s].append(r >= p * q_roll)

print("\n[B] mean cost per decision on drifted test stream (lower is better)")
for k, v in res.items():
    print(f"   {k:13s} {np.mean(v):8.2f}")
print(f"\n    target one-sided coverage P(r >= d_tau) = {1 - tau:.3f}")
print("    stage   static   rolling")
for s in STAGES:
    print(f"    {s:.1f}    {np.mean(cover['static_conf'][s]):.3f}    {np.mean(cover['rolling_conf'][s]):.3f}")
