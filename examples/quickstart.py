"""Quick start: choose replacement dates from completed units, online, with and without censored labels.

Run:  python examples/quickstart.py
"""
import numpy as np

import rulcost as rc

rng = np.random.default_rng(0)


def fleet(n):
    """Point predictions at the alarm (around 60 cycles) and the true RUL that followed: R = p * rho, rho log-normal."""
    p = rng.uniform(50, 70, n)
    return p, p * np.exp(rng.normal(0.0, 0.35, n))


c_under, c_over = 10.0, 1.0            # running past failure costs 10 times more per unit time than wasting life
cal = rc.CostAwareCalibrator(c_under, c_over)
print(cal, "\n")

# 1. labels of units already completed (e.g. tested to failure): predicted RUL at the alarm and what was observed
p0, r0 = fleet(40)
cal.fit(p0, r0)
print(f"multiplier after {cal.n_labels()} units: {cal.multiplier():.3f}")

# 2. a new unit raises an alarm with point prediction 58: plan its replacement
print(f"unit with point prediction 58 -> replace after {cal.replacement_lead_time(58.0):.1f}\n")

# 3. check on a large fresh sample: coverage P(R >= d) should be close to 1 - tau*, and the cost far below the point policy
p, r = fleet(50_000)
d = cal.replacement_lead_time(p)
cost = lambda dd: rc.asymmetric_cost(dd, r, c_under, c_over).mean() / 60
print(f"coverage {np.mean(r >= d):.3f} (target {cal.target_coverage:.3f});  cost / 60: calibrated {cost(d):.3f}, point policy {cost(p):.3f}\n")

# 4. units replaced before failure only reveal that R > d (right-censored labels): keep them as censored, do not drop them
closed = rc.CostAwareCalibrator(c_under, c_over)                # method="auto": conformal while labels are exact, Kaplan-Meier otherwise
closed.fit(*fleet(20))
for _ in range(200):
    pi, ri = (a[0] for a in fleet(1))
    d_i = pi * closed.multiplier()
    if d_i >= ri:
        closed.update(pi, ri)                                   # failed first: exact label
    else:
        closed.update(pi, d_i, censored=True)                   # replaced first: only R > d_i is known
print(f"closed loop after 220 units: multiplier {closed.multiplier():.3f} (open-loop benchmark {cal.multiplier():.3f})")

# 5. how many completed units does an expected excess cost of 0.02 (in units of the point prediction) need?
rho = r0 / p0
g = rc.density_at_quantile(rho, rc.critical_level(c_under, c_over))
print(f"estimated density at the quantile {g:.2f} -> n >= {rc.required_calibration_size(c_under, c_over, g, 0.02)} units")
