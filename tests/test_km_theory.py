"""Numerical checks of Theorem 5 of the article: Kaplan-Meier under policy-induced censoring (log-normal ratio)."""
import math

import numpy as np
import pytest

import rulcost as rc

MU, SIG, TAU = 0.0, 0.35, 1 / 11
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
G = lambda y: Phi((math.log(y) - MU) / SIG) if y > 0 else 0.0
Lam = lambda y: -math.log(1 - G(y))


def Ginv(p):
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if G(mid) < p else (lo, mid)
    return lo


QSTAR = Ginv(TAU)


def run_policy(n, rng, kind, n0=20, explore=None):
    """Closed loop; q_s depends on the past only.  kind: 'km' (the policy of Theorem 5), 'runmean' (a different predictable rule)."""
    rho = rng.lognormal(MU, SIG, n)
    q = np.full(n, np.inf)
    Y = np.empty(n)
    d = np.zeros(n, bool)
    expl = np.zeros(n, bool)
    for s in range(n):
        if s >= n0:
            if explore is not None and rng.random() < explore(s + 1):
                q[s] = 1.0
                expl[s] = True
            elif kind == "km":
                q[s] = rc.kaplan_meier_quantile(Y[:s], d[:s], TAU)
            else:
                q[s] = 1.3 * Y[:s][d[:s]].mean()
        Y[s] = min(rho[s], q[s])
        d[s] = rho[s] <= q[s]
    return rho, q, Y, d, expl


def na_error_and_bound(Y, d, x):
    """sup_{y<=x}|Lambda_hat - Lambda| and 2 sup_{u<=x}|M_n(u)| / R_n(x), computed exactly."""
    n = len(Y)
    pts = np.unique(np.concatenate([[0.0], Y[Y <= x], [x]]))
    A = {0.0: 0.0}
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        acc += np.sum(Y >= b) * (Lam(b) - Lam(a))      # R_n is constant (= #{Y >= b}) on (a, b]
        A[b] = acc
    S = 0.0
    for u in pts:
        S = max(S, abs(np.sum(d & (Y <= u)) - A[u]))
        if u > 0:
            S = max(S, abs(np.sum(d & (Y < u)) - A[u]))
    err, cum = 0.0, 0.0
    for t in np.sort(Y[d & (Y <= x)]):
        err = max(err, abs(cum - Lam(t)))
        cum += 1.0 / np.sum(Y >= t)
        err = max(err, abs(cum - Lam(t)))
    err = max(err, abs(cum - Lam(x)))
    return err, 2 * S / np.sum(Y >= x)


@pytest.mark.parametrize("kind", ["km", "runmean"])
def test_pathwise_bound_holds_for_predictable_censoring(kind):
    rng = np.random.default_rng(1)
    for _ in range(6):
        rho, q, Y, d, _ = run_policy(int(rng.integers(120, 260)), rng, kind)
        err, bound = na_error_and_bound(Y, d, Ginv(0.6))
        assert err <= bound + 1e-9


def test_no_lock_in_below_x_with_high_probability():
    """Theorem 5(a): P(ever q_{n+1} < x) <= sum_{m>=N0} exp(-m D(tau || G(x)))."""
    rng = np.random.default_rng(2)
    n0, x = 50, QSTAR - 0.20
    Gx = G(x)
    D = TAU * math.log(TAU / Gx) + (1 - TAU) * math.log((1 - TAU) / (1 - Gx))
    bound = math.exp(-n0 * D) / (1 - math.exp(-D))
    reps = 120
    hits = sum(bool(np.any(run_policy(110, rng, "km", n0)[1][n0:] < x)) for _ in range(reps))
    assert hits / reps <= bound + 3 * math.sqrt(bound * (1 - bound) / reps) + 0.02


def run_guarded(n, rng, n0, guard, init):
    """Closed loop with the risk-set guard of Theorem 6 from an adversarial initial pool."""
    rho = rng.lognormal(MU, SIG, n)
    rho[: len(init)] = init
    q = np.full(n, np.inf)
    Y = rho.copy()
    d = np.ones(n, bool)
    for s in range(n0, n):
        q[s] = rc.kaplan_meier_quantile(Y[:s], d[:s], TAU, 5, 1.0, rc.risk_floor(s, guard))
        Y[s] = min(rho[s], q[s])
        d[s] = rho[s] <= q[s]
    return q


def test_risk_set_guard_recovers_from_an_adversarial_pool():
    """Theorem 6: from five identical labels far below q* the guarded multiplier leaves the neighbourhood of 0.5 and settles near q*."""
    rng = np.random.default_rng(4)
    late = []
    for _ in range(12):
        q = run_guarded(700, rng, 5, "loglog", np.full(5, 0.50))
        assert not np.any(q[450:] < QSTAR - 0.10)
        late.append(np.mean(q[500:]))
    assert abs(np.mean(late) - QSTAR) < 0.04


def test_estimate_is_accurate_where_many_units_remain_at_risk():
    """Lemma A3: the Nelson-Aalen error at levels with at least 10 log n units at risk is small on closed-loop data."""
    rng = np.random.default_rng(6)
    for _ in range(4):
        rho, q, Y, d, _ = run_policy(400, rng, "km")
        pts = np.sort(Y[d & (Y <= Ginv(0.6))])
        cum = err = 0.0
        for t in pts:
            r = np.sum(Y >= t)
            if r < 10 * math.log(400):
                break
            err = max(err, abs(cum - Lam(t)))
            cum += 1.0 / r
            err = max(err, abs(cum - Lam(t)))
        assert err < 0.2
