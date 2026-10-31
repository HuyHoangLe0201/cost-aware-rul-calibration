import numpy as np

import rulcost as rc


def fleet(seed, n=400, sigma=0.35):
    rng = np.random.default_rng(seed)
    p = rng.uniform(50, 70, n)
    return p, p * np.exp(rng.normal(0.0, sigma, n))


def test_closed_loop_bias_directions_and_km_recovery():
    p, r = fleet(0)
    res = rc.simulate_closed_loop(p, r, 10.0, 1.0, n_orders=20, normalizer=60.0)
    target = rc.target_coverage(10.0)
    # Proposition 3: dropping or recording as exact biases the multiplier down, i.e. coverage above the target (early replacement)
    assert res["drop"]["coverage"] > res["naive"]["coverage"] > res["oracle"]["coverage"]
    assert res["drop"]["cost"] > res["oracle"]["cost"]
    # Kaplan-Meier recovers the exact-label benchmark
    assert abs(res["kaplan-meier"]["coverage"] - res["oracle"]["coverage"]) < 0.02
    assert abs(res["kaplan-meier"]["cost"] - res["oracle"]["cost"]) < 0.01
    assert abs(res["oracle"]["coverage"] - target) < 0.03


def test_policy_report_structure_and_ranking():
    p, r = fleet(1)
    rep = rc.evaluate_policies(p, r, 20.0, 1.0, n_orders=20, normalizer=60.0)
    for k in ("point", "fixed90", "gauss", "static", "cum", "roll", "safety@0.5", "safety@0.7", "safety@0.85"):
        assert k in rep.cost and k in rep.coverage
    assert rep.cost["cum"] < 0.5 * rep.cost["point"]                    # the calibrated rule cuts the cost of the point policy
    assert abs(rep.coverage["cum"] - 0.952) < 0.03                      # tau* = 1/21
    assert "proposed" in rep.table()


def test_alarm_rule():
    traj = np.concatenate([np.linspace(120, 70, 30), np.linspace(70, 5, 30)])
    idx, smooth = rc.first_alarm(traj, 60.0)
    assert smooth <= 60.0 and idx >= 10
    assert rc.first_alarm(np.full(50, 100.0), 60.0) is None
    assert rc.first_alarm(np.full(8, 10.0), 60.0) is None               # no alarm before min_age
    units, pred, true = rc.alarm_table([traj, np.full(40, 100.0)], [np.arange(60, 0, -1.0), np.arange(40, 0, -1.0)], 60.0)
    assert units.tolist() == [0] and pred.shape == (1,) and true.shape == (1,)
