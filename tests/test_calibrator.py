import numpy as np
import pytest

import rulcost as rc


def make_fleet(rng, n, sigma=0.35):
    p = rng.uniform(50, 70, n)
    return p, p * np.exp(rng.normal(0.0, sigma, n))


def test_online_equals_batch():
    rng = np.random.default_rng(0)
    p, r = make_fleet(rng, 60)
    a = rc.CostAwareCalibrator(5).fit(p, r)
    b = rc.CostAwareCalibrator(5)
    for pi, ri in zip(p, r):
        b.update(pi, ri)
    assert a.multiplier() == b.multiplier()


def test_fallback_before_enough_labels():
    cal = rc.CostAwareCalibrator(5)
    assert cal.multiplier() == 1.0
    for _ in range(4):
        cal.update(60, 55)
    assert cal.multiplier() == 1.0
    assert cal.replacement_lead_time(60) == 60.0


def test_coverage_is_close_to_target_on_fresh_data():
    rng = np.random.default_rng(1)
    for c_under in (2, 10):
        cal = rc.CostAwareCalibrator(c_under).fit(*make_fleet(rng, 2000))
        p, r = make_fleet(rng, 100_000)
        cov = np.mean(r >= cal.replacement_lead_time(p))
        assert cov == pytest.approx(cal.target_coverage, abs=0.015)


def test_calibrated_policy_beats_point_and_fixed_rules_on_cost():
    rng = np.random.default_rng(2)
    cal = rc.CostAwareCalibrator(20).fit(*make_fleet(rng, 500))
    p, r = make_fleet(rng, 50_000)
    cost = lambda d: rc.asymmetric_cost(d, r, 20, 1).mean()
    assert cost(cal.replacement_lead_time(p)) < 0.5 * cost(p)
    assert cost(cal.replacement_lead_time(p)) < cost(0.7 * p)


def test_window_keeps_last_labels_and_tracks_drift():
    rng = np.random.default_rng(3)
    cal = rc.CostAwareCalibrator(5, window=30)
    for _ in range(100):
        cal.update(60, 60 * np.exp(rng.normal(0.5, 0.1)))        # early regime: units live longer than predicted
    for _ in range(30):
        cal.update(60, 60 * np.exp(rng.normal(-0.5, 0.1)))       # drift: units now fail earlier than predicted
    assert cal.n_labels() == 30
    assert cal.multiplier() < 0.8
    cum = rc.CostAwareCalibrator(5)
    for _ in range(100):
        cum.update(60, 60 * np.exp(rng.normal(0.5, 0.1)))
    for _ in range(30):
        cum.update(60, 60 * np.exp(rng.normal(-0.5, 0.1)))
    assert cum.multiplier() > cal.multiplier()                   # the cumulative pool lags behind the drift


def test_groups_are_calibrated_separately():
    rng = np.random.default_rng(4)
    cal = rc.CostAwareCalibrator(5)
    for _ in range(80):
        cal.update(60, 60 * np.exp(rng.normal(0.3, 0.1)), group="a")
        cal.update(60, 60 * np.exp(rng.normal(-0.3, 0.1)), group="b")
    assert cal.multiplier("a") > 1.0 > cal.multiplier("b")
    assert cal.multiplier("unseen") == 1.0


def test_censoring_modes():
    rng = np.random.default_rng(5)
    p, r = make_fleet(rng, 30)
    exact = rc.CostAwareCalibrator(5, method="conformal").fit(p, r)
    with pytest.raises(ValueError):
        exact.update(60, 40, censored=True)
    dropped = rc.CostAwareCalibrator(5, method="drop").fit(p, r)
    dropped.update(60, 40, censored=True)
    assert dropped.n_labels() == 30                              # the censored unit is discarded
    naive = rc.CostAwareCalibrator(5, method="naive").fit(p, r)
    naive.update(60, 40, censored=True)
    assert naive.n_labels() == 31
    auto = rc.CostAwareCalibrator(5).fit(p, r)
    before = auto.multiplier()
    assert before == exact.multiplier()                          # all labels exact -> conformal quantile
    auto.update(60, 40, censored=True)
    assert auto.n_labels() == 31 and np.isfinite(auto.multiplier())


def test_input_validation():
    cal = rc.CostAwareCalibrator(5)
    with pytest.raises(ValueError):
        cal.update(0, 10)
    with pytest.raises(ValueError):
        rc.CostAwareCalibrator(5, method="nope")
    with pytest.raises(ValueError):
        cal.fit([1, 2], [1])


def test_vector_and_scalar_lead_times():
    cal = rc.CostAwareCalibrator(5).fit([60] * 10, [50] * 10)
    assert cal.replacement_lead_time(60.0) == pytest.approx(50.0)
    assert np.allclose(cal.replacement_lead_time([60.0, 30.0]), [50.0, 25.0])
