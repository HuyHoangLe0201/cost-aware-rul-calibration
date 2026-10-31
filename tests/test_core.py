import numpy as np
import pytest

import rulcost as rc


def test_critical_level_and_target():
    assert rc.critical_level(9, 1) == pytest.approx(0.1)
    assert rc.critical_level(1, 1) == 0.5
    assert rc.target_coverage(9, 1) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        rc.critical_level(0, 1)


def test_cost_is_linear_asymmetric():
    # d beyond failure by 4: c_under * 4 ; d before failure by 3: c_over * 3
    assert rc.asymmetric_cost(14, 10, 5, 1) == pytest.approx(20)
    assert rc.asymmetric_cost(7, 10, 5, 1) == pytest.approx(3)
    assert rc.asymmetric_cost(10, 10, 5, 1) == 0


def test_conformal_order_statistic_index():
    v = np.arange(1.0, 10.0)                       # n = 9
    assert rc.conformal_quantile(v, 0.5) == 5.0    # k = ceil(10 * 0.5) = 5
    assert rc.conformal_quantile(v, 0.05) == 1.0   # k = ceil(0.5) = 1
    assert rc.conformal_quantile(v, 0.99) == 9.0   # k clipped to n
    assert rc.conformal_quantile(v[:4], 0.5) == 1.0   # fewer than min_pool labels -> fallback
    assert rc.conformal_quantile(v[:4], 0.5, fallback=0.8) == 0.8


def test_conformal_is_invariant_to_order():
    rng = np.random.default_rng(1)
    v = rng.lognormal(size=50)
    assert rc.conformal_quantile(v, 0.2) == rc.conformal_quantile(rng.permutation(v), 0.2)


def test_gaussian_quantile_formula():
    rng = np.random.default_rng(2)
    v = rng.lognormal(0.1, 0.4, 5000)
    lv = np.log(v)
    z = -1.2815515655446004                          # normal quantile at 0.10
    assert rc.gaussian_quantile(v, 0.10) == pytest.approx(np.exp(lv.mean() + z * lv.std()), rel=1e-9)


@pytest.mark.parametrize("tau", [0.05, 0.1, 0.3, 0.5, 0.8])
def test_kaplan_meier_equals_conformal_without_censoring(tau):
    rng = np.random.default_rng(3)
    v = rng.lognormal(size=200)
    assert rc.kaplan_meier_quantile(v, np.ones(200), tau) == rc.conformal_quantile(v, tau)


def test_kaplan_meier_recovers_quantile_under_random_censoring():
    rng = np.random.default_rng(4)
    n = 20000
    rho = rng.lognormal(0.0, 0.4, n)
    cens = rng.lognormal(0.3, 0.4, n)                # independent censoring times
    obs = np.minimum(rho, cens)
    exact = rho <= cens
    tau = 0.2
    truth = np.quantile(rho, tau)
    assert rc.kaplan_meier_quantile(obs, exact, tau) == pytest.approx(truth, rel=0.03)
    # treating the censored values as exact, or dropping them, is biased towards early replacement (a smaller multiplier)
    assert rc.conformal_quantile(obs, tau) < truth
    assert rc.conformal_quantile(obs[exact], tau) < truth


def test_kaplan_meier_unidentified_quantile_falls_back():
    v = np.linspace(0.5, 1.5, 20)
    assert rc.kaplan_meier_quantile(v, np.zeros(20), 0.3) == 1.0


def test_risk_floor_values_and_guard():
    import rulcost as rc
    assert rc.risk_floor(100, None) == 0 and rc.risk_floor(100, 7) == 7
    assert rc.risk_floor(1000, "loglog") == 14 and rc.risk_floor(1000, "log2") == 48
    v = np.arange(1.0, 9.0)
    e = np.ones(8)
    assert rc.kaplan_meier_quantile(v, e, 0.9) == 8.0
    assert rc.kaplan_meier_quantile(v, e, 0.9, min_risk=3) == 1.0          # the quantile rests on one unit at risk: not trusted
    assert rc.kaplan_meier_quantile(v, e, 0.5, min_risk=3) == rc.kaplan_meier_quantile(v, e, 0.5)
