import math

import numpy as np
import pytest

import rulcost as rc

scipy_special = pytest.importorskip("scipy.special")
ndtr = scipy_special.ndtr


def lognormal_cost(q, mu, sigma, c_under, c_over):
    """Expected cost Phi(q) = c_under E(q - rho)^+ + c_over E(rho - q)^+ for rho ~ LogNormal(mu, sigma)."""
    z1 = (np.log(q) - mu) / sigma
    z2 = z1 - sigma
    mean = math.exp(mu + sigma ** 2 / 2)
    under = q * ndtr(z1) - mean * ndtr(z2)
    over = mean * (1 - ndtr(z2)) - q * (1 - ndtr(z1))
    return c_under * under + c_over * over


def test_excess_cost_scales_as_constant_over_n():
    mu, sigma, c_under, c_over, n = 0.0, 0.35, 2.0, 1.0, 320
    tau = rc.critical_level(c_under, c_over)
    q_star = math.exp(mu - sigma * 0.4307272992954575)           # normal quantile at tau = 1/3 is -0.4307
    g = math.exp(-0.5 * ((math.log(q_star) - mu) / sigma) ** 2) / (q_star * sigma * math.sqrt(2 * math.pi))
    rng = np.random.default_rng(0)
    qhat = np.array([rc.conformal_quantile(rng.lognormal(mu, sigma, n), tau) for _ in range(4000)])
    excess = lognormal_cost(qhat, mu, sigma, c_under, c_over) - lognormal_cost(q_star, mu, sigma, c_under, c_over)
    predicted = rc.excess_cost_constant(c_under, c_over, g)
    assert n * excess.mean() == pytest.approx(predicted, rel=0.12)


def test_expected_coverage_offset():
    for n in (10, 40, 200):
        tau = 0.1
        assert 1 - tau - 1 / (n + 1) < rc.expected_coverage(n, tau) <= 1 - tau + 1e-12


def test_sample_size_rule():
    c = rc.excess_cost_constant(10, 1, 1.0)
    assert c == pytest.approx(11 * (1 / 11) * (10 / 11) / 2)
    assert rc.required_calibration_size(10, 1, 1.0, 0.01) == math.ceil(c / 0.01)
    assert rc.required_calibration_size(10, 1, 1.0, 0.005) > rc.required_calibration_size(10, 1, 1.0, 0.01)
    with pytest.raises(ValueError):
        rc.excess_cost_constant(10, 1, 0.0)


def test_density_estimate_is_roughly_right():
    mu, sigma, tau = 0.0, 0.35, 0.2
    rng = np.random.default_rng(1)
    rho = rng.lognormal(mu, sigma, 5000)
    q = math.exp(mu + sigma * -0.8416212335729143)
    true_g = math.exp(-0.5 * ((math.log(q) - mu) / sigma) ** 2) / (q * sigma * math.sqrt(2 * math.pi))
    assert rc.density_at_quantile(rho, tau) == pytest.approx(true_g, rel=0.15)
