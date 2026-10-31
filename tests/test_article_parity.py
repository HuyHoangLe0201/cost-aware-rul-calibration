"""The library reproduces the numbers of the article (needs the shipped article/ directory)."""
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

import rulcost as rc

WORK = Path(__file__).resolve().parents[1] / "article" / "work"
pytestmark = pytest.mark.skipif(not (WORK / "oof_cmapss_FD001.npz").exists(), reason="article/work not available")


@pytest.fixture(scope="module")
def alarms():
    Z = np.load(WORK / "oof_cmapss_FD001.npz")
    ids = sorted({k.split("__")[0] for k in Z.files})
    p, r = [], []
    for e in ids:
        a = rc.first_alarm(np.exp(Z[f"{e}__data"]), 60.0)
        if a:
            p.append(a[1])
            r.append(float(Z[f"{e}__rul"][a[0]]))
    return np.array(p), np.array(r)


def test_alarms_equal_the_policy_suite(alarms):
    S = pickle.load(open(WORK / "suite" / "mlp_FD001.pkl", "rb"))
    assert np.allclose(alarms[0], S["p_a"]) and np.allclose(alarms[1], S["r_a"])


@pytest.mark.parametrize("c_under", [2, 5, 10, 20])
def test_policy_costs_equal_the_policy_suite(alarms, c_under):
    S = pickle.load(open(WORK / "suite" / "mlp_FD001.pkl", "rb"))
    rep = rc.evaluate_policies(*alarms, c_under, 1.0, normalizer=60.0)
    for lib, key in (("point", "point"), ("safety@0.5", "a50"), ("safety@0.7", "a70"), ("safety@0.85", "a85"), ("fixed90", "f90"),
                     ("gauss", "gauss"), ("static", "static"), ("cum", "cum"), ("roll", "roll")):
        assert np.nanmax(np.abs(rep.per_unit_cost[lib] - S["per"][key][c_under])) < 1e-5     # the suite was computed in float32


@pytest.mark.parametrize("c_under", [2, 5, 10, 20])
def test_closed_loop_equals_table3_source(alarms, c_under):
    line = [l for l in (WORK / "results_censor_FD001.txt").read_text().split("\n") if l.startswith(f"cu/co={c_under:>2}")][0]
    res = rc.simulate_closed_loop(*alarms, c_under, 1.0, normalizer=60.0)
    for name, key in (("oracle", "oracle"), ("naive", "naive"), ("drop", "drop"), ("km", "kaplan-meier")):
        cost, cov = map(float, re.search(rf"{name}: cost ([0-9.]+) cov ([0-9.]+)", line).groups())
        assert round(res[key]["cost"], 2) == cost and round(res[key]["coverage"], 2) == cov
