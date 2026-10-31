# rulcost: cost-aware calibration of RUL predictions for replacement decisions

You have a predictor of the **remaining useful life (RUL)** and an alarm that fires when the prediction falls to a threshold. *When should the unit actually be replaced?* Replacing at the point prediction is expensive when running past failure costs more than wasting life, and a fixed safety factor or a fixed "90 % lower bound" suits only one cost ratio.

`rulcost` answers that question from the units you have already completed:

* the best replacement date is `d = p * q`, where `p` is the point prediction at the alarm and `q` is the **tau\*-quantile of the ratio true RUL / predicted RUL**, with `tau* = c_over / (c_under + c_over)` fixed by the costs alone;
* `q` is estimated with a **conformal order statistic**: finite-sample, distribution-free, and the expected excess cost shrinks like `C / n` with a constant you can compute (`rulcost.theory`);
* units replaced before failing only reveal that their RUL exceeded the replacement date: such **right-censored labels** are handled with a Kaplan-Meier estimate instead of being dropped or recorded as failures, both of which bias `q` towards early replacement;
* the closed loop cannot lock itself into early replacement: from any initial pool, almost surely, all but a vanishing fraction of the Kaplan-Meier multipliers approach the optimal one (Theorem 2 of the article);
* it works with **any** RUL predictor: you only need the point prediction at the alarm and the outcome.

The method and its theory are described in the accompanying article (see *Citation*); `article/` reproduces its experiments.

## Install

```bash
pip install git+https://github.com/HuyHoangLe0201/cost-aware-rul-calibration      # or: git clone ... && pip install -e .
```

Python >= 3.9; the only dependency is NumPy.

## 60-second example

```python
import numpy as np
import rulcost as rc

# running past failure costs 10 times more per unit time than wasting life  ->  tau* = 1/11, target coverage 0.909
cal = rc.CostAwareCalibrator(c_under=10, c_over=1)

# units already completed: point prediction at the alarm, and the RUL that was actually observed after it
cal.fit(predicted_rul=[58, 61, 55, 63, 57, 60], observed=[41, 70, 52, 66, 38, 59])

cal.multiplier()                       # q
cal.replacement_lead_time(58.0)        # a new alarm with point prediction 58 -> replace after 58 * q
```

Online use: call `cal.update(predicted_rul, observed)` whenever a unit is completed. A unit that was replaced preventively after the lead time `d` without failing is a censored label:

```python
cal.update(predicted_rul=58.0, observed=d, censored=True)       # its RUL is only known to exceed d
```

`examples/quickstart.py` runs this end to end on simulated data (coverage on fresh units, cost against the point policy, closed loop with censoring, sample-size planning).

## Command line

Put one row per completed unit in a CSV file with columns `predicted_rul` and `true_rul` (optionally `censored`, `group`; column names can be changed with `--predicted`, `--observed`, `--censored`, `--group`).

```text
$ python -m rulcost calibrate examples/example_units.csv --c-under 10 --predict 55
completed units: 100 (0 censored); tau* = 0.0909; target coverage 1 - tau* = 0.9091
multiplier q = 0.6706  (labels used: 100)
point prediction 55 -> replace after 36.88

$ python -m rulcost compare examples/example_units.csv --c-under 20 --normalizer 60
policy                           cost / normalizer    coverage
point RUL                                    2.346       0.349
safety factor @0.5                           0.443       1.000
safety factor @0.7                           0.397       0.880
safety factor @0.85                          0.936       0.639
fixed 90%                                    0.415       0.884
Gauss quantile (log-normal)                  0.362       0.967
tau* static (first n_init)                   0.377       0.952
tau* cumulative (proposed)                   0.367       0.942
tau* rolling                                 0.370       0.948
(target coverage of the tau* policies: 0.952; c_under/c_over = 20)

$ python -m rulcost plan --c-under 10 --density 1.0 --tolerance 0.01
excess cost ~ 0.4545 / n; n >= 46 completed units for an expected excess cost <= 0.01 (in units of the point prediction)
```

`compare` replays your units in random order (the first `n_init` supply initial labels, every later unit is decided with the labels so far), so it shows how each rule would have performed on *your* data. Add `--censoring` to also compare the handling of censored labels in a closed loop. The example file holds the alarms of an MLP predictor on C-MAPSS FD001.

## API

| Function / class | Purpose |
|---|---|
| `CostAwareCalibrator(c_under, c_over, window=None, method="auto")` | `update`, `fit`, `multiplier`, `replacement_lead_time`; `window=w` for rolling calibration under drift; `group=` for stratified calibration |
| `critical_level`, `target_coverage`, `asymmetric_cost` | `tau*`, `1 - tau*` and the linear asymmetric cost |
| `conformal_quantile`, `gaussian_quantile`, `kaplan_meier_quantile` | the quantile rules on a sample of ratios |
| `first_alarm`, `alarm_table` | the alarm rule of the article (median of the last 5 predictions <= threshold, after at least 10 time steps) |
| `evaluate_policies`, `simulate_closed_loop` | replay your data under point, safety factors, fixed 90 %, Gauss, static, cumulative and rolling policies; compare oracle, naive, drop and Kaplan-Meier labelling |
| `excess_cost_constant`, `required_calibration_size`, `expected_coverage`, `density_at_quantile` | sample-size planning from the finite-sample theory |

Options that matter:

* `window=None` (cumulative, default) is best when the fleet is stable; a window `w` helps when the ratio drifts, with the best `w` scaling as `s^(-2/3)` for a drift `s` per unit (Proposition 5 of the article; an upper-bound rule, tune it on your data).
* `method="auto"` uses the conformal quantile while all labels are exact and Kaplan-Meier once a censored label is present. `"naive"` and `"drop"` exist only to reproduce the biased alternatives.
* With fewer than `min_pool=5` labels (or when the quantile is not identified by censored data) the multiplier is `fallback=1`: replace at the point prediction.
* `group=` keeps separate pools. It pays when a covariate known at the alarm shifts the ratio (Proposition 4); with many groups each pool is small.

## When does it apply?

The guarantees assume that

1. the ratio `true RUL / predicted RUL` of a new unit is exchangeable with those of the completed units, and independent of the point prediction (in practice: alarms at a fixed threshold, so predictions at the alarm are similar, and a stable fleet; use `window` if it drifts);
2. the cost is linear in the time of running past failure and in the time of life wasted, with no fixed cost per failure;
3. the labels come from units that were *alarmed in the same way* as the units you will decide for.

It does not make a bad predictor good: the cost that remains is dominated by the predictor's error. It only chooses the best date given the predictor and the costs, and tells you what that costs.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests check the quantile index, the Kaplan-Meier estimate, the direction of the bias of the naive treatments of censoring, the `C / n` law of the excess cost, and (when `article/work/` is present) that the library reproduces the policy costs and the closed-loop results of the article.

## Reproducing the article

`article/` holds the code, the out-of-fold predictions and the result files behind every table and figure of the article, with a staged `reproduce.sh`. See `article/README.md`.

## Citation

Please cite the article: Huy Hoang Le and Kim-Anh Nguyen, *Cost-aware calibration of remaining-useful-life predictions for replacement decisions* (manuscript prepared for the *Journal of Reliability Science and Engineering*; reference to be completed on publication). See also `CITATION.cff`.

## License

MIT (see `LICENSE`). The C-MAPSS and PRONOSTIA data sets keep their own terms of use.
