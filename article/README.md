# Reproducing the article

This directory contains the code, intermediate predictions and result files of

> Huy Hoang Le and Kim-Anh Nguyen, *Cost-aware calibration of remaining-useful-life predictions for replacement decisions.* Manuscript prepared for the *Journal of Reliability Science and Engineering* (IOP Publishing).

The article studies the decision that follows a remaining-useful-life (RUL) alarm: the best replacement date under a linear asymmetric cost is a quantile of the ratio of true to predicted RUL, at a level fixed by the cost ratio. It gives the exact excess cost of any date and a finite-sample bound for a conformal quantile, analyses calibration when the policy censors its own labels, and tests seven ways of injecting physics into the predictor on four C-MAPSS fleets and the PRONOSTIA bearings.

## What is here

| Path | Content |
|---|---|
| `src/` | All analysis scripts (Python). `src/paths.py` holds every file location and can be overridden with environment variables. |
| `work/` | Precomputed intermediate artifacts and result files: out-of-fold predictions (`oof_*.npz`), PRONOSTIA features, policy-suite results (`suite/`), fleet decision results (`r3/`), censoring, physics and theory logs. With these, everything after the predictors can be re-run without the raw data. |
| `figs/` | The figures of the article (PNG and PDF) as produced by `src/make_figures.py`. |
| `tikz/` | LaTeX/TikZ source of the framework figure (Fig. 1). |
| `reproduce.sh` | One command per stage (see below). |

The raw data are **not** redistributed (see *Data*). The manuscript itself is not part of this repository. To *use* the method on your own data, see the `rulcost` package in the parent directory.

## Quick start

```bash
cd article
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

./reproduce.sh theory       # numerical checks of the theory                       (~1 min)
./reproduce.sh suite        # every policy x predictor x fleet -> work/suite/       (~1.5 min, 4 cores)
./reproduce.sh decisions    # fleet decision evaluation          -> work/r3/        (~45 s per fleet and predictor)
./reproduce.sh censoring    # closed-loop censoring experiment   -> work/results_censor*.txt
./reproduce.sh figures      # all figures                        -> figs/           (~15 s)
```

These stages need only the shipped `work/` directory. To write into a different directory, set `RUL_WORK` and `RUL_FIGS`.

Re-training the predictors needs the raw data (see below) and takes hours on a CPU:

```bash
export CMAPSS_DIR=/path/to/dir/with/train_FD001.txt
export PRONOSTIA_DIR=/path/to/phm-ieee-2012-data-challenge-dataset   # contains Learning_set/ and Full_Test_Set/
./reproduce.sh train-cmapss
./reproduce.sh countdown
./reproduce.sh robustness
./reproduce.sh pronostia
```

## Data

* **C-MAPSS** (NASA turbofan run-to-failure simulations FD001-FD004). NASA Prognostics Center of Excellence data repository, <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/> (direct file: <https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip>). Unzip so that `train_FD001.txt`, ..., `RUL_FD004.txt` are in `CMAPSS_DIR`.
* **PRONOSTIA / FEMTO-ST** (IEEE PHM 2012 Prognostic Challenge, 17 bearings). The data were originally hosted by the FEMTO-ST institute; the copy used here is the mirror <https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset>, which also contains the challenge description with the bearing geometry. The scripts read `Learning_set/` and `Full_Test_Set/`.

Both data sets keep their own terms of use.

## Environment

Python 3.13, CPU only; package versions are in `requirements.txt`. Seeds are fixed in the scripts (fold partition `FOLDSEED`, model seeds). Everything was run on a 4-core machine.

## From the article to the files

| Article item | Scripts | Main files in `work/` or `figs/` |
|---|---|---|
| Fig. 1 (framework) | `tikz/` | `figs/fig_framework.pdf` |
| Figs. 2-9 | `src/make_figures.py` (reads the policy suite and the fleet results; Fig. 9 also uses numbers copied from the `semisup_*` logs) | `figs/fig_*.pdf`, `suite/*.pkl`, `r3/*.txt` |
| Tables 1, 2, 5 (policies, predictors, drift stream) | `src/fleet_eval.py`, `src/policy_suite.py` | `r3/*.txt`, `suite/*.pkl` |
| Predictors (MLP, Paris-law hybrid, ridge, Hist-GB, random forest, GRU, 1D-CNN) | `src/cmapss_train.py`, `src/cmapss_baselines.py`, `src/cmapss_cnn.py` | `oof_cmapss_*.npz`, `*.log` |
| Tables 3, 4 (censoring, Kaplan-Meier) | `src/censor_eval.py`, `src/censor_ci.py` | `results_censor_*.txt`, `results_censor_ci.txt` |
| Sec. 5.3 and Fig. 8 (theory checks) | `src/verify_theory.py`, `src/check_const.py` | printed output; `fig8_stats.json` |
| Table 6 (seven physics mechanisms) | `src/explore_physics.py`, `src/explore_timescale.py`, `src/physics_eval.py`, `src/semisup_physics.py`, `src/semisup_oof.py`, `src/ssm_physics.py`, `src/pron_physfeat.py`, `src/pron_control.py` | `semisup_*.log`, `results_fleet_ss*.txt`, `ssm_sens_FD001.log`, `pron_control.log`, `results_pron_*.txt` |
| Theorems 5 and 6 (Sec. 3, App. A) and their numerical checks (Sec. 5.4) | `src/km_theory_check.py`, `src/km_vs_naive_longrun.py`, `src/guard_check.py`; the library tests `tests/test_km_theory.py` | `km_theory_check.txt`, `km_vs_naive_longrun.txt` |
| Sec. 5.4 stress test (biased initial pool) | `src/adversarial_init.py` | `adversarial_init.txt` |
| Sec. 5.7 (independence of $p$ and the ratio; operating regimes) | `src/sensitivity_a0.py`, `src/regime_reliability.py` | `sensitivity_a0.txt`, `regime_reliability.txt` |
| Sec. 5.8 (fixed predictor on a disjoint fleet) | `src/prospective_eval.py` | `prospective_eval.txt` |
| Fig. 9c (paired on engines alarmed by both models); Monte Carlo error over stream orders | `src/paired_common_alarms.py`, `src/order_variability.py` | `paired_common_alarms.txt`, `order_variability.txt` |
| Table 7 and Fig. 9 (countdown constraint) | `src/semisup_physics.py`, `src/semisup_oof.py`, `src/fleet_eval.py` | `semisup_FD00*.log` (default, `_lam3` threefold weight, `_late` later truncation, `_age` cut-off age independent of the life), `results_fleet_ss*.txt` |
| PRONOSTIA pilot | `src/extract_features*.py`, `src/extract_faultfreq.py`, `src/extract_envspec.py`, `src/train_oof*.py`, `src/decision_eval*.py` | `features*.npz`, `envspec.npz`, `oof_pron_*.npz` |

Notes on files whose origin is not a single script: `fold_repeat.json` and `fold_repeat3.json` summarise the repeated fold partitions (`r3/s0_*`, `r3/fs1_*`, `r3/fs2_*`) and were aggregated by hand; some numbers in `src/make_figures.py` are copied from named logs, as stated in its header.

## Verification of this package

The package was assembled from the working directory of the study and then run from clean copies, comparing every output with the shipped file:

* `./reproduce.sh theory`, `censoring`, `suite`, `figures` and `decisions` (run end to end from the shipped `work/`) reproduce **all** shipped results byte for byte or exactly: the 7 censoring result files (including the variants with N0 = 5 and 10 and the bootstrap intervals), the 29 `suite/*.pkl` files, the 17 `r3/*.txt` files and the 8 matplotlib figures of the article (PNG, Figs. 2-9). `verify_theory.py` and `check_const.py` give the slope (-0.57) and the constants (0.216 at c_u/c_o = 2) quoted in the article.
* Re-training from the raw data was checked on **FD001** and on the PRONOSTIA features: the MLP and Paris-law hybrid (`cmapss_train.py`), also for the second fold partition (`FOLDSEED=1`), and the wear-law state-space model (`ssm_physics.py FD001 3 1.5`) agree with the shipped predictions to 1e-6 (float32 rounding); ridge, Hist-GB, random forest, GRU (`cmapss_baselines.py`), the 1D-CNN (`cmapss_cnn.py`) and the semi-supervised countdown model (`semisup_oof.py FD001`) agree exactly; the PRONOSTIA features (`extract_features2.py`, `extract_faultfreq.py`) agree exactly; the countdown-constraint experiment on FD001 (`semisup_physics.py`, default and the two variants `LAM=3 TAG=_lam3` and `TLO=0.5 THI=0.95 TAG=_late` cited in Sec. 6) reproduces the logged numbers (only the header line of the default log differs, because the print format was later extended).

* The robustness checks of Sections 5.4, 5.7 and 5.8 and the pairing of Fig. 9c (`sensitivity_a0.py`, `adversarial_init.py`, `order_variability.py`, `regime_reliability.py`, `paired_common_alarms.py`, and `prospective_eval.py` on FD001, which retrains the predictor) reproduce the stored outputs exactly.

Not re-run for this check: FD002-FD004 re-training, the other fold partitions, `prospective_eval.py` on FD002-FD004, and the PRONOSTIA predictors and null control (`train_oof*.py`, `pron_physfeat.py`, `pron_control.py`); their results are shipped as computed. Results may differ in the last digits with other library versions or CPU architectures.

## License

Code: MIT (see `LICENSE`). The data sets are subject to their own terms.

## Citation

Please cite the article (details to be added when published). A `CITATION.cff` is provided.
