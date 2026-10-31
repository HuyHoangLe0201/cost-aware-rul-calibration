#!/usr/bin/env bash
# usage: ./reproduce.sh <stage>      (see README.md)
#   theory       numerical checks of the theory (Sec. 5.3, Fig. 8 inputs)           ~1 min, no raw data
#   suite        policy suite -> work/suite/*.pkl                                    minutes, no raw data
#   decisions    fleet decision evaluation -> work/r3/*.txt                          minutes, no raw data
#   censoring    closed-loop censoring experiment -> work/results_censor*.txt        minutes, no raw data
#   figures      all matplotlib figures -> figs/                                     <1 min, needs suite + r3
#   train-cmapss re-train the C-MAPSS predictors (out-of-fold)                        hours, needs C-MAPSS
#   countdown    countdown-constraint experiment (Table 7, Fig. 9) and its two variants  ~5 min per fleet, needs C-MAPSS
#   pronostia    PRONOSTIA features, predictors and the null control                  hours, needs PRONOSTIA
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
WORK=${RUL_WORK:-$PWD/work}
export PYTHONHASHSEED=0
FLEETS="FD001 FD002 FD003 FD004"
mkdir -p "$WORK/r3"

r3() {  # r3 TAG PFX BOTH DS  -> work/r3/TAG_DS.txt  (as in the original runs)
  if [ -n "$3" ]; then BOTH=1 PFX="$2" $PY src/fleet_eval.py "$4" > "$WORK/r3/$1_$4.txt"
  else PFX="$2" $PY src/fleet_eval.py "$4" > "$WORK/r3/$1_$4.txt"; fi
}

case "${1:-}" in
  theory)    $PY src/verify_theory.py; $PY src/check_const.py ;;
  suite)     $PY src/policy_suite.py ;;
  decisions) for ds in $FLEETS; do r3 s0 "" 1 $ds; r3 fs1 fs1_ "" $ds; r3 fs2 fs2_ "" $ds; r3 cnn cnn_ "" $ds; done
             r3 ssm ssmL3_b1.5_ 1 FD001 ;;
  censoring) : > "$WORK/results_censor_ci.txt"
             for ds in $FLEETS; do $PY src/censor_eval.py $ds > "$WORK/results_censor_$ds.txt"; $PY src/censor_ci.py $ds >> "$WORK/results_censor_ci.txt"; done
             N0=5 $PY src/censor_eval.py FD002 > "$WORK/results_censor_FD002_N0_5.txt"
             N0=10 $PY src/censor_eval.py FD002 > "$WORK/results_censor_FD002_N0_10.txt" ;;
  figures)   $PY src/make_figures.py ;;
  train-cmapss)   # needs CMAPSS_DIR
             for ds in $FLEETS; do
               $PY src/cmapss_train.py $ds                                  # MLP ("data") and Paris-law hybrid -> oof_cmapss_<DS>.npz
               FOLDSEED=1 OUTTAG=fs1_ NOHYB=1 $PY src/cmapss_train.py $ds   # other fold partitions
               FOLDSEED=2 OUTTAG=fs2_ NOHYB=1 $PY src/cmapss_train.py $ds
               $PY src/cmapss_baselines.py $ds                              # ridge, Hist-GB, random forest, GRU
               $PY src/cmapss_cnn.py $ds                                    # 1D-CNN
             done ;;
  countdown)      # needs CMAPSS_DIR; the variants are the ones cited in Sec. 6 (threefold weight, later truncation)
             for ds in FD001 FD003; do $PY src/semisup_physics.py $ds > "$WORK/semisup_$ds.log"; done
             LAM=3 TAG=_lam3 $PY src/semisup_physics.py FD001 > "$WORK/semisup_FD001_lam3.log"
             TLO=0.5 THI=0.95 TAG=_late $PY src/semisup_physics.py FD001 > "$WORK/semisup_FD001_late.log"
             for ds in FD001 FD003; do $PY src/semisup_oof.py $ds; done ;;
  pronostia)      # needs PRONOSTIA_DIR
             $PY src/extract_features2.py; $PY src/extract_faultfreq.py; $PY src/extract_envspec.py
             $PY src/pron_physfeat.py; $PY src/pron_control.py ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
