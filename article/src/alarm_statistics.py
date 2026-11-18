"""Alarm statistics of the seven predictors on the four C-MAPSS fleets (P0 = 60, alarm at the first cycle t >= 10 at which the median of the last five predictions is at most P0).

For every predictor and fleet: the number of engines, the share that raises an alarm, the true RUL at the alarm (median and quartiles; ideal value: P0), the share of life that has elapsed at the
alarm (median) and the median of |log(p / r)| at the alarm.  An engine whose prediction never falls to P0 raises no alarm and fails without a replacement date.
usage: python alarm_statistics.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np

P0 = 60
MODELS = [("MLP", "", "data"), ("Paris-law hybrid", "", "hybrid"), ("Ridge", "ridge_", "data"), ("Hist-GB", "hgb_", "data"), ("Random forest", "rf_", "data"), ("GRU", "gru_", "data"), ("1D-CNN", "cnn_", "data")]
FLEETS = ["FD001", "FD002", "FD003", "FD004"]


def stats(fleet, pfx, key):
    Z = np.load(f"{WORK}/oof_cmapss_{pfx}{fleet}.npz"); ids = sorted({k.split("__")[0] for k in Z.files}); rr, life, err = [], [], []
    for e in ids:
        r = Z[f"{e}__rul"]; p = np.exp(Z[f"{e}__{key}"]); T = len(r)
        ps = np.array([np.median(p[max(0, i - 4):i + 1]) for i in range(T)]); ix = np.where((np.arange(T) >= 10) & (ps <= P0))[0]
        if len(ix): i = ix[0]; rr.append(r[i]); life.append(i / T); err.append(abs(np.log(ps[i] / r[i])))
    return len(ids), len(rr), np.array(rr), np.array(life), np.array(err)


if __name__ == "__main__":
    print(f"{'predictor':18s} {'fleet':6s} engines alarmed  rate | true RUL at alarm: median [Q1, Q3] | life elapsed (median) | median |log(p/r)|")
    for name, pfx, key in MODELS:
        for fl in FLEETS:
            n, k, rr, lf, er = stats(fl, pfx, key)
            q1, q2, q3 = np.percentile(rr, [25, 50, 75])
            print(f"{name:18s} {fl:6s} {n:7d} {k:7d} {k / n:5.2f} | {q2:5.1f} [{q1:5.1f}, {q3:5.1f}] | {np.median(lf):.2f} | {np.median(er):.2f}")
