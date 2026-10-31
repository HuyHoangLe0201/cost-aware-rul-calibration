"""Extract per-snapshot features from PRONOSTIA (PHM 2012) run-to-failure bearings.
Uses Learning_set + Full_Test_Set => 17 bearings, each run to failure."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, glob, os, io
import numpy as np
from multiprocessing import Pool
from scipy.stats import kurtosis

ROOT = PRON_ROOT
OUT = WORK + "/features.npz"

def load(path):
    txt = open(path).read().replace(";", ",")
    return np.loadtxt(io.StringIO(txt), delimiter=",")

def feats(path):
    a = load(path)
    h, v = a[:, 4], a[:, 5]
    f = []
    for x in (h, v):
        rms = np.sqrt(np.mean(x**2))
        f += [rms, kurtosis(x), np.max(np.abs(x)), np.max(np.abs(x))/rms]
    return f

def bearing(args):
    name, d = args
    files = sorted(glob.glob(os.path.join(d, "acc_*.csv")))
    return name, np.array([feats(p) for p in files])

if __name__ == "__main__":
    jobs = []
    for sub in ("Learning_set", "Full_Test_Set"):
        for d in sorted(glob.glob(f"{ROOT}/{sub}/Bearing*")):
            jobs.append((os.path.basename(d), d))
    with Pool(4) as p:
        res = dict(p.imap_unordered(bearing, jobs))
    np.savez(OUT, **res)
    for k in sorted(res): print(k, res[k].shape)
