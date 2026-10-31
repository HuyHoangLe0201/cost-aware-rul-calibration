"""v2 features: time-domain + 6 log band energies + high-band envelope stats, per channel (24 features)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import glob, os, io, numpy as np
from multiprocessing import Pool
from scipy.stats import kurtosis
from scipy.signal import hilbert
ROOT = PRON_ROOT
FS = 25600.0
EDGES = [0, 500, 1500, 3000, 5000, 8000, 12800]

def load(p):
    return np.loadtxt(io.StringIO(open(p).read().replace(";", ",")), delimiter=",")

def chan(x):
    x = x - x.mean()
    rms = np.sqrt(np.mean(x**2)); mx = np.max(np.abs(x))
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / FS); P = np.abs(X) ** 2 / len(x)
    be = [np.log(P[(f >= EDGES[i]) & (f < EDGES[i + 1])].sum() + 1e-9) for i in range(6)]
    Xh = X.copy(); Xh[f < 2000] = 0
    env = np.abs(hilbert(np.fft.irfft(Xh, n=len(x)))); env = env - env.mean()
    return [np.log(rms + 1e-9), kurtosis(x), np.log(mx + 1e-9), mx / rms] + be + [np.log(np.sqrt(np.mean(env**2)) + 1e-9), kurtosis(env)]

def feats(p):
    a = load(p); return chan(a[:, 4]) + chan(a[:, 5])

def bearing(args):
    n, d = args
    return n, np.array([feats(p) for p in sorted(glob.glob(os.path.join(d, "acc_*.csv")))])

if __name__ == "__main__":
    jobs = [(os.path.basename(d), d) for s in ("Learning_set", "Full_Test_Set") for d in sorted(glob.glob(f"{ROOT}/{s}/Bearing*"))]
    with Pool(4) as p: res = dict(p.imap_unordered(bearing, jobs))
    np.savez(WORK + "/features2.npz", **res)
    print({k: v.shape for k, v in sorted(res.items())})
