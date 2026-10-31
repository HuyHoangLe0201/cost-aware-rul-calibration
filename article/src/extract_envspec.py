"""Store log(envelope-spectrum amplitude / floor) for bins 1..60 (df=10 Hz, up to 600 Hz), both channels -> (T,120). Allows any frequency-set features."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import glob, os, io, numpy as np
from multiprocessing import Pool
from scipy.signal import hilbert
ROOT = PRON_ROOT; FS = 25600.0; NB = 60
def load(p): return np.loadtxt(io.StringIO(open(p).read().replace(";", ",")), delimiter=",")
def chan(x):
    x = x - x.mean(); X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / FS); X[f < 2000] = 0
    env = np.abs(hilbert(np.fft.irfft(X, n=len(x)))); env -= env.mean(); A = np.abs(np.fft.rfft(env)) / len(env); fa = np.fft.rfftfreq(len(env), 1 / FS)
    floor = np.median(A[(fa >= 20) & (fa <= 500)]) + 1e-12; return np.log(A[1:NB + 1] / floor + 1e-9)
def bearing(a):
    n, d = a; return n, np.array([np.concatenate([chan(load(p)[:, 4]), chan(load(p)[:, 5])]) for p in sorted(glob.glob(os.path.join(d, "acc_*.csv")))])
if __name__ == "__main__":
    jobs = [(os.path.basename(p), p) for s in ("Learning_set", "Full_Test_Set") for p in sorted(glob.glob(f"{ROOT}/{s}/Bearing*"))]
    with Pool(4) as p: res = dict(p.imap_unordered(bearing, jobs))
    np.savez(WORK + "/envspec.npz", **res); print("ok", next(iter(res.values())).shape)
