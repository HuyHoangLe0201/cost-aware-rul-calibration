"""Physics-based features: envelope-spectrum amplitude at bearing characteristic frequencies (PRONOSTIA geometry, App. A.1):
 d=3.5mm, Z=13, Dm=25.6mm, contact angle 0.  fr = 30/27.5/25 Hz for conditions 1/2/3.
 BPFO=(Z/2)fr(1-d/Dm)  BPFI=(Z/2)fr(1+d/Dm)  BSF=(Dm/2d)fr(1-(d/Dm)^2)  FTF=(fr/2)(1-d/Dm)
 Feature = log( peak amplitude within +-1 bin (df=10 Hz) of f0 and of 2*f0 / median envelope-spectrum amplitude in 20-500 Hz ), per channel -> 4 faults x 2 harmonics x 2 ch = 16."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import glob, os, io, numpy as np
from multiprocessing import Pool
from scipy.signal import hilbert
ROOT = PRON_ROOT; FS = 25600.0; N = 2560; DF = FS / N
d, Z, Dm = 3.5, 13, 25.6; r = d / Dm
def fault_freqs(fr): return [0.5 * fr * (1 - r), (Dm / (2 * d)) * fr * (1 - r**2), (Z / 2) * fr * (1 - r), (Z / 2) * fr * (1 + r)]   # FTF, BSF, BPFO, BPFI
FR = {"1": 30.0, "2": 27.5, "3": 25.0}
def load(p): return np.loadtxt(io.StringIO(open(p).read().replace(";", ",")), delimiter=",")
def chan(x, f0s):
    x = x - x.mean(); X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / FS); X[f < 2000] = 0
    env = np.abs(hilbert(np.fft.irfft(X, n=len(x)))); env -= env.mean(); A = np.abs(np.fft.rfft(env)) / len(env); fa = np.fft.rfftfreq(len(env), 1 / FS)
    floor = np.median(A[(fa >= 20) & (fa <= 500)]) + 1e-12; out = []
    for f0 in f0s:
        for h in (1, 2):
            k = int(round(h * f0 / DF)); out.append(np.log(A[max(k - 1, 1):k + 2].max() / floor + 1e-9))
    return out
def bearing(args):
    n, dpath = args; f0s = fault_freqs(FR[n[7]]); res = []
    for p in sorted(glob.glob(os.path.join(dpath, "acc_*.csv"))):
        a = load(p); res.append(chan(a[:, 4], f0s) + chan(a[:, 5], f0s))
    return n, np.array(res)
if __name__ == "__main__":
    print("conditions -> [FTF, BSF, BPFO, BPFI] Hz:", {k: [round(v, 1) for v in fault_freqs(fr)] for k, fr in FR.items()})
    jobs = [(os.path.basename(p), p) for s in ("Learning_set", "Full_Test_Set") for p in sorted(glob.glob(f"{ROOT}/{s}/Bearing*"))]
    with Pool(4) as p: res = dict(p.imap_unordered(bearing, jobs))
    np.savez(WORK + "/features_fault.npz", **res); print({k: v.shape for k, v in sorted(res.items())})
