"""PRONOSTIA LOBO: does physics-based (fault-frequency envelope) feature set improve RUL / decisions?
Inputs: base (24 time/band/envelope features) | phys (16 fault-frequency envelope features) | both (40). Same MLP, 3 seeds, 300 epochs.
Output: oof_pron_A.npz (data=base, hybrid=both), oof_pron_B.npz (data=base, hybrid=phys-only)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import time, numpy as np, torch, torch.nn as nn
from multiprocessing import Pool
torch.set_num_threads(1)
W, SEEDS, EPOCHS = 10, 3, 300
F1 = np.load(WORK + "/features2.npz"); F2 = np.load(WORK + "/features_fault.npz"); names = sorted(F1.files)
def win(x):
    z = x - np.median(x[:50], axis=0); T = len(z); c = np.cumsum(np.vstack([np.zeros((1, z.shape[1])), z]), 0)
    now = np.empty_like(z); prev = np.empty_like(z)
    for i in range(T):
        a = max(0, i + 1 - W); now[i] = (c[i + 1] - c[a]) / (i + 1 - a); b = max(0, i + 1 - 2 * W); prev[i] = (c[a] - c[b]) / (a - b) if a > b else now[i]
    return np.hstack([now, now - prev, np.log1p(np.arange(T))[:, None]]).astype(np.float32)
data = {n: {"base": win(F1[n].astype(float)), "phys": win(F2[n].astype(float)), "rul": (len(F1[n]) - np.arange(len(F1[n]))).astype(float)} for n in names}
for n in names: data[n]["both"] = np.hstack([data[n]["base"][:, :-1], data[n]["phys"]])
def net(din): return nn.Sequential(nn.Linear(din, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
def fold(a):
    held, fs, seed = a; tn = [n for n in names if n != held]
    Xs = np.vstack([data[n][fs] for n in tn]); mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    X = torch.tensor((Xs - mu) / sd, dtype=torch.float32); y = torch.tensor(np.concatenate([np.log(data[n]["rul"]) for n in tn]), dtype=torch.float32)
    torch.manual_seed(seed); m = net(X.shape[1]); b = nn.Parameter(torch.tensor(5.0)); opt = torch.optim.Adam(list(m.parameters()) + [b], 3e-3, weight_decay=1e-4)
    for _ in range(EPOCHS):
        opt.zero_grad(); loss = ((b + 2 * m(X).squeeze(-1) - y) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad(): p = (b + 2 * m(torch.tensor((data[held][fs] - mu) / sd, dtype=torch.float32)).squeeze(-1)).numpy()
    return held, fs, p
if __name__ == "__main__":
    t0 = time.time(); jobs = [(n, fs, s) for n in names for fs in ("base", "phys", "both") for s in range(SEEDS)]; acc = {}
    with Pool(4) as p:
        for held, fs, pr in p.imap_unordered(fold, jobs): acc.setdefault((held, fs), []).append(pr)
    for tag, alt in (("A", "both"), ("B", "phys")):
        o = {}
        for n in names: o[f"{n}__rul"] = data[n]["rul"]; o[f"{n}__data"] = np.mean(acc[(n, "base")], 0); o[f"{n}__hybrid"] = np.mean(acc[(n, alt)], 0)
        np.savez(f"{WORK}/oof_pron_{tag}.npz", **o)
    print("done", round(time.time() - t0), "s")
    for fs in ("base", "phys", "both"):
        e = [np.median(np.abs(np.mean(acc[(n, fs)], 0)[[int(s * len(data[n]['rul'])) for s in (.2, .4, .6, .8)]] - np.log(data[n]["rul"][[int(s * len(data[n]['rul'])) for s in (.2, .4, .6, .8)]]))) for n in names]
        print(f"{fs:5s}: median over bearings of median|log(p/r)| @20-80% = {np.median(e):.2f}   mean {np.mean(e):.2f}")
