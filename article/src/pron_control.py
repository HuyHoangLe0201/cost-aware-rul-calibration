"""Null-control for physics-based features: same feature construction (4 orders x 2 harmonics x 2 channels, max over +-1 bin) at
(i) TRUE bearing orders [FTF, BSF, BPFO, BPFI] and (ii) NC random orders avoiding the true orders (and 2x) by >= 0.35 order.
Same LOBO MLP. Metrics per set: median over bearings of median|log(p/r)| at 20-80% of life, and over snapshots with RUL<=200 (alarm-like region)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import time, numpy as np, torch, torch.nn as nn
from multiprocessing import Pool
torch.set_num_threads(1)
W, SEEDS, EPOCHS, NC, DF = 10, 2, 300, 8, 10.0
E = np.load(WORK + "/envspec.npz"); names = sorted(E.files); FR = {"1": 30.0, "2": 27.5, "3": 25.0}
r = 3.5 / 25.6; TRUE = [0.5 * (1 - r), (25.6 / 7.0) * (1 - r**2), 6.5 * (1 - r), 6.5 * (1 + r)]       # orders (x fr): FTF, BSF, BPFO, BPFI
rng = np.random.default_rng(2024); sets = {"TRUE": TRUE}
avoid = np.array(TRUE + [2 * t for t in TRUE])
for c in range(NC):
    o = []
    while len(o) < 4:
        x = rng.uniform(0.3, 9.0)
        if np.all(np.abs(avoid - x) >= 0.35) and np.all(np.abs(avoid - 2 * x) >= 0.35) and all(abs(x - y) > 0.5 for y in o): o.append(x)
    sets[f"C{c}"] = o
def feats(n, orders):
    fr = FR[n[7]]; S = E[n]; T = S.shape[0]; cols = []
    for ch in (0, 1):
        for o in orders:
            for h in (1, 2):
                k = int(round(h * o * fr / DF)); k = min(max(k, 2), 58); cols.append(S[:, ch * 60 + k - 2: ch * 60 + k + 1].max(1))     # stored index j = bin j+1; window = bins k-1,k,k+1 => indices k-2..k
    return np.stack(cols, 1)
def win(x):
    z = x - np.median(x[:50], axis=0); T = len(z); c = np.cumsum(np.vstack([np.zeros((1, z.shape[1])), z]), 0); now = np.empty_like(z); prev = np.empty_like(z)
    for i in range(T):
        a = max(0, i + 1 - W); now[i] = (c[i + 1] - c[a]) / (i + 1 - a); b = max(0, i + 1 - 2 * W); prev[i] = (c[a] - c[b]) / (a - b) if a > b else now[i]
    return np.hstack([now, now - prev, np.log1p(np.arange(T))[:, None]]).astype(np.float32)
rul = {n: (E[n].shape[0] - np.arange(E[n].shape[0])).astype(float) for n in names}
D = {s: {n: win(feats(n, o)) for n in names} for s, o in sets.items()}
def fold(a):
    s, held, seed = a; tn = [n for n in names if n != held]; Xs = np.vstack([D[s][n] for n in tn]); mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    X = torch.tensor((Xs - mu) / sd, dtype=torch.float32); y = torch.tensor(np.concatenate([np.log(rul[n]) for n in tn]), dtype=torch.float32)
    torch.manual_seed(seed); m = nn.Sequential(nn.Linear(X.shape[1], 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)); b = nn.Parameter(torch.tensor(5.0))
    opt = torch.optim.Adam(list(m.parameters()) + [b], 3e-3, weight_decay=1e-4)
    for _ in range(EPOCHS): opt.zero_grad(); ((b + 2 * m(X).squeeze(-1) - y) ** 2).mean().backward(); opt.step()
    with torch.no_grad(): p = (b + 2 * m(torch.tensor((D[s][held] - mu) / sd, dtype=torch.float32)).squeeze(-1)).numpy()
    return s, held, p
if __name__ == "__main__":
    t0 = time.time(); jobs = [(s, n, sd) for s in sets for n in names for sd in range(SEEDS)]; acc = {}
    with Pool(4) as p:
        for s, held, pr in p.imap_unordered(fold, jobs): acc.setdefault((s, held), []).append(pr)
    print("done", round(time.time() - t0), "s"); print("orders TRUE", np.round(TRUE, 2)); out = {}
    for s in sets:
        m20, ml = [], []
        for n in names:
            p = np.mean(acc[(s, n)], 0); T = len(p); ix = [int(q * T) for q in (.2, .4, .6, .8)]
            m20.append(np.median(np.abs(p[ix] - np.log(rul[n][ix])))); msk = rul[n] <= 200; ml.append(np.median(np.abs(p[msk] - np.log(rul[n][msk]))))
        out[s] = (np.median(m20), np.mean(m20), np.median(ml), np.mean(ml))
        print(f"{s:5s} orders {np.round(sets[s],2)}: 20-80% med-of-med {out[s][0]:.2f} mean {out[s][1]:.2f} | RUL<=200 region med-of-med {out[s][2]:.2f} mean {out[s][3]:.2f}")
    ctrl = [out[s] for s in sets if s != "TRUE"]
    for j, nm in enumerate(("20-80% median", "20-80% mean", "RUL<=200 median", "RUL<=200 mean")):
        v = np.array([c[j] for c in ctrl]); print(f"{nm}: TRUE {out['TRUE'][j]:.2f} | controls mean {v.mean():.2f} sd {v.std():.2f} min {v.min():.2f} | TRUE better than {np.sum(out['TRUE'][j] < v)}/{len(v)} controls")
    np.save(WORK + "/pron_control.npy", out, allow_pickle=True)
