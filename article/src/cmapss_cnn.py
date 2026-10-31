"""C-MAPSS stronger predictor: causal 1D-CNN on the last L=30 cycles of standardised sensors -> log RUL (clipped 125).
Same 5-fold-by-engine split / seeds protocol as cmapss_train.py (same permutation seed => same folds).
usage: python cmapss_cnn.py FD001  -> oof_cmapss_cnn_FD001.npz (key __data = CNN prediction)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, time, numpy as np, torch, torch.nn as nn
from sklearn.cluster import KMeans
from multiprocessing import Pool
torch.set_num_threads(1)
DS = sys.argv[1]; L, CLIP, NF, SEEDS, EPOCHS, BS = 30, 125, 5, 2, 25, 256
ROOT = CMAPSS_ROOT
raw = np.loadtxt(f"{ROOT}train_{DS}.txt"); eng = raw[:, 0].astype(int); sets = raw[:, 2:5]; sens = raw[:, 5:]
sens = sens[:, sens.std(0) > 1e-6]; k = 1 if DS in ("FD001", "FD003") else 6
cond = KMeans(k, n_init=5, random_state=0).fit_predict(sets) if k > 1 else np.zeros(len(raw), int)
Z = np.empty_like(sens)
for c in range(k):
    m = cond == c; Z[m] = (sens[m] - sens[m].mean(0)) / (sens[m].std(0) + 1e-9)
ids = sorted(set(eng)); data = {}
for e in ids:
    z = Z[eng == e]; T = len(z)
    pad = np.vstack([np.repeat(z[:1], L - 1, 0), z])                       # causal left-padding with first row
    W = np.stack([pad[i:i + L] for i in range(T)]).astype(np.float32)       # (T, L, S)
    tt = np.log1p(np.arange(T)).astype(np.float32)
    data[e] = (W, tt, (T - np.arange(T)).astype(np.float64))

class CNN(nn.Module):
    def __init__(s, S):
        super().__init__()
        s.c = nn.Sequential(nn.Conv1d(S, 32, 5, padding=2), nn.ReLU(), nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        s.h = nn.Sequential(nn.Linear(33, 48), nn.ReLU(), nn.Linear(48, 1)); s.b = nn.Parameter(torch.tensor(4.0))
    def forward(s, w, t):
        f = s.c(w.transpose(1, 2)).squeeze(-1); return s.b + s.h(torch.cat([f, t[:, None]], 1)).squeeze(-1)

def task(a):
    f, seed, tr, te = a; torch.manual_seed(seed)
    W = torch.tensor(np.concatenate([data[e][0] for e in tr])); tt = torch.tensor(np.concatenate([data[e][1] for e in tr]))
    y = torch.tensor(np.log(np.minimum(np.concatenate([data[e][2] for e in tr]), CLIP)), dtype=torch.float32)
    net = CNN(W.shape[2]); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4); n = len(y)
    for ep in range(EPOCHS):
        pm = torch.randperm(n)
        for i in range(0, n, BS):
            ix = pm[i:i + BS]; opt.zero_grad(); loss = ((net(W[ix], tt[ix]) - y[ix]) ** 2).mean(); loss.backward(); opt.step()
    net.eval(); out = {}
    with torch.no_grad():
        for e in te: out[e] = net(torch.tensor(data[e][0]), torch.tensor(data[e][1])).numpy()
    return out

if __name__ == "__main__":
    t0 = time.time(); perm = np.random.default_rng(0).permutation(ids); folds = [list(perm[i::NF]) for i in range(NF)]
    jobs = [(f, s, [e for e in ids if e not in folds[f]], folds[f]) for f in range(NF) for s in range(SEEDS)]
    acc = {e: [] for e in ids}
    with Pool(4) as p:
        for out in p.imap_unordered(task, jobs):
            for e, v in out.items(): acc[e].append(v)
    res = {}
    for e in ids: res[f"e{e}__rul"] = data[e][2]; res[f"e{e}__data"] = np.mean(acc[e], 0)
    np.savez(f"{WORK}/oof_cmapss_cnn_{DS}.npz", **res)
    err = np.concatenate([np.exp(res[f"e{e}__data"]) - np.minimum(res[f"e{e}__rul"], CLIP) for e in ids])
    print(DS, "CNN RMSE(clipped RUL, all cycles) = %.1f" % np.sqrt(np.mean(err ** 2)), "time", round(time.time() - t0), "s")
