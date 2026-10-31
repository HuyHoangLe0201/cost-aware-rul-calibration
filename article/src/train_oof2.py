"""v2: LOBO out-of-fold RUL. Models:
 data   : MLP -> log RUL
 hybrid : Paris-law closed form  RUL = (exp((m-1)s)-1)/((m-1)C),  s = latent log-size distance to failure level (net),
          m>1, C learnable; bounded per-sample cond/residual terms; + kinematic consistency + monotone-s losses.
Seed-averaged (logRUL). Output oof2_preds.npz"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
torch.set_num_threads(1)
D, W, SEEDS, EPOCHS = 20, 10, 3, 300
F = np.load(WORK + "/features2.npz"); names = sorted(F.files)

def build(n):
    x = F[n].astype(np.float64); T = len(x)
    z = x - np.median(x[:50], axis=0)
    c = np.cumsum(np.vstack([np.zeros((1, z.shape[1])), z]), 0)
    now = np.empty_like(z); prev = np.empty_like(z)
    for i in range(T):
        a = max(0, i + 1 - W); now[i] = (c[i + 1] - c[a]) / (i + 1 - a)
        b = max(0, i + 1 - 2 * W); prev[i] = (c[a] - c[b]) / (a - b) if a > b else now[i]
    X = np.hstack([now, now - prev, np.log1p(np.arange(T))[:, None]])
    return X.astype(np.float32), (T - np.arange(T)).astype(np.float64)
data = {n: build(n) for n in names}

class Net(nn.Module):
    def __init__(s, din, hybrid):
        super().__init__(); s.h = hybrid
        s.f = nn.Sequential(nn.Linear(din, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
        s.m_raw = nn.Parameter(torch.tensor(0.5)); s.logC = nn.Parameter(torch.tensor(-4.0)); s.b = nn.Parameter(torch.tensor(5.0))
    def forward(s, x):
        o = s.f(x)
        if not s.h: return s.b + 2 * o[:, 0], o[:, 0]
        m1 = Fn.softplus(s.m_raw) + 0.05                       # m-1 > 0
        sdist = Fn.softplus(o[:, 0]) + 1e-3                    # distance-to-failure in log-size, >0
        y = (m1 * sdist).clamp_min(1e-4)
        logR = y + torch.log1p(-torch.exp(-y)) - torch.log(m1) - s.logC       # log[(e^y-1)/((m-1)C)]
        return logR + 0.3 * torch.tanh(o[:, 1]) + 0.5 * torch.tanh(o[:, 2]), sdist

def train(tn, hybrid, seed):
    torch.manual_seed(seed)
    Xs = np.vstack([data[n][0] for n in tn]); mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    net = Net(Xs.shape[1], hybrid); opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    Xa = torch.tensor(np.vstack([(data[n][0] - mu) / sd for n in tn]), dtype=torch.float32)
    ya = torch.tensor(np.concatenate([np.log(data[n][1]) for n in tn]), dtype=torch.float32)
    i0, off = [], 0
    for n in tn:
        T = len(data[n][1]); i0 += list(range(off, off + T - D)); off += T
    i0 = torch.tensor(i0)
    for ep in range(EPOCHS):
        opt.zero_grad(); l, lat = net(Xa); loss = ((l - ya) ** 2).mean()
        if hybrid:
            kin = (l[i0] - torch.log(torch.exp(l[i0 + D]) + D)) ** 2
            loss = loss + 0.5 * kin.mean() + 0.1 * torch.relu(lat[i0 + D] - lat[i0]).mean()
        loss.backward(); opt.step()
    net.eval(); return net, mu, sd

def pred(net, mu, sd, X):
    with torch.no_grad(): return net(torch.tensor((X - mu) / sd, dtype=torch.float32))[0].numpy()

def fold(held):
    tn = [n for n in names if n != held]; out = {}
    for hy in (False, True):
        P, ms = [], []
        for s in range(SEEDS):
            net, mu, sd = train(tn, hy, s); P.append(pred(net, mu, sd, data[held][0]))
            if hy: ms.append(float(Fn.softplus(net.m_raw).item() + 0.05 + 1))
        out["hybrid" if hy else "data"] = np.mean(P, 0)
        if hy: out["m"] = float(np.mean(ms))
    return held, out

if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(4) as p: res = dict(p.imap_unordered(fold, names))
    o = {}
    for n in names: o[f"{n}__rul"] = data[n][1]; o[f"{n}__data"] = res[n]["data"]; o[f"{n}__hybrid"] = res[n]["hybrid"]
    np.savez(WORK + "/oof2_preds.npz", **o); print("done", round(time.time() - t0), "s")
    for n in names:
        r = data[n][1]; T = len(r); ix = [int(s * T) for s in (.2, .4, .6, .8)]
        f = lambda k: np.median(np.abs(res[n][k][ix] - np.log(r[ix])))
        print(f"{n}: median|log(p/r)| @20-80%  data {f('data'):.2f}  hybrid {f('hybrid'):.2f}  m={res[n]['m']:.2f}")
