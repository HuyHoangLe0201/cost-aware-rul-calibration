"""C-MAPSS fleet experiment, step 1: 5-fold (by engine) out-of-fold log-RUL predictions.
Models: data (MLP) vs hybrid (Paris-law closed form + kinematic consistency). Target: log of RUL clipped at 125.
usage: python cmapss_train.py FD001"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.cluster import KMeans
from multiprocessing import Pool
torch.set_num_threads(1)
import os
DS = sys.argv[1] if len(sys.argv) > 1 else "FD001"
FOLDSEED = int(os.environ.get("FOLDSEED", 0)); NOHYB = bool(os.environ.get("NOHYB"))
ROOT = CMAPSS_ROOT
CLIP, D, W, SEEDS, EPOCHS, NF = 125, 10, 10, 3, 250, 5

raw = np.loadtxt(f"{ROOT}train_{DS}.txt")
eng = raw[:, 0].astype(int); sets = raw[:, 2:5]; sens = raw[:, 5:]
keep = sens.std(0) > 1e-6; sens = sens[:, keep]
k = 1 if DS in ("FD001", "FD003") else 6
cond = KMeans(k, n_init=5, random_state=0).fit_predict(sets) if k > 1 else np.zeros(len(raw), int)
Z = np.empty_like(sens)
for c in range(k):
    m = cond == c; Z[m] = (sens[m] - sens[m].mean(0)) / (sens[m].std(0) + 1e-9)
ids = sorted(set(eng)); data = {}
for e in ids:
    z = Z[eng == e]; T = len(z)
    cs = np.cumsum(np.vstack([np.zeros((1, z.shape[1])), z]), 0)
    now = np.empty_like(z); prev = np.empty_like(z)
    for i in range(T):
        a = max(0, i + 1 - W); now[i] = (cs[i + 1] - cs[a]) / (i + 1 - a)
        b = max(0, i + 1 - 2 * W); prev[i] = (cs[a] - cs[b]) / (a - b) if a > b else now[i]
    X = np.hstack([now, now - prev, np.log1p(np.arange(T))[:, None]]).astype(np.float32)
    data[e] = (X, (T - np.arange(T)).astype(np.float64))     # true RUL: T-i (>=1)

class Net(nn.Module):
    def __init__(s, din, hybrid):
        super().__init__(); s.h = hybrid
        s.f = nn.Sequential(nn.Linear(din, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
        s.m_raw = nn.Parameter(torch.tensor(0.5)); s.logC = nn.Parameter(torch.tensor(-3.0)); s.b = nn.Parameter(torch.tensor(4.0))
    def forward(s, x):
        o = s.f(x)
        if not s.h: return s.b + 2 * o[:, 0], o[:, 0]
        m1 = Fn.softplus(s.m_raw) + 0.05
        sd = Fn.softplus(o[:, 0]) + 1e-3; y = (m1 * sd).clamp_min(1e-4)
        logR = y + torch.log1p(-torch.exp(-y)) - torch.log(m1) - s.logC
        return logR + 0.3 * torch.tanh(o[:, 1]) + 0.5 * torch.tanh(o[:, 2]), sd

def task(args):
    fold, hybrid, seed, tr_ids, te_ids = args
    torch.manual_seed(seed)
    Xs = np.vstack([data[e][0] for e in tr_ids]); mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    Xa = torch.tensor((Xs - mu) / sd, dtype=torch.float32)
    rt = np.concatenate([data[e][1] for e in tr_ids])
    ya = torch.tensor(np.log(np.minimum(rt, CLIP)), dtype=torch.float32)
    i0, off, msk = [], 0, []
    for e in tr_ids:
        T = len(data[e][1]); i0 += list(range(off, off + T - D)); msk += list(data[e][1][:T - D] <= CLIP); off += T
    i0 = torch.tensor(i0); msk = torch.tensor(msk, dtype=torch.float32)
    net = Net(Xa.shape[1], hybrid); opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    for ep in range(EPOCHS):
        opt.zero_grad(); l, lat = net(Xa); loss = ((l - ya) ** 2).mean()
        if hybrid:
            kin = ((l[i0] - torch.log(torch.exp(l[i0 + D]) + D)) ** 2 * msk).sum() / msk.sum().clamp_min(1)
            mono = (torch.relu(lat[i0 + D] - lat[i0]) * msk).sum() / msk.sum().clamp_min(1)
            loss = loss + 0.5 * kin + 0.1 * mono
        loss.backward(); opt.step()
    net.eval(); out = {}
    with torch.no_grad():
        for e in te_ids: out[e] = net(torch.tensor((data[e][0] - mu) / sd, dtype=torch.float32))[0].numpy()
    return fold, hybrid, seed, out, float(Fn.softplus(net.m_raw).item() + 1.05)

if __name__ == "__main__":
    t0 = time.time(); rng = np.random.default_rng(FOLDSEED); perm = rng.permutation(ids)
    folds = [list(perm[i::NF]) for i in range(NF)]
    jobs = [(f, hy, s, [e for e in ids if e not in folds[f]], folds[f]) for f in range(NF) for hy in ((False,) if NOHYB else (False, True)) for s in range(SEEDS)]
    acc = {(hy, e): [] for hy in (False, True) for e in ids}; ms = []
    with Pool(4) as p:
        for f, hy, s, out, m in p.imap_unordered(task, jobs):
            for e, v in out.items(): acc[(hy, e)].append(v)
            if hy: ms.append(m)
    res = {}
    for e in ids:
        res[f"e{e}__rul"] = data[e][1]; res[f"e{e}__data"] = np.mean(acc[(False, e)], 0); res[f"e{e}__hybrid"] = np.mean(acc[(True, e)], 0) if not NOHYB else res[f"e{e}__data"]
        res[f"e{e}__base"] = data[e][0][:, 0]       # first standardised-sensor window feature, used as drift covariate
    np.savez(f"{WORK}/oof_cmapss_{os.environ.get('OUTTAG', '')}{DS}.npz", **res)
    for m in ("data", "hybrid"):
        err = np.concatenate([np.exp(res[f"e{e}__{m}"]) - np.minimum(res[f"e{e}__rul"], CLIP) for e in ids])
        print(DS, m, "RMSE(clipped RUL, all cycles) = %.1f" % np.sqrt(np.mean(err ** 2)))
    print("learned m mean %.3f (init 2.05)" % np.mean(ms), "time", round(time.time() - t0), "s")
