"""Leave-one-bearing-out (LOBO) out-of-fold RUL predictions on PRONOSTIA (17 run-to-failure bearings).
M_data   : MLP -> log RUL (no physics)
M_hybrid : physics-structured (Paris-law closed form, learnable exponent/coef) + residual MLP,
           + kinematic consistency loss  RUL(t) ~ RUL(t+D) + D  and monotone crack-proxy loss.
Output: oof_preds.npz  (per bearing: true RUL, pred logRUL for both models)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np, torch, torch.nn as nn, time
torch.set_num_threads(1)
D = 20            # snapshot gap for the kinematic-consistency term
W = 10            # causal smoothing window
F = np.load(WORK + "/features.npz")
names = sorted(F.files)

def build(name):
    x = F[name].copy()                      # (T,8): rms_h,kurt_h,max_h,crest_h, rms_v,kurt_v,max_v,crest_v
    T = len(x)
    base = np.median(x[:50], axis=0)
    z = np.empty_like(x)
    for j in (0, 2, 4, 6):                  # amplitude features -> log ratio to healthy baseline
        z[:, j] = np.log(np.maximum(x[:, j], 1e-6) / np.maximum(base[j], 1e-6))
    for j in (1, 3, 5, 7):
        z[:, j] = x[:, j] - base[j]
    c = np.cumsum(np.vstack([np.zeros((1, 8)), z]), axis=0)
    m_now = np.empty_like(z); m_prev = np.empty_like(z)
    for i in range(T):
        a = max(0, i + 1 - W); m_now[i] = (c[i + 1] - c[a]) / (i + 1 - a)
        b = max(0, i + 1 - 2 * W); m_prev[i] = (c[a] - c[b]) / max(a - b, 1) if a > b else m_now[i]
    t = np.log1p(np.arange(T))[:, None]
    X = np.hstack([m_now, m_now - m_prev, t])
    rul = (T - np.arange(T)).astype(np.float64)       # snapshots until failure
    return X.astype(np.float32), rul

data = {n: build(n) for n in names}

class Net(nn.Module):
    def __init__(self, din, hybrid):
        super().__init__()
        self.hybrid = hybrid
        self.f = nn.Sequential(nn.Linear(din, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 2))
        self.logp = nn.Parameter(torch.tensor(0.0))      # p=(m-2)/2 = softplus-ish
        self.logC = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(5.0))
    def forward(self, x):
        o = self.f(x)
        if not self.hybrid:
            return o[:, 0] * 2 + self.b, o[:, 1]
        p = torch.exp(self.logp)
        # Paris-law closed form: RUL = a^{-p}/(p C)  => logRUL = -p*log a - log p - log C ; log a := z (learned crack proxy)
        l_phys = -p * o[:, 0] - torch.log(p) - self.logC + self.b
        return l_phys + 0.5 * torch.tanh(o[:, 1]) * 2, o[:, 0]   # bounded residual, crack proxy

def train(train_names, hybrid, seed, epochs=250):
    torch.manual_seed(seed); np.random.seed(seed)
    Xs = np.vstack([data[n][0] for n in train_names]); mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    net = Net(Xs.shape[1], hybrid)
    opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    tr = []
    for n in train_names:
        X, r = data[n]; Xn = torch.tensor((X - mu) / sd)
        tr.append((Xn, torch.tensor(np.log(r), dtype=torch.float32), torch.tensor(r, dtype=torch.float32)))
    Xa = torch.cat([a for a, _, _ in tr]); ya = torch.cat([b for _, b, _ in tr])
    # pairs (t, t+D) inside each bearing
    i0 = []; off = 0
    for a, _, _ in tr:
        T = len(a); i0 += list(range(off, off + T - D)); off += T
    i0 = torch.tensor(i0)
    for ep in range(epochs):
        opt.zero_grad()
        l, zc = net(Xa)
        loss = ((l - ya) ** 2).mean()
        if hybrid:
            l1, l2 = l[i0], l[i0 + D]
            kin = (l1 - torch.log(torch.exp(l2) + D)) ** 2          # RUL(t) = RUL(t+D)+D
            mono = torch.relu(zc[i0] - zc[i0 + D])                    # crack proxy z non-decreasing (z ~ -a^{-p}.. sign via p>0)
            loss = loss + 0.5 * kin.mean() + 0.05 * mono.mean()
        loss.backward(); opt.step()
    net.eval()
    return net, mu, sd

def predict(net, mu, sd, X):
    with torch.no_grad():
        return net(torch.tensor((X - mu) / sd))[0].numpy()

def fold(args):
    held, seed = args
    tn = [n for n in names if n != held]
    out = {}
    for hy in (False, True):
        net, mu, sd = train(tn, hy, seed)
        out["hybrid" if hy else "data"] = predict(net, mu, sd, data[held][0])
    if True:
        out["p"] = float(np.exp(net.logp.item()))
    return held, out

if __name__ == "__main__":
    from multiprocessing import Pool
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t0 = time.time()
    with Pool(4) as p:
        res = dict(p.imap_unordered(fold, [(n, seed) for n in names]))
    out = {}
    for n in names:
        out[f"{n}__rul"] = data[n][1]; out[f"{n}__data"] = res[n]["data"]; out[f"{n}__hybrid"] = res[n]["hybrid"]
    np.savez(f"{WORK}/oof_preds_s{seed}.npz", **out)
    print("done", time.time() - t0, "s")
    for n in names:
        r = data[n][1]
        e1 = np.exp(res[n]["data"]) ; e2 = np.exp(res[n]["hybrid"])
        print(n, "MAE(rel) data %.2f hybrid %.2f | p_learned %.2f" % (np.mean(np.abs(e1 - r) / r), np.mean(np.abs(e2 - r) / r), res[n]["p"]))
