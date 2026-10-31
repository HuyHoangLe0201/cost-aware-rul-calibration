"""Physics state-space RUL for C-MAPSS from the Saxena et al. (2008) generative wear law  loss(t) ~ exp(a t^b).
Normalised wear coordinate:  omega(s; T) = (exp(L (s/T)^b) - 1) / (exp(L) - 1),   omega(T)=1  (fleet-level shape L, b; engine-level life T).
1. HI network g: sensor-window features (no time feature) -> hhat_t, trained on training engines to regress omega(t; T_i) with known T_i.
2. Per-engine Bayesian inference of T on a grid from the WHOLE recent history (K cycles) of hhat: Gaussian likelihood (sigma, AR(1)-tempered) x lognormal prior of fleet lives.
3. Posterior over RUL = T - t -> median / quantiles.
Primary configuration fixed in advance: L=3, b=1.5, K=60.
usage: python ssm_physics.py FD001 [L] [b]"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, time, numpy as np, torch, torch.nn as nn
DS = sys.argv[1] if len(sys.argv) > 1 else "FD001"; LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0; B = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
sys.argv = ["x", DS]
import cmapss_train as C
from multiprocessing import Pool
torch.set_num_threads(1)
data = C.data; ids = C.ids; K, SEEDS, EPOCHS = 60, 3, 250
def omega(s, T): return (np.exp(LAM * (s / T) ** B) - 1.0) / (np.exp(LAM) - 1.0)

def train_g(a):
    tr, seed = a; torch.manual_seed(seed)
    Xs = np.vstack([data[e][0][:, :-1] for e in tr]); mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    y = np.concatenate([omega(np.arange(1, len(data[e][1]) + 1), len(data[e][1])) for e in tr])
    X = torch.tensor((Xs - mu) / sd, dtype=torch.float32); Y = torch.tensor(y, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)); opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    for _ in range(EPOCHS): opt.zero_grad(); ((net(X).squeeze(-1) - Y) ** 2).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad(): return {e: net(torch.tensor((data[e][0][:, :-1] - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy() for e in ids}

def posterior(h, t, mu_lt, sd_lt, sig, kap):
    """Posterior over RUL r=1..RMAX given hhat[0..t] (cycle index s=1..t+1). Returns grid pdf over r."""
    RMAX = 450; r = np.arange(1, RMAX + 1); Tg = (t + 1) + r                         # life T in cycles (t+1 = current cycle number)
    lo = max(0, t + 1 - K); s = np.arange(lo + 1, t + 2)[:, None]                  # cycle numbers in window
    W = omega(s, Tg[None, :]); res = h[lo:t + 1][:, None] - W
    ll = -0.5 * kap * np.sum(res ** 2, 0) / sig ** 2
    lp = -0.5 * ((np.log(Tg) - mu_lt) / sd_lt) ** 2 - np.log(Tg)
    z = ll + lp; z -= z.max(); p = np.exp(z); return r, p / p.sum()

if __name__ == "__main__":
    t0 = time.time(); perm = np.random.default_rng(0).permutation(ids); folds = [list(perm[i::5]) for i in range(5)]
    jobs = [(f, [e for e in ids if e not in folds[f]], s) for f in range(5) for s in range(SEEDS)]
    H = {}
    with Pool(4) as p:
        for (f, tr, s), out in zip(jobs, p.imap(train_g, [(j[1], j[2]) for j in jobs])):
            for e in folds[f]: H.setdefault(e, []).append(out[e])
    res = {}; fit_r2 = []
    for f in range(5):
        tr = [e for e in ids if e not in folds[f]]
        # noise model from TRAIN engines is optimistic (in-sample); use held-out fold residual scale from OTHER folds is unavailable here -> use in-fold-train residuals inflated x1.5 (documented)
        Hm = {e: np.mean(H[e], 0) for e in folds[f]}
        trlife = np.array([len(data[e][1]) for e in tr]); mu_lt, sd_lt = np.log(trlife).mean(), np.log(trlife).std()
        # residual scale / AR(1) from the held-out fold itself would leak; use the OTHER folds' held-out predictions instead
        others = [e for e in ids if e not in folds[f] and e in H]
        r_all = np.concatenate([np.mean(H[e], 0) - omega(np.arange(1, len(data[e][1]) + 1), len(data[e][1])) for e in others])
        r_lag = [np.corrcoef((np.mean(H[e], 0) - omega(np.arange(1, len(data[e][1]) + 1), len(data[e][1])))[:-1], (np.mean(H[e], 0) - omega(np.arange(1, len(data[e][1]) + 1), len(data[e][1])))[1:])[0, 1] for e in others]
        sig = r_all.std(); rho = float(np.clip(np.mean(r_lag), 0, 0.98)); kap = (1 - rho) / (1 + rho)
        for e in folds[f]:
            h = Hm[e]; T = len(h); med = np.empty(T); q10 = np.empty(T)
            for t in range(T):
                r, p = posterior(h, t, mu_lt, sd_lt, sig, kap); c = np.cumsum(p); med[t] = r[np.searchsorted(c, 0.5)]; q10[t] = r[np.searchsorted(c, 0.1)]
            res[f"e{e}__rul"] = data[e][1]; res[f"e{e}__hybrid"] = np.log(med); res[f"e{e}__q10"] = np.log(q10); res[f"e{e}__hh"] = h
        print(f"fold {f}: sigma {sig:.3f} rho {rho:.2f}", flush=True)
    base = np.load(f"{WORK}/oof_cmapss_{DS}.npz")
    for e in ids: res[f"e{e}__data"] = base[f"e{e}__data"]
    tag = f"L{LAM:g}_b{B:g}"; np.savez(f"{WORK}/oof_cmapss_ssm{tag}_{DS}.npz", **res)
    err = {"mlp": [], "ssm": []}
    for e in ids:
        r = data[e][1]; m = (r <= 100) & (np.arange(len(r)) >= 25)
        err["mlp"].append(np.abs(res[f"e{e}__data"][m] - np.log(r[m]))); err["ssm"].append(np.abs(res[f"e{e}__hybrid"][m] - np.log(r[m])))
    print(f"{DS} L={LAM} b={B}: time {round(time.time()-t0)} s | RUL<=100 region median|log err| mlp {np.median(np.concatenate(err['mlp'])):.3f} ssm {np.median(np.concatenate(err['ssm'])):.3f} | mean mlp {np.concatenate(err['mlp']).mean():.3f} ssm {np.concatenate(err['ssm']).mean():.3f}")
