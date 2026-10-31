"""Additional predictor baselines on C-MAPSS (same 5-fold-by-engine split as cmapss_train.py, fold seed 0):
 ridge | hgb (HistGradientBoosting) | rf (RandomForest) | gru.  Target: log RUL clipped at 125.  Output oof_cmapss_{m}_{DS}.npz with __rul and __data.
usage: python cmapss_baselines.py FD001"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, time, numpy as np
DS = sys.argv[1]; sys.argv = ["x", DS]
import cmapss_train as C
import cmapss_cnn as K
import torch, torch.nn as nn
from multiprocessing import Pool
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
torch.set_num_threads(1)
ids, CLIP = C.ids, 125
def split():
    perm = np.random.default_rng(0).permutation(ids); return [list(perm[i::5]) for i in range(5)]
class GRUNet(nn.Module):
    def __init__(s, S):
        super().__init__(); s.g = nn.GRU(S, 64, batch_first=True); s.h = nn.Sequential(nn.Linear(65, 48), nn.ReLU(), nn.Linear(48, 1)); s.b = nn.Parameter(torch.tensor(4.0))
    def forward(s, w, t):
        o, _ = s.g(w); return s.b + s.h(torch.cat([o[:, -1], t[:, None]], 1)).squeeze(-1)
def task(a):
    m, f, tr, te = a; out = {}
    if m == "gru":
        torch.manual_seed(f); W = torch.tensor(np.concatenate([K.data[e][0] for e in tr])); tt = torch.tensor(np.concatenate([K.data[e][1] for e in tr]))
        y = torch.tensor(np.log(np.minimum(np.concatenate([K.data[e][2] for e in tr]), CLIP)), dtype=torch.float32)
        net = GRUNet(W.shape[2]); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4); n = len(y)
        for ep in range(20):
            pm = torch.randperm(n)
            for i in range(0, n, 256):
                ix = pm[i:i + 256]; opt.zero_grad(); ((net(W[ix], tt[ix]) - y[ix]) ** 2).mean().backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for e in te: out[e] = net(torch.tensor(K.data[e][0]), torch.tensor(K.data[e][1])).numpy()
        return m, out
    X = np.vstack([C.data[e][0] for e in tr]); y = np.log(np.minimum(np.concatenate([C.data[e][1] for e in tr]), CLIP)); mu, sd = X.mean(0), X.std(0) + 1e-6
    if m == "ridge": mdl = Ridge(alpha=10.0)
    elif m == "hgb": mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    else: mdl = RandomForestRegressor(n_estimators=80, min_samples_leaf=20, max_features=0.5, max_samples=0.5, n_jobs=1, random_state=0)
    mdl.fit((X - mu) / sd, y)
    for e in te: out[e] = mdl.predict((C.data[e][0] - mu) / sd)
    return m, out
if __name__ == "__main__":
    t0 = time.time(); folds = split(); jobs = [(m, f, [e for e in ids if e not in folds[f]], folds[f]) for m in ("ridge", "hgb", "rf", "gru") for f in range(5)]; acc = {}
    with Pool(4) as p:
        for m, out in p.imap_unordered(task, jobs):
            for e, v in out.items(): acc[(m, e)] = v
    for m in ("ridge", "hgb", "rf", "gru"):
        res = {}
        for e in ids: res[f"e{e}__rul"] = C.data[e][1]; res[f"e{e}__data"] = acc[(m, e)]
        np.savez(f"{WORK}/oof_cmapss_{m}_{DS}.npz", **res)
        err = np.concatenate([np.exp(res[f"e{e}__data"]) - np.minimum(res[f"e{e}__rul"], CLIP) for e in ids]); print(DS, m, "RMSE %.1f" % np.sqrt(np.mean(err ** 2)), flush=True)
    print("time", round(time.time() - t0), "s", flush=True)
