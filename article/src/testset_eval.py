"""Accuracy of the seven predictors in the protocol of the C-MAPSS literature: train on the whole training set, predict the official test set, report the RMSE and the PHM08 score of the
predicted RUL at the last observed cycle of every test engine against the RUL files RUL_FD00x.txt.  Used only to place the predictors of the article among published ones (the article itself uses
out-of-fold predictions on the training trajectories, because the decision experiments need the true life of every unit).  Same features, targets (log RUL clipped at 125), architectures and
seeds as cmapss_train.py / cmapss_baselines.py / cmapss_cnn.py; the Paris-law hybrid is not included (it is a variant of the MLP).
usage: python testset_eval.py FD001"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
DS = sys.argv[1] if len(sys.argv) > 1 else "FD001"
sys.argv = ["x", DS]
import torch, torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import cmapss_train as C
import cmapss_cnn as K
import cmapss_baselines as B
torch.set_num_threads(2)
ROOT, CLIP, W10 = C.ROOT, 125, 10

# ---- test engines, preprocessed with the statistics of the training set ----
tr = np.loadtxt(f"{ROOT}train_{DS}.txt"); te = np.loadtxt(f"{ROOT}test_{DS}.txt"); rul_true = np.loadtxt(f"{ROOT}RUL_{DS}.txt")
sets_tr, sens_tr = tr[:, 2:5], tr[:, 5:]; keep = sens_tr.std(0) > 1e-6; sens_tr = sens_tr[:, keep]
k = 1 if DS in ("FD001", "FD003") else 6
km = KMeans(k, n_init=5, random_state=0).fit(sets_tr) if k > 1 else None
cond_tr = km.predict(sets_tr) if km is not None else np.zeros(len(tr), int)
cond_te = km.predict(te[:, 2:5]) if km is not None else np.zeros(len(te), int)
mu_c = [sens_tr[cond_tr == c].mean(0) for c in range(k)]; sd_c = [sens_tr[cond_tr == c].std(0) + 1e-9 for c in range(k)]
sens_te = te[:, 5:][:, keep]; Zte = np.empty_like(sens_te)
for c in range(k):
    m = cond_te == c; Zte[m] = (sens_te[m] - mu_c[c]) / sd_c[c]
eng_te = te[:, 0].astype(int); test_ids = sorted(set(eng_te))


def feats(z):
    T = len(z); cs = np.cumsum(np.vstack([np.zeros((1, z.shape[1])), z]), 0); now = np.empty_like(z); prev = np.empty_like(z)
    for i in range(T):
        a = max(0, i + 1 - W10); now[i] = (cs[i + 1] - cs[a]) / (i + 1 - a)
        b = max(0, i + 1 - 2 * W10); prev[i] = (cs[a] - cs[b]) / (a - b) if a > b else now[i]
    return np.hstack([now, now - prev, np.log1p(np.arange(T))[:, None]]).astype(np.float32)


def windows(z, L=K.L):
    T = len(z); pad = np.vstack([np.repeat(z[:1], L - 1, 0), z]); return np.stack([pad[i:i + L] for i in range(T)]).astype(np.float32), np.log1p(np.arange(T)).astype(np.float32)


Xte = {e: feats(Zte[eng_te == e]) for e in test_ids}; Wte = {e: windows(Zte[eng_te == e]) for e in test_ids}
last = lambda d: np.array([d[e][-1] for e in test_ids])
ids = C.ids; allids = list(ids)


def score(pred, true):
    d = pred - true; return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def report(name, logpred_last):
    p = np.exp(logpred_last); rmse = np.sqrt(np.mean((p - rul_true) ** 2)); rmse_c = np.sqrt(np.mean((p - np.minimum(rul_true, CLIP)) ** 2))
    print(f"{DS} {name:14s} RMSE {rmse:6.2f}   RMSE vs RUL capped at 125 {rmse_c:6.2f}   score {score(p, rul_true):9.0f}", flush=True)


if __name__ == "__main__":
    # MLP (3 seeds, as in cmapss_train.py): reuse its task with a pseudo test set
    for e in test_ids: C.data[1000 + e] = (Xte[e], np.ones(len(Xte[e])))
    outs = [C.task((0, False, s, allids, [1000 + e for e in test_ids]))[3] for s in range(3)]
    report("MLP", np.mean([[o[1000 + e][-1] for e in test_ids] for o in outs], 0))
    # ridge / hist-GB / random forest (as in cmapss_baselines.py)
    Xs = np.vstack([C.data[e][0] for e in allids]); y = np.log(np.minimum(np.concatenate([C.data[e][1] for e in allids]), CLIP)); mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    models = {"Ridge": Ridge(alpha=10.0), "Hist-GB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0),
              "Random forest": RandomForestRegressor(n_estimators=80, min_samples_leaf=20, max_features=0.5, max_samples=0.5, n_jobs=2, random_state=0)}
    for name, mdl in models.items():
        mdl.fit((Xs - mu) / sd, y); report(name, np.array([mdl.predict((Xte[e][-1:] - mu) / sd)[0] for e in test_ids]))
    # GRU (as in cmapss_baselines.py, 3 seeds averaged) and 1D-CNN (as in cmapss_cnn.py, 2 seeds averaged)
    Wt = torch.tensor(np.concatenate([K.data[e][0] for e in allids])); tt = torch.tensor(np.concatenate([K.data[e][1] for e in allids]))
    yt = torch.tensor(np.log(np.minimum(np.concatenate([K.data[e][2] for e in allids]), CLIP)), dtype=torch.float32); n = len(yt)
    for name, make, seeds, epochs in (("GRU", lambda S: B.GRUNet(S), 3, 20), ("1D-CNN", lambda S: K.CNN(S), 2, 25)):
        preds = []
        for s in range(seeds):
            torch.manual_seed(s); net = make(Wt.shape[2]); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
            for ep in range(epochs):
                pm = torch.randperm(n)
                for i in range(0, n, 256):
                    ix = pm[i:i + 256]; opt.zero_grad(); ((net(Wt[ix], tt[ix]) - yt[ix]) ** 2).mean().backward(); opt.step()
            net.eval()
            with torch.no_grad(): preds.append([net(torch.tensor(Wte[e][0][-1:]), torch.tensor(Wte[e][1][-1:])).numpy()[0] for e in test_ids])
        report(name, np.mean(preds, 0))
