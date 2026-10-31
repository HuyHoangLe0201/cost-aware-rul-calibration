"""Physics-informed semi-supervised RUL: the unit-rate law RUL(t)-RUL(t+D)=D needs NO labels, so it can be enforced on
trajectories of engines that are still operating (right-censored, unlabeled).
Variants:  data      : MLP on labeled run-to-failure engines only
           kin_lab   : + kinematic loss on labeled engines only
           kin_unlab : + kinematic loss on labeled AND truncated unlabeled engines   (physics-informed semi-supervised)
Setting: FD001, 5 folds by engine (20 test engines); of the remaining 80, n_lab are labeled (full run-to-failure),
the rest are observed only up to a random fraction U(0.3,0.7) of their life and carry no labels.
usage: python semisup_physics.py FD001"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, time, numpy as np, torch, torch.nn as nn
DS = sys.argv[1] if len(sys.argv) > 1 else "FD001"
sys.argv = ["cmapss_train.py", DS]
import cmapss_train as C                      # reuses feature builder: C.data[e] = (X, true RUL), C.ids, C.CLIP, C.D
from multiprocessing import Pool
torch.set_num_threads(1)
import os
D, CLIP, EPOCHS = 10, 125, 250
LAM = float(os.environ.get("LAM", 1.0)); TLO = float(os.environ.get("TLO", 0.3)); THI = float(os.environ.get("THI", 0.7)); TAG = os.environ.get("TAG", "")
ids = C.ids; data = C.data

class Net(nn.Module):
    def __init__(s, din):
        super().__init__(); s.f = nn.Sequential(nn.Linear(din, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)); s.b = nn.Parameter(torch.tensor(4.0))
    def forward(s, x): return s.b + 2 * s.f(x).squeeze(-1)

def pairs(Xlist_lens):
    i0, off = [], 0
    for T in Xlist_lens: i0 += list(range(off, off + T - D)); off += T
    return torch.tensor(i0, dtype=torch.long)

def task(a):
    variant, fold, rep, n_lab, lab, unl, te, seed = a
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xl = np.vstack([data[e][0] for e in lab]); mu, sd = Xl.mean(0), Xl.std(0) + 1e-6
    norm = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    XL = norm(Xl); yL = torch.tensor(np.log(np.minimum(np.concatenate([data[e][1] for e in lab]), CLIP)), dtype=torch.float32)
    iL = pairs([len(data[e][1]) for e in lab]); mL = torch.tensor(np.concatenate([data[e][1][:len(data[e][1]) - D] for e in lab]) <= CLIP - D, dtype=torch.float32)
    # unlabeled, truncated engines
    ut = [(e, max(D + 5, int(rng.uniform(TLO, THI) * len(data[e][1])))) for e in unl]
    XU = norm(np.vstack([data[e][0][:n] for e, n in ut])) if ut else None; iU = pairs([n for _, n in ut]) if ut else None
    net = Net(XL.shape[1]); opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    for ep in range(EPOCHS):
        opt.zero_grad(); l = net(XL); loss = ((l - yL) ** 2).mean()
        if variant in ("kin_lab", "kin_unlab"):
            kin = ((l[iL] - torch.log(torch.exp(l[iL + D]) + D)) ** 2 * mL).sum() / mL.sum().clamp_min(1); loss = loss + LAM * kin
        if variant == "kin_unlab" and XU is not None:
            lu = net(XU); m = (torch.exp(lu[iU]).detach() < CLIP - D).float()                    # only where the (predicted) RUL is below the clip plateau
            ku = ((lu[iU] - torch.log(torch.exp(lu[iU + D]) + D)) ** 2 * m).sum() / m.sum().clamp_min(1); loss = loss + LAM * ku
        loss.backward(); opt.step()
    net.eval(); se, le = [], []
    with torch.no_grad():
        for e in te:
            r = data[e][1]; p = torch.exp(net(norm(data[e][0]))).numpy(); se.append((np.minimum(p, CLIP) - np.minimum(r, CLIP)) ** 2)
            m = (r <= 100) & (np.arange(len(r)) >= 25); le.append(np.abs(np.log(p[m] / r[m])))
    return variant, fold, rep, n_lab, float(np.sqrt(np.concatenate(se).mean())), float(np.median(np.concatenate(le)))

if __name__ == "__main__":
    t0 = time.time(); rng = np.random.default_rng(0); perm = rng.permutation(ids); folds = [list(perm[i::5]) for i in range(5)]
    jobs = []
    for f in range(5):
        pool = [e for e in ids if e not in folds[f]]
        for n_lab in (5, 10, 20, 40):
            for rep in range(2):
                r2 = np.random.default_rng(1000 * f + 10 * n_lab + rep); sel = list(r2.permutation(pool)); lab, unl = sel[:n_lab], sel[n_lab:]
                for v in ("data", "kin_lab", "kin_unlab"): jobs.append((v, f, rep, n_lab, lab, unl, folds[f], 100 * f + rep))
    R = {}
    with Pool(4) as p:
        for v, f, rep, n, rmse, mle in p.imap_unordered(task, jobs): R[(v, f, rep, n)] = (rmse, mle)
    np.save(f"{WORK}/semisup_{DS}{TAG}.npy", R, allow_pickle=True)
    def boot(x, B=4000):
        r = np.random.default_rng(0); idx = r.integers(0, len(x), (B, len(x))); s = x[idx].mean(1); return x.mean(), *np.percentile(s, [2.5, 97.5])
    print(f"{DS}{TAG} (LAM={LAM}, trunc U({TLO},{THI})) semi-supervised physics (RMSE of clipped RUL on 20 held-out engines; mean over 5 folds x 2 reps); time {round(time.time()-t0)} s")
    for n in (5, 10, 20, 40):
        keys = [(f, rep) for f in range(5) for rep in range(2)]
        g = lambda v, j: np.array([R[(v, f, rep, n)][j] for f, rep in keys])
        s = f"n_lab={n:>2}: RMSE data {g('data',0).mean():.2f} kin_lab {g('kin_lab',0).mean():.2f} kin_unlab {g('kin_unlab',0).mean():.2f}"
        m, lo, hi = boot(g('data', 0) - g('kin_unlab', 0)); s += f" | gain data-kin_unlab {m:+.2f} [{lo:+.2f},{hi:+.2f}]"
        m, lo, hi = boot(g('kin_lab', 0) - g('kin_unlab', 0)); s += f" | gain kin_lab-kin_unlab {m:+.2f} [{lo:+.2f},{hi:+.2f}]"
        s += f" | median|log err| data {g('data',1).mean():.3f} kin_unlab {g('kin_unlab',1).mean():.3f}"
        print(s)
