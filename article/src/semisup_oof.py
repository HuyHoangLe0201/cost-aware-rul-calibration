"""End-to-end: physics-informed semi-supervised predictor -> OOF predictions for ALL engines -> fleet decision evaluation.
For each fold (20 test engines) and each n_lab: train (a) data-only on n_lab labeled engines, (b) data + kinematic loss on labeled + truncated unlabeled engines.
Saves oof_cmapss_ss{os.environ.get("TAG","")}{n}_{DS}.npz with __data (baseline) and __hybrid (= physics-informed semi-supervised).
usage: python semisup_oof.py FD001"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import os, sys, time, numpy as np, torch
DS = sys.argv[1] if len(sys.argv) > 1 else "FD001"
sys.argv = ["x", DS]
import semisup_physics as S
from multiprocessing import Pool
torch.set_num_threads(1)
D, CLIP, EPOCHS, LAM = S.D, S.CLIP, S.EPOCHS, 1.0; data = S.data; ids = S.ids

def train_predict(a):
    variant, fold, n_lab, lab, unl, te, seed = a
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xl = np.vstack([data[e][0] for e in lab]); mu, sd = Xl.mean(0), Xl.std(0) + 1e-6
    norm = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    XL = norm(Xl); yL = torch.tensor(np.log(np.minimum(np.concatenate([data[e][1] for e in lab]), CLIP)), dtype=torch.float32)
    iL = S.pairs([len(data[e][1]) for e in lab]); mL = torch.tensor(np.concatenate([data[e][1][:len(data[e][1]) - D] for e in lab]) <= CLIP - D, dtype=torch.float32)
    ut = [(e, max(D + 5, int(rng.uniform(float(os.environ.get('TLO',0.3)), float(os.environ.get('THI',0.7))) * len(data[e][1])))) for e in unl]
    XU = norm(np.vstack([data[e][0][:n] for e, n in ut])); iU = S.pairs([n for _, n in ut])
    net = S.Net(XL.shape[1]); opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    for ep in range(EPOCHS):
        opt.zero_grad(); l = net(XL); loss = ((l - yL) ** 2).mean()
        if variant == "phys":
            kin = ((l[iL] - torch.log(torch.exp(l[iL + D]) + D)) ** 2 * mL).sum() / mL.sum().clamp_min(1); lu = net(XU)
            m = (torch.exp(lu[iU]).detach() < CLIP - D).float(); ku = ((lu[iU] - torch.log(torch.exp(lu[iU + D]) + D)) ** 2 * m).sum() / m.sum().clamp_min(1)
            loss = loss + LAM * (kin + ku)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): out = {e: net(norm(data[e][0])).numpy() for e in te}
    return variant, n_lab, out

if __name__ == "__main__":
    t0 = time.time(); perm = np.random.default_rng(0).permutation(ids); folds = [list(perm[i::5]) for i in range(5)]; jobs = []
    for n_lab in (5, 10, 20):
        for f in range(5):
            pool = [e for e in ids if e not in folds[f]]; r2 = np.random.default_rng(1000 * f + 10 * n_lab); sel = list(r2.permutation(pool)); lab, unl = sel[:n_lab], sel[n_lab:]
            for v in ("data", "phys"):
                for sd_ in range(2): jobs.append((v, f, n_lab, lab, unl, folds[f], 100 * f + sd_))
    acc = {}
    with Pool(4) as p:
        for v, n, out in p.imap_unordered(train_predict, jobs):
            for e, pr in out.items(): acc.setdefault((v, n, e), []).append(pr)
    for n in (5, 10, 20):
        res = {}
        for e in ids:
            res[f"e{e}__rul"] = data[e][1]; res[f"e{e}__data"] = np.mean(acc[("data", n, e)], 0); res[f"e{e}__hybrid"] = np.mean(acc[("phys", n, e)], 0)
        np.savez(f"{WORK}/oof_cmapss_ss{os.environ.get("TAG","")}{n}_{DS}.npz", **res)
    print("done", round(time.time() - t0), "s")
