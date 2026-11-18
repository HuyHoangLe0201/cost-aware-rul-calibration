"""Fixed-predictor evaluation for a sequential and a tree-based predictor (extension of prospective_eval.py, which covers the MLP).

For each of NSPLIT random 50/50 splits of the engines of a C-MAPSS fleet: (1) train the predictor on the training half only (GRU: three seeds averaged; random forest: three seeds averaged),
(2) fix it and predict the evaluation half (no engine of the evaluation fleet is seen in training), (3) alarm at P0 = 60, (4) the first N0 = 20 units of every random order of the evaluation fleet
form the initial calibration and the rest is decided sequentially with each policy; the closed-loop comparison of labelling methods is repeated.  Printed per split and as mean (and standard
deviation across the splits) of cost per alarm / P0 and coverage.
usage: python prospective_models.py FD002 gru     (or: rf)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
DS = sys.argv[1] if len(sys.argv) > 1 else "FD002"; MODEL = sys.argv[2] if len(sys.argv) > 2 else "gru"
sys.argv = ["cmapss_baselines.py", DS]
import cmapss_baselines as B
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor
from rulcost import first_alarm, evaluate_policies, simulate_closed_loop, target_coverage

C, CLIP = B.C, B.CLIP
NSPLIT, CUS = 5, (2, 5, 10, 20)


def task_rf(a):
    seed, tr, te = a
    X = np.vstack([C.data[e][0] for e in tr]); y = np.log(np.minimum(np.concatenate([C.data[e][1] for e in tr]), CLIP)); mu, sd = X.mean(0), X.std(0) + 1e-6
    mdl = RandomForestRegressor(n_estimators=80, min_samples_leaf=20, max_features=0.5, max_samples=0.5, n_jobs=1, random_state=seed).fit((X - mu) / sd, y)
    return {e: mdl.predict((C.data[e][0] - mu) / sd) for e in te}


def task_gru(a):
    seed, tr, te = a; return B.task(("gru", seed, tr, te))[1]


if __name__ == "__main__":
    rng = np.random.default_rng(100); ids = list(C.ids); acc = {cu: {} for cu in CUS}; cl = {cu: {} for cu in CUS}; nal = []; rm = []
    for sp in range(NSPLIT):
        perm = [int(x) for x in rng.permutation(ids)]; half = len(perm) // 2; train, ev = perm[:half], perm[half:]
        with Pool(3) as pool: outs = pool.map(task_gru if MODEL == "gru" else task_rf, [(s, train, ev) for s in range(3)])
        pred = {e: np.mean([o[e] for o in outs], 0) for e in ev}
        P, R = [], []
        for e in ev:
            a = first_alarm(np.exp(pred[e]), 60.0)
            if a: P.append(a[1]); R.append(float(C.data[e][1][a[0]]))
        P, R = np.array(P), np.array(R); nal.append(len(P))
        rmse = np.sqrt(np.mean(np.concatenate([(np.exp(pred[e]) - np.minimum(C.data[e][1], CLIP)) ** 2 for e in ev]))); rm.append(rmse)
        print(f"{DS} {MODEL} split {sp}: train {len(train)} / evaluation {len(ev)} engines, {len(P)} alarmed, held-out RMSE {rmse:.1f}", flush=True)
        for cu in CUS:
            rep = evaluate_policies(P, R, cu, 1.0, normalizer=60.0, n_orders=60)
            for k in ("point", "fixed90", "gauss", "static", "cum", "roll"): acc[cu].setdefault(k, []).append((rep.cost[k], rep.coverage[k]))
            res = simulate_closed_loop(P, R, cu, 1.0, normalizer=60.0, n_orders=30)
            for m, v in res.items(): cl[cu].setdefault(m, []).append((v["cost"], v["coverage"]))
    f = lambda v: f"{np.mean([a[0] for a in v]):.3f} (sd {np.std([a[0] for a in v]):.3f})/{np.mean([a[1] for a in v]):.2f}"
    print(f"=== {DS} {MODEL}: mean over {NSPLIT} splits, evaluation fleet of {int(np.mean(nal))} alarmed units on average, held-out RMSE {np.mean(rm):.1f} (range {min(rm):.1f}-{max(rm):.1f})")
    for cu in CUS:
        print(f" c_u/c_o={cu:>2} (target coverage {target_coverage(cu):.2f}):  " + " | ".join(f"{k} {f(v)}" for k, v in acc[cu].items()))
        print(f"        closed loop: " + " | ".join(f"{m} {f(v)}" for m, v in cl[cu].items()))
