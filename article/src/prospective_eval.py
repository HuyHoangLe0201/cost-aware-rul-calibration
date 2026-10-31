"""Prospective-style evaluation: one fixed predictor trained on a training fleet, then a sequential closed-loop stream on a disjoint evaluation fleet.
For each of NSPLIT random 50/50 splits of the engines of a C-MAPSS fleet: (1) train the MLP predictor (3 seeds averaged, as in the article) on the training half only;
(2) fix it and predict the evaluation half (no out-of-fold fitting, no engine of the evaluation fleet is seen in training); (3) alarm at P0 = 60;
(4) the first N0 = 20 units of every random order of the evaluation fleet form the initial calibration, the rest is decided sequentially with each policy,
and the closed-loop comparison of labelling methods is repeated.
usage: python prospective_eval.py FD002"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
DS = sys.argv[1] if len(sys.argv) > 1 else "FD002"
sys.argv = ["cmapss_train.py", DS]
import cmapss_train as C
from multiprocessing import Pool
from rulcost import first_alarm, evaluate_policies, simulate_closed_loop, target_coverage

NSPLIT, CUS = 5, (2, 5, 10, 20)
if __name__ == "__main__":
    rng = np.random.default_rng(100); ids = list(C.ids); acc = {cu: {} for cu in CUS}; cl = {cu: {} for cu in CUS}; nal = []
    for sp in range(NSPLIT):
        perm = [int(x) for x in rng.permutation(ids)]; half = len(perm) // 2; train, ev = perm[:half], perm[half:]
        with Pool(3) as pool: outs = pool.map(C.task, [(0, False, s, train, ev) for s in range(C.SEEDS)])
        pred = {e: np.mean([o[3][e] for o in outs], 0) for e in ev}
        P, R = [], []
        for e in ev:
            a = first_alarm(np.exp(pred[e]), 60.0)
            if a: P.append(a[1]); R.append(float(C.data[e][1][a[0]]))
        P, R = np.array(P), np.array(R); nal.append(len(P)); rmse = np.sqrt(np.mean(np.concatenate([(np.exp(pred[e]) - np.minimum(C.data[e][1], C.CLIP)) ** 2 for e in ev])))
        print(f"{DS} split {sp}: train {len(train)} / evaluation {len(ev)} engines, {len(P)} alarmed, held-out RMSE {rmse:.1f}", flush=True)
        for cu in CUS:
            rep = evaluate_policies(P, R, cu, 1.0, normalizer=60.0, n_orders=60)
            for k in ("point", "fixed90", "gauss", "static", "cum", "roll"):
                acc[cu].setdefault(k, []).append((rep.cost[k], rep.coverage[k]))
            res = simulate_closed_loop(P, R, cu, 1.0, normalizer=60.0, n_orders=30)
            for m, v in res.items(): cl[cu].setdefault(m, []).append((v["cost"], v["coverage"]))
    print(f"=== {DS}: mean over {NSPLIT} splits, evaluation fleet of {int(np.mean(nal))} alarmed units on average")
    for cu in CUS:
        print(f" c_u/c_o={cu:>2} (target coverage {target_coverage(cu):.2f}):  " + " | ".join(f"{k} {np.mean([a[0] for a in v]):.3f}/{np.mean([a[1] for a in v]):.2f}" for k, v in acc[cu].items()))
        print(f"        closed loop: " + " | ".join(f"{m} {np.mean([a[0] for a in v]):.3f}/{np.mean([a[1] for a in v]):.2f}" for m, v in cl[cu].items()))
