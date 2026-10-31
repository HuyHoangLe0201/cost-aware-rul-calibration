"""Paired decision comparison (countdown-constrained minus data-only model) restricted to the engines alarmed by BOTH models, so that it also covers
settings in which the two models alarm different numbers of engines (FD003 with 5 labelled engines: 95 against 91 alarmed).  Cumulative tau* policy,
P0 = 60, random stream, 150 orders, engine-level 95% bootstrap interval; negative favours the constrained model.
usage: python paired_common_alarms.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import os, sys, numpy as np
rows = []
for ds in ("FD001", "FD003"):
    for n in (5, 10, 20):
        os.environ["PFX"] = f"ss{n}_"; os.environ["BOTH"] = "1"; sys.argv = ["fleet_eval.py", ds]
        for m in [k for k in list(sys.modules) if k == "fleet_eval"]: del sys.modules[m]
        import fleet_eval as F
        line = f"{ds} n_lab={n:>2}:"
        for cu in (2, 5, 10, 20):
            Ah, eh, _ = F.run("hybrid", 60, cu, "random", 150); Ad, ed, _ = F.run("data", 60, cu, "random", 150)
            ih, idd = list(Ah), list(Ad); assert len(eh["tau_cum"]) == len(ih) and len(ed["tau_cum"]) == len(idd)
            dh = dict(zip(ih, eh["tau_cum"])); dd = dict(zip(idd, ed["tau_cum"])); common = [e for e in ih if e in dd]
            diff = np.array([dh[e] - dd[e] for e in common]); m_, lo, hi = F.boot(diff)
            line += f" c_u/c_o={cu}: {m_:+.3f} [{lo:+.3f},{hi:+.3f}] (common {len(common)}, alarmed {len(ih)} vs {len(idd)});"
        print(line, flush=True)
