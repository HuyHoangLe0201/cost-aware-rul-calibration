"""Does restricting the physics comparison (Figure 9c) to the engines alarmed by both models bias it?

An engine whose predicted RUL never falls to P0 = 60 raises no alarm, so it receives no replacement date and fails unplanned; its cost is not
defined by the cost model.  Comparing the countdown-constrained model with the data-only model on the common alarmed engines therefore ignores the
engines that only one of them alarms.  This script reports, for the six settings of Figure 9c (FD001 and FD003 with 5, 10 and 20 labelled engines):
  - the numbers of engines alarmed by each model, by both and by neither;
  - when the two alarm sets differ, the paired difference over the WHOLE fleet, constrained minus data-only, if an engine that raises no alarm is charged a penalty pi
    (cost per alarm divided by P0): the engines alarmed by both models contribute their cost difference, an engine alarmed only by the constrained model contributes
    cost_constrained - pi, an engine alarmed only by the data-only model pi - cost_data, and engines alarmed by neither cost the same under both models (difference 0).
    The mean over the fleet and its 95% engine-level bootstrap interval are printed for pi = 0, 1, 2, 5, together with the break-even penalty at which the mean is zero.
Cumulative tau* policy, P0 = 60, random stream, 150 orders (as in paired_common_alarms.py); negative favours the constrained model.
usage: python alarm_coverage_check.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import os, sys, numpy as np

for ds in ("FD001", "FD003"):
    for n in (5, 10, 20):
        os.environ["PFX"] = f"ss{n}_"; os.environ["BOTH"] = "1"; sys.argv = ["fleet_eval.py", ds]
        for m in [k for k in list(sys.modules) if k == "fleet_eval"]: del sys.modules[m]
        import fleet_eval as F
        out = []
        for cu in (2, 5, 10, 20):
            Ah, eh, _ = F.run("hybrid", 60, cu, "random", 150); Ad, ed, _ = F.run("data", 60, cu, "random", 150)
            ih, idd = list(Ah), list(Ad); dh = dict(zip(ih, eh["tau_cum"])); dd = dict(zip(idd, ed["tau_cum"]))
            common = [e for e in ih if e in dd]; only_d = [e for e in idd if e not in dh]; only_h = [e for e in ih if e not in dd]
            neither = F.N - len(set(ih) | set(idd))
            sd = sum(dh[e] - dd[e] for e in common)
            k = len(only_d) + len(only_h)
            if k:
                pi = (-sd + sum(dd[e] for e in only_d) - sum(dh[e] for e in only_h)) / (len(only_d) - len(only_h)) if len(only_d) != len(only_h) else float("nan")
                full = []
                for pen in (0, 1, 2, 5):
                    v = np.array([dh[e] - dd[e] for e in common] + [pen - dd[e] for e in only_d] + [dh[e] - pen for e in only_h] + [0.0] * neither)
                    m_, lo, hi = F.boot(v); full.append(f"pi={pen}: {m_:+.3f} [{lo:+.3f},{hi:+.3f}]")
                cm, clo, chi = F.boot(np.array([dh[e] - dd[e] for e in common]))
                tail = f"common {cm:+.3f} [{clo:+.3f},{chi:+.3f}]; whole fleet " + ", ".join(full) + f"; break-even penalty {pi:.2f}"
            else:
                tail = "identical alarm sets"
            out.append(f"c_u/c_o={cu}: {tail}")
        print(f"{ds} n_lab={n:>2}: fleet {F.N}, constrained alarms {len(ih)}, data-only alarms {len(idd)}, both {len(common)}, only data-only {len(only_d)}, only constrained {len(only_h)}, neither {neither} | " + "; ".join(out), flush=True)
