"""Exact counts for the repeated fold partitions of Section 5.1: gain of the cumulative tau* policy over the fixed 90% rule (MLP, random stream, P0 = 60),
engine-level 95% bootstrap interval printed with five decimals, for the main partition and the two further partitions (prefixes fs1_ and fs2_).
The same run also prints the cost of the rolling policy (last 40 units) minus that of the cumulative policy (Section 5.5, random stream).
usage: python fold_gain_check.py            (runs the 12 fleet-partition pairs; each pair is one call of fleet_eval.run for four cost ratios)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, os, subprocess
if len(sys.argv) == 1:
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        for pf in ("main", "fs1_", "fs2_"):
            r = subprocess.run([sys.executable, os.path.abspath(__file__), pf, ds], capture_output=True, text=True, check=True)
            sys.stdout.write(r.stdout); sys.stdout.flush()
else:
    pfx, ds = sys.argv[1], sys.argv[2]
    os.environ["PFX"] = "" if pfx == "main" else pfx; sys.argv = ["x", ds]
    import fleet_eval as F
    for cu in (2, 5, 10, 20):
        A_, em, cv = F.run("data", 60, cu, "random", 150)
        m, lo, hi = F.boot(em["fixed90"] - em["tau_cum"])
        print(f"{pfx} {ds} cu/co={cu:>2} gain {m:+.5f} [{lo:+.5f},{hi:+.5f}] excludes zero: {'yes' if lo > 0 or hi < 0 else 'no'}", flush=True)
        m, lo, hi = F.boot(em["tau_roll"] - em["tau_cum"])
        print(f"{pfx} {ds} cu/co={cu:>2} rolling minus cumulative {m:+.5f} [{lo:+.5f},{hi:+.5f}] excludes zero: {'yes' if lo > 0 or hi < 0 else 'no'}", flush=True)
