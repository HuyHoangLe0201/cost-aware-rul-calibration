"""Compare replacement policies on a table of completed units (here: C-MAPSS FD001 alarms of an MLP predictor).

Run:  python examples/compare_policies.py [file.csv]   (columns: predicted_rul, true_rul)
"""
import csv
import sys
from pathlib import Path

import numpy as np

import rulcost as rc

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("example_units.csv")
rows = list(csv.DictReader(open(path)))
p = np.array([float(r["predicted_rul"]) for r in rows])
r = np.array([float(r["true_rul"]) for r in rows])
print(f"{len(p)} units from {path.name}\n")
for cu in (2, 20):
    print(rc.evaluate_policies(p, r, c_under=cu, c_over=1.0, normalizer=60.0), "\n")
print("Closed loop with preventive-replacement censoring, c_under/c_over = 10:")
for m, v in rc.simulate_closed_loop(p, r, 10.0, 1.0, normalizer=60.0).items():
    print(f"  {m:<13} cost {v['cost']:.3f}  coverage {v['coverage']:.3f}")
