import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import sys, numpy as np
fn = sys.argv[1]; Z = np.load(f"{WORK}/{fn}"); ids = sorted({k.split("__")[0] for k in Z.files})
rows = []
for e in ids:
    r = Z[f"{e}__rul"]; a = Z[f"{e}__data"]; b = Z[f"{e}__hybrid"]; T = len(r); m = (r <= 100) & (np.arange(T) >= 25)
    rows.append((np.log(r[m]), a[m], b[m], r[m]))
lr = np.concatenate([x[0] for x in rows]); a = np.concatenate([x[1] for x in rows]); b = np.concatenate([x[2] for x in rows]); r = np.concatenate([x[3] for x in rows])
ea, eb = a - lr, b - lr
print(f"signed bias (log): mlp {ea.mean():+.3f}  ssm {eb.mean():+.3f} | sd: mlp {ea.std():.3f} ssm {eb.std():.3f} | corr(errors) {np.corrcoef(ea, eb)[0,1]:.2f}")
for lo, hi in ((1, 20), (20, 50), (50, 100)):
    m = (r >= lo) & (r < hi); print(f"  RUL in [{lo},{hi}): median|err| mlp {np.median(np.abs(ea[m])):.3f} ssm {np.median(np.abs(eb[m])):.3f} bias mlp {ea[m].mean():+.3f} ssm {eb[m].mean():+.3f}")
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    f = (1 - w) * a + w * b; print(f"  fusion w_ssm={w:.2f}: median|err| {np.median(np.abs(f - lr)):.3f} mean {np.abs(f - lr).mean():.3f}")
# engine-level: error of life estimate (bias per engine) -> is the SSM error persistent per engine?
pe_a = np.array([np.mean(x[1] - x[0]) for x in rows]); pe_b = np.array([np.mean(x[2] - x[0]) for x in rows])
print(f"per-engine mean error sd: mlp {pe_a.std():.3f} ssm {pe_b.std():.3f}")
