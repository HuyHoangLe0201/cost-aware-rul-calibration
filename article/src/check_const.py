import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np
rng=np.random.default_rng(3)
Zf=np.load(WORK + "/oof_cmapss_FD002.npz"); ids=sorted({k.split("__")[0] for k in Zf.files}); rho=[]
for e in ids:
    r=Zf[f"{e}__rul"]; p=np.exp(Zf[f"{e}__data"]); T=len(r)
    ps=np.array([np.median(p[max(0,i-4):i+1]) for i in range(T)]); ix=np.where((np.arange(T)>=10)&(ps<=60))[0]
    if len(ix): rho.append(r[ix[0]]/ps[ix[0]])
rho=np.array(rho); Pop=np.sort(rho[rng.integers(0,len(rho),1_500_000)]*np.exp(0.08*rng.normal(size=1_500_000)))
for cu in (2.,10.):
    kap=cu+1; tau=1/kap; q=Pop[int(tau*len(Pop))]; h=0.02*q
    g=(np.searchsorted(Pop,q+h)-np.searchsorted(Pop,q-h))/(2*h*len(Pop))
    print(f"cu/co={cu:g}: g(q*)={g:.3f}  predicted n*E[excess] = kappa*tau(1-tau)/(2g) = {kap*tau*(1-tau)/(2*g):.3f}")
