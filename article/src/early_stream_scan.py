"""Scan of n * E[excess] / (kappa tau*(1-tau*) / (2 g(q*))) for the conformal quantile of a log-normal(0, 0.35) ratio, n = 200 ... 460 and 500, 600, 800, 1000, 1200, c_u/c_o = 2, 5, 10, 20 (exact form of Theorem 1; Remark 4).
usage: python early_stream_scan.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import math, numpy as np
SIG=0.35; erf=np.vectorize(math.erf)
G=lambda x: 0.5*(1+erf(np.log(np.maximum(x,1e-300))/(SIG*math.sqrt(2))))
GRID=np.linspace(1e-4,12,200001); GG=G(GRID); dens=lambda x: np.exp(-0.5*(np.log(x)/SIG)**2)/(x*SIG*math.sqrt(2*math.pi))
Ginv=lambda p: float(np.interp(p,GG,GRID))
def bcdf(kk,n,pr):
    pr=np.clip(np.asarray(pr,float),1e-300,1-1e-16); tot=np.zeros_like(pr)
    for j in range(kk+1): tot+=np.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)+j*np.log(pr)+(n-j)*np.log1p(-pr))
    return np.minimum(tot,1.0)
vv=np.linspace(1e-6,1-1e-6,40001); Qv=np.interp(vv,GG,GRID); Qp=1/dens(Qv)
for cu in (2,5,10,20):
    tau=1/(1+cu); kap=cu+1; const=kap*tau*(1-tau)/(2*float(dens(Ginv(tau)))); dev=[]
    for n in list(range(200,461))+[500,600,800,1000,1200]:
        k=int(math.ceil((n+1)*tau)); T=np.where(vv>tau,bcdf(k-1,n,vv),1-bcdf(k-1,n,vv)); ex=kap*np.trapezoid(np.abs(vv-tau)*T*Qp,vv); dev.append((n,n*ex/const-1))
    dd=np.array([d for _,d in dev]); ns=np.array([n for n,_ in dev]); worst=int(np.argmax(np.abs(dd))); bad3=ns[np.abs(dd)>0.03]; bad2=ns[np.abs(dd)>0.02]
    print(f"cu/co={cu:2d}: max deviation over the n tested: {100*dd[worst]:+.2f}% at n={ns[worst]}; last n with |dev|>3%: {bad3.max() if len(bad3) else None}; last n with |dev|>2%: {bad2.max() if len(bad2) else None}; at n=210: {100*dd[10]:+.2f}%, n=420: {100*dd[220]:+.2f}%")
