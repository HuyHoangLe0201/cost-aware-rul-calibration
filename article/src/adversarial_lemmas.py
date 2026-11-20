"""Adversarial tests of the lemmas that hold for ANY predictable censoring levels (log-normal ratio sigma = 0.35, tau* = 1/11, five initial labels).
A. Lemma 3(a): the statistic |Lambda_hat(y) - Lambda(y)| sqrt(R_n(y) / L_n) over all observed levels y <= x (G(x) = 0.9) with R_n(y) >= L_n = 5 log n must stay bounded
   (it is bounded by 2 sqrt6 A + 4 B0 in the lemma), for the Kaplan-Meier loop and six other predictable policies.
C. Lemma 2: the risk-set statistic |R^+ - (1 - G) J^+| / sqrt(L max(J^+, L)) of Step 2, the one-sided quantity of (i) against C1, and the band statistic of (ii) on a grid of levels.
B. Lemmas 4 and 5: the number of labels created below x at a rank below K against its bound K / (1 - G(x)), and the inequality S_u > beta u / 2 - (3 / 4 beta) log N on the uses of each label.
usage: python adversarial_lemmas.py   (a few minutes)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import numpy as np, math, sys, bisect
from statistics import NormalDist

Nn=NormalDist(); SIG=0.35; tau=1/11
G=lambda x: 0.5*(1+math.erf(math.log(x)/(SIG*math.sqrt(2)))) if x>0 else 0.0
Gv=np.vectorize(G)
qstar=math.exp(SIG*Nn.inv_cdf(tau)); xlev=math.exp(SIG*Nn.inv_cdf(0.90))     # level x with G(x) = 0.9 (beta = 0.1)

def km_quantile(Y,d,t,M):
    o=np.argsort(Y,kind="stable"); Ys,ds=Y[o],d[o]; n=len(Y)
    first=np.concatenate(([0],np.where(Ys[1:]!=Ys[:-1])[0]+1)); dd=np.add.reduceat(ds,first)
    cdf=1.0-np.cumprod(1.0-dd/(n-first)); hit=np.where((cdf>=t-1e-12)&(dd>0))[0]
    return Ys[first[hit[0]]] if hit.size else M

def policy_stream(pol,n,rng,init):
    """returns arrays Y (observed value), d (exact indicator) for n units; multipliers are predictable given the past"""
    Y=list(init); D=[1]*len(init); Qm=[np.inf]*len(init); rho=np.exp(SIG*rng.standard_normal(n))
    for s in range(len(init),n):
        Ya=np.array(Y); Da=np.array(D,float)
        if pol=="KM": q=km_quantile(Ya,Da,min(tau*(s+1)/s,1.0),1.0)
        elif pol=="fixed_low": q=qstar-0.1
        elif pol=="uniform": q=rng.uniform(0.3,1.2)
        elif pol=="alternate": q=0.4 if s%2==0 else 1.5
        elif pol=="max_label":                        # replace at the largest exact label below x (thinnest risk sets)
            ex=Ya[(Da==1)&(Ya<xlev)]; q=ex.max() if ex.size else 1.0
        elif pol=="third_largest":
            ex=np.sort(Ya[(Da==1)&(Ya<xlev)]); q=ex[-3] if ex.size>=3 else 1.0
        elif pol=="decreasing":                        # slowly decreasing multiplier
            q=max(0.3,1.2-0.9*s/n)
        Qm.append(q)
        if rho[s]<=q: Y.append(rho[s]); D.append(1)
        else: Y.append(q); D.append(0)
    return np.array(Y),np.array(D,float),np.array(Qm)

def stats(Y,d,n):
    L=5*math.log(n); o=np.argsort(Y,kind="stable"); Ys,ds=Y[o],d[o]
    R=n-np.searchsorted(Ys,Ys,side="left")              # R_n(y) = #{Y >= y} at each observed value
    haz=np.where(ds==1,1.0/R,0.0); Lam_hat=np.cumsum(haz)
    # NA estimator at y = Ys[i] includes jumps at all exact values <= y: ties among exact values share one R, handled by the cumulative sum over sorted order
    # (ties only occur among the initial identical labels, add all of them: take the value at the last tied index)
    last=np.searchsorted(Ys,Ys,side="right")-1; Lam_hat=Lam_hat[last]
    m=(Ys<=xlev)&(R>=L)&(ds==1)
    if not m.any(): return 0,0,0
    y=Ys[m]; Lam=-np.log(1-Gv(y)); err=np.abs(Lam_hat[m]-Lam); r=R[m]
    return float((err*np.sqrt(r/L)).max()), float((err*r/L).max()), int(m.sum())



def lemma2_stats(Y,d,Qm,n):
    """Lemma 2 on a grid of 25 levels w <= x (G(w) from 0.02 to 0.90): (i) the risk-set martingale statistic |R^+(w) - (1 - G(w)) J^+(w)| / sqrt(L max(J^+(w), L))
    (Step 2 of the proof: at most 2 + 3 sqrt(L/J) in the proof), and the one-sided quantity (beta J^+/2 - R^+)/L against C1 = 2/beta + 3; (ii) the band statistic
    |M(u, v)| / sqrt(L (G(v) - G(u)) (R^+(u) + L)) over all pairs of grid levels, where M(u, v) = N(v) - N(u) - int_u^v R dLambda."""
    L=5*math.log(n); lev=np.array([Ginv_(p) for p in np.linspace(0.02,0.90,25)]); Gl=Gv(lev); Lam=-np.log(1-Gl)
    Rp=np.array([(Y>w).sum() for w in lev],float); Jp=np.array([(Qm>w).sum() for w in lev],float)
    s_i=np.abs(Rp-(1-Gl)*Jp)/np.sqrt(L*np.maximum(Jp,L)); one=((1-0.90)*Jp/2-Rp)/L
    # N(w) = number of exact values <= w ; int_0^w R dLambda = sum_s Lambda(min(Y_s, w))
    Nw=np.array([((Y<=w)&(d==1)).sum() for w in lev],float); Iw=np.array([np.minimum(-np.log(1-Gv(np.minimum(Y,w))),Lam[i]).sum() for i,w in enumerate(lev)])
    Mw=Nw-Iw; worst=0.0
    for i in range(len(lev)):
        for j in range(i+1,len(lev)):
            worst=max(worst,abs(Mw[j]-Mw[i])/math.sqrt(L*(Gl[j]-Gl[i])*(Rp[i]+L)))
    return float(s_i.max()),float(one.max()),worst
Ginv_=lambda p: math.exp(SIG*Nn.inv_cdf(p))

X=xlev; beta=1-0.90
def run_B(pol,n,rng,K):
    Y=[0.4*qstar]*5; D=[1]*5; rho=np.exp(SIG*rng.standard_normal(n)); A=0; uses={}   # uses[label] -> list of censored flags in order
    Ysort=sorted(Y,reverse=True)
    for j in range(5,n):
        # K-th largest of the previous Y values
        Yk=Ysort[K-1] if len(Ysort)>=K else 0.0
        ex=[y for y,d in zip(Y,D) if d==1 and y<X]
        if pol=="just_above_U":       # replace just above the K-th largest observed value (thin region)
            cand=[y for y in ex if y>Yk]; q=min(cand) if cand else X
        elif pol=="max_label": q=max(ex) if ex else X
        elif pol=="top_K": 
            cand=sorted([y for y in ex if y>Yk]); q=cand[len(cand)//2] if cand else X
        elif pol=="x": q=X
        elif pol=="low": q=qstar-0.1
        else: raise ValueError
        r=rho[j]
        if Yk<r<=min(q,X): A+=1
        if r<=q: y=r; d=1
        else: y=q; d=0
        # record uses (labels are exact observed values below X)
        if q<X and q in set(ex): uses.setdefault(q,[]).append(0 if d==1 else 1)   # censored flag = 1 when rho>q
        Y.append(y); D.append(d)
        ys=[-v for v in Ysort]; k=bisect.bisect_left(ys,-y); Ysort.insert(k,y)
    return A,uses

if __name__ == "__main__":
    print("A. Lemma 3(a): maximum over levels of |Lambda_hat - Lambda| sqrt(R/L) (S1) and of |Lambda_hat - Lambda| R/L (S2, not bounded by the lemma)")
    runs = 20; rng = np.random.default_rng(11)
    for pol in ("KM", "fixed_low", "uniform", "alternate", "max_label", "third_largest", "decreasing"):
        out = []
        for n in (500, 2000, 5000):
            r_ = int(max(10, runs*(500/n)**0.5)); S1 = []; S2 = []
            for _ in range(r_):
                init = [0.4*qstar]*5 if pol == "KM" else list(np.exp(SIG*rng.standard_normal(5)))
                Y, d, Qm = policy_stream(pol, n, rng, init); a, b, c = stats(Y, d, n); S1.append(a); S2.append(b)
            out.append(f"n={n}: max S1 {max(S1):.2f}, max S2 {max(S2):.2f} [{r_} runs]")
        print(f"   {pol:14s} " + " | ".join(out), flush=True)
    print("C. Lemma 2 (n = 2000, 10 runs per policy): max over 25 levels and runs of the risk-set statistic of (i) and of the one-sided quantity (beta J/2 - R)/L (must be <= C1 = 2/beta + 3 = 23), and of the band statistic of (ii)")
    rng = np.random.default_rng(31)
    for pol in ("KM", "fixed_low", "uniform", "alternate", "max_label", "third_largest", "decreasing"):
        S3 = []; S5 = []; S4 = []
        for _ in range(10):
            init = [0.4*qstar]*5 if pol == "KM" else list(np.exp(SIG*rng.standard_normal(5)))
            Y, d, Qm = policy_stream(pol, 2000, rng, init); a, b, c = lemma2_stats(Y, d, Qm, 2000); S3.append(a); S5.append(b); S4.append(c)
        print(f"   {pol:14s} risk-set statistic {max(S3):.2f}, one-sided quantity {max(S5):.2f}, band statistic {max(S4):.2f}", flush=True)
    print("B. Lemmas 4 and 5 (G(x) = 0.9, beta = 0.1, n = 1000, 10 runs)")
    runs = 10; n = 1000; rng = np.random.default_rng(21)
    for pol in ("just_above_U", "max_label", "top_K", "x", "low"):
        for K in (5, 20):
            As = []; worst5 = 1e9; maxuse = 0
            for _ in range(runs):
                A, uses = run_B(pol, n, rng, K); As.append(A)
                for z, fl in uses.items():
                    S = np.cumsum(fl); u = np.arange(1, len(fl) + 1)
                    worst5 = min(worst5, float((S - beta*u/2 + 3/(4*beta)*math.log(n)).min())); maxuse = max(maxuse, len(fl))
            w5 = "no use" if worst5 > 1e8 else f"{worst5:.1f}"
            print(f"   {pol:13s} K={K:2d}: mean #A_j(K) {np.mean(As):6.2f}, max {max(As):3d}, Lemma 4 bound K/beta = {K/beta:.0f}; Lemma 5: min over labels and u of S_u - beta u/2 + (3/4beta) log N = {w5} (must be > 0); most uses of one label {maxuse}", flush=True)
