"""Numerical corrections used by the rerun. No clinical validity is implied."""
import numpy as np
from scipy.special import logsumexp

def lca2(X,w,iters=2000,tol=1e-9,starts=5,seed=734,diagnostics=False):
 X=np.asarray(X,float);w=np.asarray(w,float)
 if X.ndim!=2 or len(X)!=len(w) or not len(X) or not np.isfinite(X).all() or not np.isin(X,[0,1]).all() or not np.isfinite(w).all() or (w<=0).any():raise ValueError('LCA requires complete binary indicators and positive finite weights')
 U,inv=np.unique(X,axis=0,return_inverse=True);W=np.bincount(inv,weights=w);W=W/W.sum();k=X.shape[1];rng=np.random.default_rng(seed);fits=[]
 for start in range(starts):
  pi=.2 if start==0 else rng.uniform(.05,.6)
  p=np.vstack([np.full(k,.1),np.full(k,.6)]) if start==0 else np.sort(rng.uniform(.02,.98,(2,k)),axis=0)
  old=-np.inf;converged=False
  for it in range(iters):
   p=np.clip(p,1e-8,1-1e-8);pi=np.clip(pi,1e-8,1-1e-8)
   lp=U@np.log(p).T+(1-U)@np.log1p(-p).T+np.log([1-pi,pi]);den=logsumexp(lp,axis=1);resp=np.exp(lp[:,1]-den);ll=float(W@den)
   if it and abs(ll-old)<tol:converged=True;break
   mass=np.array([W@(1-resp),W@resp]);p=np.vstack([(W*(1-resp))@U/mass[0],(W*resp)@U/mass[1]]);pi=mass[1];old=ll
  if p[1].mean()<p[0].mean():p=p[::-1].copy();pi=1-pi
  fits.append((ll,float(pi),p.copy(),converged,it+1))
 best=max(fits,key=lambda x:x[0]);meta={'mean_loglik':best[0],'converged':best[3],'iterations':best[4],'starts':starts,'solutions':[{'mean_loglik':a,'converged':d,'iterations':e} for a,b,c,d,e in fits]}
 return (best[1],best[2],meta) if diagnostics else (best[1],best[2])

def roc_pts(s,y,w):
 s=np.asarray(s,float);y=np.asarray(y,float);w=np.asarray(w,float)
 if not np.isfinite(s).all() or np.sum(w*y)<=0 or np.sum(w*(1-y))<=0:raise ValueError('ROC requires finite scores and both weighted classes')
 o=np.argsort(-s,kind='stable');s,y,w=s[o],y[o],w[o];last=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
 return np.r_[np.nextafter(s[0],np.inf),s[last]],np.r_[0,np.cumsum(w*y)[last]/np.sum(w*y)],np.r_[0,np.cumsum(w*(1-y))[last]/np.sum(w*(1-y))]

def rate(y,q,w):
 y=np.asarray(y,float);q=np.asarray(q,float);w=np.asarray(w,float)
 return (float(np.sum(w*y*q)/np.sum(w*y)),float(np.sum(w*(1-y)*q)/np.sum(w*(1-y))),float(np.sum(w*q)/np.sum(w)))

def bootstrap_weights(t,rng):
 """Rao-Wu rescaled PSU bootstrap; singleton strata fail explicitly."""
 factor={}
 for st,group in t.groupby('stratum',sort=True):
  ps=group.psu.unique();n=len(ps)
  if n<2:raise ValueError('Singleton stratum '+str(st))
  chosen=rng.choice(ps,n-1,replace=True);u,c=np.unique(chosen,return_counts=True)
  factor.update({p:float(n/(n-1)*count) for p,count in zip(u,c)})
 return t.psu.map(factor).fillna(0).to_numpy()*t.wt_pooled.to_numpy()
