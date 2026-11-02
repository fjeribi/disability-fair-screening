"""Exploratory item interactions with stratified PSU sandwich uncertainty and BH adjustment."""
import os,sys,pathlib,numpy as np,pandas as pd,statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
r=pathlib.Path(__file__).resolve().parents[1];os.chdir(r);sys.path.insert(0,'code');from fairmetrics import DOMS
rows=[]
for scale,stem,k in [('dep','phq8',8),('anx','gad7',7)]:
 d=pd.read_parquet(f'data/proc/ml_{scale}.parquet');d=d[d.split=='train'];items=[stem+str(i) for i in range(1,k+1)]
 for domain in DOMS:
  t=d[(d.disab==0)|(d['d_'+domain]==1)].copy();D=t['d_'+domain].to_numpy();X=t[items].copy();X['D']=D
  for it in items:X[it+'xD']=t[it]*D
  X=sm.add_constant(X);xx=X.to_numpy();w=t.wt_pooled.to_numpy();w=w/w.mean();y=t.yA.to_numpy();fit=sm.GLM(y,X,family=sm.families.Binomial(),freq_weights=w).fit();p=np.asarray(fit.predict());bread=xx.T@((w*p*(1-p))[:,None]*xx);scores=xx*(w*(y-p))[:,None]
  ss=pd.DataFrame(scores);ss['stratum']=t.stratum.to_numpy();ss['psu']=t.psu.to_numpy();u=ss.groupby(['stratum','psu']).sum();meat=np.zeros_like(bread)
  for st,a in u.groupby(level=0):
   v=a.to_numpy();n=len(v)
   if n<2:raise ValueError('Singleton stratum in item inference')
   v=v-v.mean(axis=0);meat+=n/(n-1)*(v.T@v)
  inv=np.linalg.pinv(bread);se=np.sqrt(np.diag(inv@meat@inv));beta=np.asarray(fit.params)
  for it in items:
   j=list(X.columns).index(it+'xD');z=beta[j]/se[j];rows.append(dict(scale=scale,domain=domain,item=it,beta=beta[j],SE=se[j],OR_ratio=np.exp(beta[j]),lo=np.exp(beta[j]-1.96*se[j]),hi=np.exp(beta[j]+1.96*se[j]),p=2*norm.sf(abs(z)),converged=fit.converged))
  print(scale,domain,'complete',flush=True)
res=pd.DataFrame(rows);res['q_BH']=multipletests(res.p,method='fdr_bh')[1];res.to_csv('out/corrected_item_interactions.csv',index=False)
print('ITEM INTERACTIONS COMPLETE',flush=True)
